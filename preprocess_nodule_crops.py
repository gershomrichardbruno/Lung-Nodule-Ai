import os
import csv
import argparse
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET


def load_dicom_series(folder):
    slices = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(".dcm"):
            try:
                ds = pydicom.dcmread(path)
                if hasattr(ds, "ImagePositionPatient"):
                    slices.append(ds)
            except:
                pass

    if len(slices) == 0:
        raise ValueError(f"No DICOM files found in {folder}")

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    slope = float(getattr(slices[0], "RescaleSlope", 1))

    volume = volume * slope + intercept

    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]

    return volume, slices, z_positions


def normalize_resize_crop(crop, output_size=(64, 64, 64)):
    crop = np.clip(crop, -1000, 400)
    crop = (crop + 1000) / 1400
    crop = crop * 2 - 1

    crop_t = torch.tensor(crop, dtype=torch.float32)[None, None]

    crop_t = F.interpolate(
        crop_t,
        size=output_size,
        mode="trilinear",
        align_corners=False
    )

    return crop_t[0].numpy().astype(np.float32)


def get_namespace(tag):
    if tag.startswith("{"):
        return tag.split("}")[0] + "}"
    return ""


def parse_xml_nodules(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = get_namespace(root.tag)

    nodules = []

    for nodule in root.iter():
        if not nodule.tag.endswith("unblindedReadNodule"):
            continue

        malignancy_scores = []
        points = []

        for child in nodule.iter():
            if child.tag.endswith("malignancy") and child.text:
                try:
                    malignancy_scores.append(int(child.text))
                except:
                    pass

            if child.tag.endswith("roi"):
                z_pos = None
                xy_points = []

                for roi_child in child:
                    if roi_child.tag.endswith("imageZposition"):
                        try:
                            z_pos = float(roi_child.text)
                        except:
                            pass

                    if roi_child.tag.endswith("edgeMap"):
                        x = None
                        y = None

                        for edge_child in roi_child:
                            if edge_child.tag.endswith("xCoord"):
                                x = int(edge_child.text)
                            if edge_child.tag.endswith("yCoord"):
                                y = int(edge_child.text)

                        if x is not None and y is not None:
                            xy_points.append((x, y))

                for x, y in xy_points:
                    if z_pos is not None:
                        points.append((z_pos, x, y))

        if len(points) > 0 and len(malignancy_scores) > 0:
            avg_malignancy = sum(malignancy_scores) / len(malignancy_scores)

            if avg_malignancy < 3:
                label = 0
            else:
                label = 1

            nodules.append({
                "points": points,
                "label": label,
                "avg_malignancy": avg_malignancy
            })

    return nodules


def z_to_index(z_value, z_positions):
    z_positions = np.array(z_positions)
    return int(np.argmin(np.abs(z_positions - z_value)))


def crop_nodule(volume, z_positions, points, margin=16):
    coords = []

    for z, x, y in points:
        zi = z_to_index(z, z_positions)
        coords.append((zi, y, x))

    coords = np.array(coords)

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    zmin = max(zmin - margin, 0)
    ymin = max(ymin - margin, 0)
    xmin = max(xmin - margin, 0)

    zmax = min(zmax + margin, volume.shape[0] - 1)
    ymax = min(ymax + margin, volume.shape[1] - 1)
    xmax = min(xmax + margin, volume.shape[2] - 1)

    crop = volume[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]

    return crop


def find_series_folders(root_dir):
    series_folders = []

    for root, _, files in os.walk(root_dir):
        dcm_count = sum(1 for f in files if f.lower().endswith(".dcm"))
        xml_count = sum(1 for f in files if f.lower().endswith(".xml"))

        if dcm_count > 5 and xml_count > 0:
            series_folders.append(root)

    return series_folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="processed_nodules")
    parser.add_argument("--margin", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")

    series_folders = find_series_folders(args.input_dir)

    print(f"Found {len(series_folders)} DICOM+XML series folders")

    total_crops = 0

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "series_id", "avg_malignancy"])

        for folder in series_folders:
            series_id = os.path.basename(folder)

            xml_files = [
                os.path.join(folder, x)
                for x in os.listdir(folder)
                if x.lower().endswith(".xml")
            ]

            if len(xml_files) == 0:
                continue

            try:
                volume, slices, z_positions = load_dicom_series(folder)

                for xml_path in xml_files:
                    nodules = parse_xml_nodules(xml_path)

                    for i, nodule in enumerate(nodules):
                        crop = crop_nodule(
                            volume,
                            z_positions,
                            nodule["points"],
                            margin=args.margin
                        )

                        if crop.size == 0:
                            continue

                        crop = normalize_resize_crop(crop)

                        out_name = f"{series_id}_nodule_{i}_label_{nodule['label']}.npy"
                        out_path = os.path.join(args.output_dir, out_name)

                        np.save(out_path, crop)

                        writer.writerow([
                            out_path,
                            nodule["label"],
                            series_id,
                            round(nodule["avg_malignancy"], 3)
                        ])

                        total_crops += 1
                        print(f"Saved crop: {out_path}, label={nodule['label']}")

            except Exception as e:
                print(f"Skipped {folder}: {e}")

    print(f"\nSaved {total_crops} nodule crops")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()