import os
import argparse
import csv
import numpy as np
import pydicom
import torch
import torch.nn.functional as F


def find_dicom_series(root_dir):
    series_dirs = []
    for root, _, files in os.walk(root_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) > 5:
            series_dirs.append(root)
    return series_dirs


def load_dicom_series(folder):
    slices = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)
        except Exception:
            pass

    if len(slices) == 0:
        raise ValueError(f"No valid DICOM slices found in {folder}")

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    slope = float(getattr(slices[0], "RescaleSlope", 1))

    volume = volume * slope + intercept

    pixel_spacing = slices[0].PixelSpacing
    spacing_y = float(pixel_spacing[0])
    spacing_x = float(pixel_spacing[1])

    if len(slices) > 1:
        slice_thickness = abs(
            float(slices[1].ImagePositionPatient[2])
            - float(slices[0].ImagePositionPatient[2])
        )
    else:
        slice_thickness = float(getattr(slices[0], "SliceThickness", 1.0))

    spacing = np.array([slice_thickness, spacing_y, spacing_x], dtype=np.float32)

    return volume, spacing


def resample_isotropic(volume, spacing, new_spacing=(1.0, 1.0, 1.0)):
    volume_t = torch.tensor(volume, dtype=torch.float32)[None, None]

    old_shape = np.array(volume.shape)
    resize_factor = spacing / np.array(new_spacing)
    new_shape = np.round(old_shape * resize_factor).astype(int)

    volume_t = F.interpolate(
        volume_t,
        size=tuple(new_shape.tolist()),
        mode="trilinear",
        align_corners=False
    )

    return volume_t[0, 0].numpy()


def lung_crop(volume):
    mask = volume < -400
    coords = np.argwhere(mask)

    if coords.shape[0] < 100:
        return volume

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    pad = 10
    zmin = max(zmin - pad, 0)
    ymin = max(ymin - pad, 0)
    xmin = max(xmin - pad, 0)

    zmax = min(zmax + pad, volume.shape[0])
    ymax = min(ymax + pad, volume.shape[1])
    xmax = min(xmax + pad, volume.shape[2])

    return volume[zmin:zmax, ymin:ymax, xmin:xmax]


def normalize_and_resize(volume, crop=True, output_size=(64, 64, 64)):
    volume = np.clip(volume, -1000, 400)

    if crop:
        volume = lung_crop(volume)

    volume = (volume + 1000) / 1400
    volume = volume * 2 - 1

    volume_t = torch.tensor(volume, dtype=torch.float32)[None, None]

    volume_t = F.interpolate(
        volume_t,
        size=output_size,
        mode="trilinear",
        align_corners=False
    )

    volume_t = volume_t[0].numpy()

    return volume_t.astype(np.float32)


def load_labels(labels_csv):
    labels = {}

    if labels_csv is None:
        return labels

    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("series_id") or row.get("folder") or row.get("id")
            label = row.get("label")
            if key is not None and label is not None:
                labels[key] = int(label)

    return labels


def preprocess_one_series(folder, output_dir, label=None, crop=True):
    volume, spacing = load_dicom_series(folder)
    volume = resample_isotropic(volume, spacing)
    volume = normalize_and_resize(volume, crop=crop)

    series_id = os.path.basename(folder.rstrip("/\\"))
    output_path = os.path.join(output_dir, f"{series_id}.npy")

    np.save(output_path, volume)

    return output_path, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--no_crop", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = load_labels(args.labels_csv)
    series_dirs = find_dicom_series(args.input_dir)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])

        for folder in series_dirs:
            series_id = os.path.basename(folder.rstrip("/\\"))
            label = labels.get(series_id, 0)

            try:
                output_path, label = preprocess_one_series(
                    folder,
                    args.output_dir,
                    label=label,
                    crop=not args.no_crop
                )
                writer.writerow([output_path, label])
                print(f"Saved: {output_path}, label={label}")
            except Exception as e:
                print(f"Skipped {folder}: {e}")

    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()