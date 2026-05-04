import os
import csv
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict


def extract_scores(xml_path):
    scores = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for elem in root.iter():
            if elem.tag.endswith("malignancy"):
                if elem.text:
                    try:
                        scores.append(int(elem.text))
                    except:
                        pass

    except Exception as e:
        print(f"Error reading {xml_path}: {e}")

    return scores


def get_series_id_from_path(xml_path):
    # folder name is the series id
    return os.path.basename(os.path.dirname(xml_path))


def score_to_label(avg):
    if avg <= 2.5:
        return 0
    elif avg >= 3.5:
        return 1
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", required=True)
    parser.add_argument("--output_csv", default="labels.csv")
    args = parser.parse_args()

    series_scores = defaultdict(list)

    for root, _, files in os.walk(args.xml_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)

                series_id = get_series_id_from_path(xml_path)
                scores = extract_scores(xml_path)

                if scores:
                    series_scores[series_id].extend(scores)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series_id", "label"])

        count = 0

        for sid, scores in series_scores.items():
            avg = sum(scores) / len(scores)
            label = score_to_label(avg)

            if label is None:
                continue

            writer.writerow([sid, label])
            count += 1

    print(f"Created labels.csv with {count} samples")


if __name__ == "__main__":
    main()