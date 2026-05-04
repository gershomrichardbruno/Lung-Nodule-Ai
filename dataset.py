import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class LIDCNoduleDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.samples = []
        self.transform = transform

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("path") or row.get("npy_path")
                label = int(row["label"])
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        volume = np.load(path).astype(np.float32)

        if volume.shape == (64, 64, 64):
            volume = volume[None, ...]

        if volume.shape != (1, 64, 64, 64):
            raise ValueError(f"Expected shape (1,64,64,64), got {volume.shape}")

        volume = torch.tensor(volume, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            volume = self.transform(volume)

        return volume, label