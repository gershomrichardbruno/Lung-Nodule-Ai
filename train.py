import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import LIDCNoduleDataset
from model import LightweightSwin3D
from utils import compute_metrics, attention_to_heatmap, show_attention_overlay


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        return loss.mean()


def count_labels(dataset):
    labels = []

    for _, label in dataset:
        labels.append(int(label.item()))

    benign = labels.count(0)
    malignant = labels.count(1)

    return benign, malignant


def train_one_epoch(model, loader, criterion, optimizer, device, threshold):
    model.train()

    total_loss = 0
    all_labels = []
    all_probs = []

    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(volumes)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * volumes.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics


def validate(model, loader, criterion, device, threshold):
    model.eval()

    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            logits = model(volumes)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * volumes.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics


def visualize_attention(model, loader, device):
    model.eval()

    volumes, labels = next(iter(loader))
    volumes = volumes.to(device)

    with torch.no_grad():
        logits, attention_maps = model(volumes, return_attention=True)
        probs = torch.sigmoid(logits)

    last_attention = attention_maps[-1]

    heatmap = attention_to_heatmap(
        last_attention,
        batch_index=0,
        target_size=(64, 64, 64)
    )

    volume_np = volumes[0].detach().cpu().numpy()

    print(f"Attention sample label: {labels[0].item()}")
    print(f"Predicted malignant probability: {probs[0].item():.4f}")

    show_attention_overlay(volume_np, heatmap, slice_index=32, axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_path", default="model.pth")
    parser.add_argument("--visualize_attention", action="store_true")

    parser.add_argument(
        "--loss",
        choices=["bce", "weighted_bce", "focal"],
        default="weighted_bce"
    )

    parser.add_argument("--threshold", type=float, default=0.45)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LIDCNoduleDataset(args.csv)

    benign, malignant = count_labels(dataset)

    print("\nDataset Summary")
    print(f"Benign samples     : {benign}")
    print(f"Malignant samples  : {malignant}")
    print(f"Total samples      : {len(dataset)}")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = LightweightSwin3D(
        in_channels=1,
        embed_dim=32,
        num_heads=(2, 4, 8),
        window_size=4
    ).to(device)

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()

    elif args.loss == "weighted_bce":
        if malignant == 0:
            pos_weight = torch.tensor([1.0]).to(device)
        else:
            pos_weight = torch.tensor([benign / malignant]).to(device)

        print(f"Using weighted BCE loss with pos_weight = {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    else:
        print("Using Focal Loss")
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1
    patience = 5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.threshold
        )

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            args.threshold
        )

        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )

        print(f"Val Loss: {val_loss:.4f}")
        print(
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if args.visualize_attention:
        visualize_attention(model, val_loader, device)


if __name__ == "__main__":
    main()