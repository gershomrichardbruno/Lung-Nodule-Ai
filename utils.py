import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def attention_to_heatmap(attention_info, batch_index=0, target_size=(64, 64, 64)):
    attn = attention_info["attention"]
    D, H, W = attention_info["grid_size"]
    window_size = attention_info["window_size"]
    shift_size = attention_info["shift_size"]

    if attn is None:
        raise ValueError("No attention stored.")

    num_windows_per_sample = (D // window_size) * (H // window_size) * (W // window_size)

    start = batch_index * num_windows_per_sample
    end = start + num_windows_per_sample

    attn = attn[start:end]

    token_importance = attn.mean(dim=1).mean(dim=1)
    token_importance = token_importance.unsqueeze(-1)

    windows = token_importance

    heatmap = windows.view(
        1,
        D // window_size,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        window_size,
        1
    )

    heatmap = heatmap.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    heatmap = heatmap.view(1, D, H, W, 1)

    if shift_size > 0:
        heatmap = torch.roll(
            heatmap,
            shifts=(shift_size, shift_size, shift_size),
            dims=(1, 2, 3)
        )

    heatmap = heatmap.permute(0, 4, 1, 2, 3)

    heatmap = F.interpolate(
        heatmap,
        size=target_size,
        mode="trilinear",
        align_corners=False
    )

    heatmap = heatmap[0, 0].detach().cpu().numpy()

    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap


def show_attention_overlay(volume, heatmap, slice_index=32, axis=0, alpha=0.45):
    if volume.ndim == 4:
        volume = volume[0]

    if axis == 0:
        ct_slice = volume[slice_index]
        heat_slice = heatmap[slice_index]
    elif axis == 1:
        ct_slice = volume[:, slice_index, :]
        heat_slice = heatmap[:, slice_index, :]
    else:
        ct_slice = volume[:, :, slice_index]
        heat_slice = heatmap[:, :, slice_index]

    plt.figure(figsize=(6, 6))
    plt.imshow(ct_slice, cmap="gray")
    plt.imshow(heat_slice, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title("3D Swin Transformer Attention Overlay")
    plt.show()


def predict_single(model, volume, device):
    model.eval()

    if isinstance(volume, np.ndarray):
        volume = torch.tensor(volume, dtype=torch.float32)

    if volume.ndim == 4:
        volume = volume.unsqueeze(0)

    volume = volume.to(device)

    with torch.no_grad():
        logits = model(volume)
        prob = torch.sigmoid(logits).item()

    return prob