import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data_loader import make_loader
from sod_model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(preds, targets, eps=1e-6):
    preds_bin = (preds > 0.5).float()
    targets   = targets.float()

    p = preds_bin.view(-1)
    t = targets.view(-1)

    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    mae = (preds - targets).abs().mean()

    return iou.item(), precision.item(), recall.item(), f1.item(), mae.item()


def evaluate_model(root, weights_path="best_unet_improved.pth", batch_size=8):

    test_loader = make_loader(root, "test", batch=batch_size, aug=False)

    model = UNet(in_channels=3, base_channels=64).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    total_iou = total_prec = total_rec = total_f1 = total_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Evaluating", ncols=80):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            iou, prec, rec, f1, mae = compute_metrics(preds, masks)

            total_iou  += iou
            total_prec += prec
            total_rec  += rec
            total_f1   += f1
            total_mae  += mae
            n_batches  += 1

    avg_iou  = total_iou / n_batches
    avg_prec = total_prec / n_batches
    avg_rec  = total_rec / n_batches
    avg_f1   = total_f1 / n_batches
    avg_mae  = total_mae / n_batches

    print("=== TEST METRICS ===")
    print(f"IoU:      {avg_iou:.4f}")
    print(f"Precision:{avg_prec:.4f}")
    print(f"Recall:   {avg_rec:.4f}")
    print(f"F1-score: {avg_f1:.4f}")
    print(f"MAE:      {avg_mae:.4f}")

    return model

def visualize_samples(root, model, n_samples=4):
    model.eval()
    
    vis_loader = make_loader(root, "test", batch=1, aug=False)
    dataset = vis_loader.dataset

    import random
    indices = random.sample(range(len(dataset)), n_samples)

    with torch.no_grad():
        for idx in indices:
            img, mask = dataset[idx]
            img = img.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE)

            pred = model(img)
            pred_bin = (pred > 0.5).float()

            img_np = img[0].permute(1, 2, 0).cpu().numpy()
            gt_np = mask[0, 0].cpu().numpy()
            pred_np = pred_bin[0, 0].cpu().numpy()

            overlay = img_np.copy()
            overlay[..., 0] = np.clip(overlay[..., 0] + pred_np * 0.5, 0, 1)

            fig, axs = plt.subplots(1, 4, figsize=(10, 3))
            axs[0].imshow(img_np); axs[0].set_title("Input"); axs[0].axis("off")
            axs[1].imshow(gt_np, cmap="gray"); axs[1].set_title("GT Mask"); axs[1].axis("off")
            axs[2].imshow(pred_np, cmap="gray"); axs[2].set_title("Pred Mask"); axs[2].axis("off")
            axs[3].imshow(overlay); axs[3].set_title("Overlay"); axs[3].axis("off")
            plt.show()

if __name__ == "__main__":
    model = evaluate_model("dataset_root", weights_path="best_unet_improved.pth")
    visualize_samples("dataset_root", model)
