import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import make_loader
from sod_model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoint.pth"

def calc_iou(pred, target, eps=1e-6):
    p = (pred > 0.5).float()
    inter = (p * target).sum((1, 2, 3))
    union = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean()

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    inter = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    soft_iou = ((inter + 1e-6) / (union + 1e-6)).mean()
    return bce + 0.5 * (1 - soft_iou)

def save_checkpoint(epoch, model, opt, best, path=CKPT):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "best": best
    }, path)
    print(f"[Checkpoint] Saved: epoch={epoch}, best_val_loss={best:.4f}, path={path}")


def load_checkpoint(model, opt, path=CKPT):
    print(f"[Checkpoint] Loading from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    print(f"[Checkpoint] Loaded: epoch={ckpt['epoch']}, best_val_loss={ckpt['best']:.4f}, resuming from epoch {ckpt['epoch'] + 1}")
    return ckpt["epoch"] + 1, ckpt["best"]

def train_epoch(model, loader, opt):
    model.train()
    loss_sum = 0
    for img, m in tqdm(loader, desc="Train"):
        img, m = img.to(DEVICE), m.to(DEVICE)
        pred = model(img)
        loss = combined_loss(pred, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)

def validate_epoch(model, loader):
    model.eval()
    loss_sum = 0
    iou_sum = 0
    with torch.no_grad():
        for img, m in tqdm(loader, desc="Val", leave=False):
            img, m = img.to(DEVICE), m.to(DEVICE)
            pred = model(img)
            loss_sum += combined_loss(pred, m).item()
            iou_sum += calc_iou(pred, m).item()
    return loss_sum / len(loader), iou_sum / len(loader)

def run_training(root, save_path="best_unet.pth",
                 epochs=25, batch=16, lr=1e-3,
                 patience=5):

    train_loader = make_loader(root, "train", batch=batch, aug=True)
    val_loader   = make_loader(root, "val",   batch=batch, aug=False)

    model = UNet(3, 64).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    wait = 0
    start = 1

    if os.path.exists(CKPT):
        start, best_val = load_checkpoint(model, opt)

    for ep in range(start, epochs + 1):
        train_loss = train_epoch(model, train_loader, opt)
        val_loss, val_iou = validate_epoch(model, val_loader)
        print(f"Epoch {ep}/{epochs}: Train={train_loss:.4f} | Val={val_loss:.4f} | IoU={val_iou:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Best] Model updated: val_loss={best_val:.4f}, saved to {save_path}")
            wait = 0
        else:
            wait += 1

        save_checkpoint(ep, model, opt, best_val)

        if wait >= patience:
            print(f"[EarlyStopping] No improvement for {patience} epochs. Stopping at epoch {ep}.")
            break

    return model

if __name__ == "__main__":
    run_training("dataset_root")
