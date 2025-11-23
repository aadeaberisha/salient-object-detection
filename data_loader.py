import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

IMG_SIZE = 128

def read_pair(img_path, mask_path):
    img  = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img  = TF.resize(img,  (IMG_SIZE, IMG_SIZE))
    mask = TF.resize(mask, (IMG_SIZE, IMG_SIZE))

    img = TF.to_tensor(img)
    mask = (TF.to_tensor(mask) > 0.5).float()
    return img, mask

def apply_aug(img, mask):
    if random.random() < 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)

    if random.random() < 0.3:
        angle = random.uniform(-7, 7)
        img   = TF.rotate(img, angle)
        mask  = TF.rotate(mask, angle)

    if random.random() < 0.3:
        h, w = img.shape[1], img.shape[2]
        scale = random.uniform(0.9, 1.0)
        crop = max(1, int(scale * min(h, w)))

        top  = random.randint(0, max(0, h - crop))
        left = random.randint(0, max(0, w - crop))

        img  = TF.crop(img, top, left, crop, crop)
        mask = TF.crop(mask, top, left, crop, crop)

        img  = TF.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = TF.resize(mask, (IMG_SIZE, IMG_SIZE))

    if random.random() < 0.3:
        img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
        img = TF.adjust_contrast(img,  random.uniform(0.9, 1.1))

    mask = (mask > 0.5).float()
    return img, mask

class SODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, use_aug=False):
        self.imgs  = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.use_aug = use_aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, mask = read_pair(self.imgs[idx], self.masks[idx])
        if self.use_aug:
            img, mask = apply_aug(img, mask)
        return img, mask

def make_loader(root, split, batch=8, aug=False):
    img_dir  = os.path.join(root, split, "images")
    mask_dir = os.path.join(root, split, "masks")

    ds = SODDataset(img_dir, mask_dir, use_aug=aug)
    return DataLoader(ds, batch_size=batch, shuffle=(split=="train"))
