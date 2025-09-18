import os, random
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def load_bsds_gt_mat(mat_path: str) -> np.ndarray:
    m = loadmat(mat_path)
    gt_list = m['groundTruth']
    boundaries = []
    for i in range(gt_list.size):
        item = gt_list[0, i] if gt_list.shape[0] == 1 else gt_list[i, 0]
        b = item['Boundaries'][0,0]
        boundaries.append(b.astype(np.float32))
    avg = np.mean(np.stack(boundaries, axis=0), axis=0)
    if avg.max() > 0:
        avg = avg / avg.max()
    return avg

class BSDSDataset(Dataset):
    def __init__(self, images_dir: str, gt_dir: str, max_images: int = -1, patch_size: Optional[int]=None):
        self.images = []
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    self.images.append(str(Path(root) / f))
        self.images.sort()
        if max_images > 0:
            self.images = self.images[:max_images]
        self.gt_dir = Path(gt_dir)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        gt_path = self.gt_dir / f"{Path(img_path).stem}.mat"

        if not gt_path.exists():
            # create empty gt
            w, h = img.size
            gt = np.zeros((h, w), dtype=np.float32)
        else:
            gt = load_bsds_gt_mat(str(gt_path))

        # convert to tensors
        img_t = TF.to_tensor(img)  # (3,H,W)
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()  # (1,H,W)

        # 🔹 resize both to a fixed size (e.g. 320×480)
        target_h, target_w = 320, 480
        img_t = TF.resize(img_t, [target_h, target_w], interpolation=TF.InterpolationMode.BILINEAR)
        gt_t = TF.resize(gt_t, [target_h, target_w], interpolation=TF.InterpolationMode.NEAREST)

        return img_t, gt_t, img_path