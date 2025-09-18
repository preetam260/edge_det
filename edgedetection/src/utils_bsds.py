import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from scipy.io import loadmat

def stem(path: str) -> str:
    return Path(path).stem

def load_bsds_gt_mat(mat_path: str) -> np.ndarray:
    """
    Load a BSDS500 groundTruth .mat file and return an averaged boundary map in [0,1].
    """
    m = loadmat(mat_path)
    if 'groundTruth' not in m:
        raise ValueError(f"groundTruth key not found in {mat_path}")
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

def match_gt_path(gt_dir: str, image_path: str) -> Optional[str]:
    s = Path(image_path).stem
    candidate = Path(gt_dir) / f"{s}.mat"
    if candidate.exists():
        return str(candidate)
    return None

def list_images(images_dir: str, exts=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) -> List[str]:
    out = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(str(Path(root) / f))
    out.sort()
    return out
