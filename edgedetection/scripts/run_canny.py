import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils_bsds import list_images, match_gt_path, load_bsds_gt_mat
from src.canny_edge import run_canny

def imread_rgb(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_fig_grid(fig_path: str, img_rgb, gt_avg, canny_maps, sigmastr, pairs_str):
    cols = 2 + len(canny_maps)
    plt.figure(figsize=(2.8*cols, 3.2))
    ax = plt.subplot(1, cols, 1); ax.imshow(img_rgb); ax.set_title("Image"); ax.axis("off")
    ax = plt.subplot(1, cols, 2); ax.imshow(gt_avg, cmap="gray", vmin=0, vmax=1); ax.set_title("GT (avg)"); ax.axis("off")
    for i, m in enumerate(canny_maps, start=3):
        ax = plt.subplot(1, cols, i); ax.imshow(m, cmap="gray"); ax.set_title(f"Canny\n{pairs_str[i-3]}"); ax.axis("off")
    plt.suptitle(f"Canny Sweep — σ={sigmastr}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str)
    ap.add_argument("--gt_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.8, 1.2, 1.6, 2.0])
    ap.add_argument("--low_high_pairs", nargs="+", type=int, default=[50,150, 100,200, 150,300])
    ap.add_argument("--max_images", type=int, default=10)
    args = ap.parse_args()

    pairs = []
    nums = args.low_high_pairs
    if len(nums) % 2 != 0:
        raise ValueError("--low_high_pairs must have even count")
    for i in range(0, len(nums), 2):
        pairs.append((nums[i], nums[i+1]))
    pair_labels = [f"{lo}-{hi}" for lo, hi in pairs]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    imgs = list_images(args.images_dir)
    if args.max_images > 0:
        imgs = imgs[:args.max_images]

    for img_path in tqdm(imgs, desc="Processing images"):
        img = imread_rgb(img_path)
        gt_path = match_gt_path(args.gt_dir, img_path)
        if gt_path is None:
            print(f"[WARN] No GT found for {img_path}. Skipping figure (still saving Canny maps).")
            gt_avg = None
        else:
            gt_avg = load_bsds_gt_mat(gt_path)

        name = Path(img_path).stem
        for sigma in args.sigmas:
            canny_maps = run_canny(img, sigma=sigma, low_high_pairs=pairs)
            if gt_avg is not None:
                fig_path = Path(args.out_dir) / f"{name}_sigma{sigma:.2f}.png"
                save_fig_grid(str(fig_path), img, gt_avg, canny_maps, f"{sigma:.2f}", pair_labels)
            for idx, m in enumerate(canny_maps):
                out_bin = Path(args.out_dir) / f"{name}_sigma{sigma:.2f}_canny_{pair_labels[idx]}.png"
                cv2.imwrite(str(out_bin), m)

if __name__ == "__main__":
    main()
