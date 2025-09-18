from typing import Tuple, Optional, Sequence
import numpy as np
import cv2

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return img
    ksize = max(3, int(6 * sigma + 1) // 2 * 2 + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def run_canny(
    img_rgb: np.ndarray,
    sigma: float,
    low_high_pairs: Sequence[Tuple[int, int]]
):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = gaussian_blur(gray, sigma)
    outs = []
    for low, high in low_high_pairs:
        e = cv2.Canny(gray, threshold1=int(low), threshold2=int(high), L2gradient=True)
        outs.append(e)
    return outs
