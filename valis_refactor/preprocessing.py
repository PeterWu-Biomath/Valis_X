"""
Preprocessing step: channel extraction, resize, CLAHE, normalization.

Functions extracted from valis_reg.py step1.
"""

import numpy as np
import cv2
from skimage import exposure

from channel import extract_channel as _extract_channel


def step1_preprocess(ref_c, moving_c, max_dim=1024):
    """Preprocess: resize, CLAHE, normalization.

    1. Resize to low-res (fit within max_dim) — faster to resize first
    2. Adaptive histogram equalization (CLAHE)
    3. Normalize intensities across the pair

    Input images are already single-channel (ref_c, moving_c).
    Returns (ref_p, moving_p) as small single-channel uint8 np.ndarray.
    """
    scale_ref = max_dim / max(ref_c.height, ref_c.width)
    scale_moving = max_dim / max(moving_c.height, moving_c.width)
    scale_ref = min(scale_ref, 1.0)
    scale_moving = min(scale_moving, 1.0)

    ref_p = ref_c.resize(scale_ref)._data if scale_ref < 1 else ref_c._data
    moving_p = moving_c.resize(scale_moving)._data if scale_moving < 1 else moving_c._data

    # 3. Adaptive histogram equalization
    ref_p = clahe(ref_p)
    moving_p = clahe(moving_p)

    # 4. Normalize intensities across pair
    ref_p, moving_p = normalize_pair(ref_p, moving_p)

    return ref_p, moving_p, scale_ref, scale_moving


def resize_to_max_dim(img, max_dim):
    """Resize image so largest dimension ≤ max_dim, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clahe(img):
    """Adaptive histogram equalization (CLAHE)."""
    img = exposure.rescale_intensity(img, in_range="image", out_range=(0.0, 1.0))
    img = exposure.equalize_adapthist(img)
    img = exposure.rescale_intensity(img, in_range="image", out_range=(0, 255))
    return img.astype(np.uint8)


def normalize_pair(img1, img2):
    """Normalize two images so their intensity distributions match.

    Uses histogram matching: each image is mapped to the average
    cumulative histogram of the pair.
    """
    # Compute combined CDF as target
    h1, _ = np.histogram(img1.ravel(), bins=256, range=(0, 255))
    h2, _ = np.histogram(img2.ravel(), bins=256, range=(0, 255))
    combined_hist = h1.astype(float) + h2.astype(float)
    combined_hist /= combined_hist.sum()
    target_cdf = np.cumsum(combined_hist)

    def _match(img, target_cdf):
        h, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))
        src_cdf = np.cumsum(h.astype(float) / h.sum())
        lut = np.zeros(256, dtype=np.uint8)
        t_idx = 0
        for s_idx in range(256):
            while t_idx < 255 and target_cdf[t_idx] < src_cdf[s_idx]:
                t_idx += 1
            lut[s_idx] = t_idx
        return lut[img]

    return _match(img1, target_cdf), _match(img2, target_cdf)