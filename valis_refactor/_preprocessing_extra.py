"""Vendored preprocessing functions needed by tile_registration."""

import numpy as np
from scipy.interpolate import Akima1DInterpolator
import pyvips
import warp_tools


def collect_img_stats(img_list, norm_percentiles=[1, 5, 95, 99], mask_list=None):
    use_masks = mask_list is not None
    if use_masks:
        use_masks = mask_list[0] is not None

    if use_masks:
        img0 = img_list[0][mask_list[0] > 0]
    else:
        img0 = img_list[0].reshape(-1)

    all_histogram, _ = np.histogram(img0, bins=256)

    n = img0.size
    total_x = img0.sum()
    for i in range(1, len(img_list)):
        img = img_list[i]
        if mask_list is None:
            img_flat = img.reshape(-1)
        else:
            if mask_list[i] is None:
                img_flat = img.reshape(-1)
            else:
                img_flat = img[mask_list[i] > 0]

        img_hist, _ = np.histogram(img_flat, bins=256)
        all_histogram += img_hist
        n += img.size
        total_x += img.sum()

    mean_x = total_x / n
    ref_cdf = 100 * np.cumsum(all_histogram) / np.sum(all_histogram)
    all_img_stats = np.array([len(np.where(ref_cdf <= q)[0]) for q in norm_percentiles])
    all_img_stats = np.hstack([all_img_stats, mean_x])
    all_img_stats = all_img_stats[np.argsort(all_img_stats)]

    return all_histogram, all_img_stats


def norm_img_stats(img, target_stats, mask=None):
    if mask is not None:
        if isinstance(mask, pyvips.Image):
            np_mask = warp_tools.vips2numpy(mask)
        else:
            np_mask = mask
    else:
        np_mask = None

    _, src_stats_flat = collect_img_stats([img], mask_list=[np_mask])

    lower_knots = np.array([0])
    upper_knots = np.array([300, 350, 400, 450])
    src_stats_flat = np.hstack([lower_knots, src_stats_flat, upper_knots]).astype(float)
    target_stats_flat = np.hstack([lower_knots, target_stats, upper_knots]).astype(float)

    eps = 100 * np.finfo(float).resolution
    eps_array = np.arange(len(src_stats_flat)) * eps
    src_stats_flat = src_stats_flat + eps_array
    target_stats_flat = target_stats_flat + eps_array

    src_order = np.argsort(src_stats_flat)
    src_stats_flat = src_stats_flat[src_order]
    target_stats_flat = target_stats_flat[src_order]

    cs = Akima1DInterpolator(src_stats_flat, target_stats_flat)

    if mask is None:
        normed_img = cs(img.reshape(-1)).reshape(img.shape)
    else:
        normed_img = img.copy()
        fg_px = np.where(np_mask > 0)
        normed_img[fg_px] = cs(img[fg_px])

    if img.dtype == np.uint8:
        normed_img = np.clip(normed_img, 0, 255)

    return normed_img
