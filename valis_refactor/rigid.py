"""
Rigid registration step: feature detection, matching, SimilarityTransform,
micro-rigid refinement, and affine warping.

Functions extracted from valis_reg.py step2.
"""

from weakref import ref

import numpy as np
from skimage import transform as sktransform

from np_img import np_img
from channel import extract_channel
from preprocessing import clahe, normalize_pair

DUMP_THUMBNAIL = True


def step2_rigid(ref_c, moving, moving_channel, ref_p, moving_p, scale=0.125, tile_wh=512):
    """Rigid registration + micro-rigid refinement.

    1. Detect features (DISK) on processed images, match (LightGlue)
    2. Filter matches, estimate SimilarityTransform → M
    3. Micro-rigid: tile full-res warped images, per-tile match → refined M
    4. Apply M to full-res moving → warped1

    Returns (warped1, M) — warped1: np_img, M: (3,3) ndarray.
    """
    from skimage import transform
    from feature_detectors import DiskFD
    from feature_matcher import LightGlueMatcher, filter_matches_tukey

    fd = DiskFD()
    matcher = LightGlueMatcher(fd)

    # Detect features on both images first (original pattern: generate_img_obj_list)
    kp1_xy, desc1 = fd.detect_and_compute(ref_p)
    kp2_xy, desc2 = fd.detect_and_compute(moving_p)

    # Match using pre-computed features (original pattern: match_img_obj_pairs)
    _, filtered_matches, _, _ = matcher.match_images(
        img1=ref_p, desc1=desc1, kp1_xy=kp1_xy,
        img2=moving_p, desc2=desc2, kp2_xy=kp2_xy,
        rotation_deg=0
    )

    matched_ref_xy = filtered_matches.matched_kp1_xy
    matched_moving_xy = filtered_matches.matched_kp2_xy

    # Tukey outlier removal
    matched_moving_xy, matched_ref_xy, _ = filter_matches_tukey(
        matched_moving_xy, matched_ref_xy,
        tform=transform.SimilarityTransform()
    )

    # Estimate SimilarityTransform: maps moving → ref (in low-res space)
    tform = transform.SimilarityTransform()
    tform.estimate(dst=matched_ref_xy, src=matched_moving_xy)
    M_low = tform.params

    # Scale M to full-resolution coordinates
    # low_pt = s * full_pt → M_full = S_ref^(-1) @ M_low @ S_moving
    s_ref = ref_p.shape[1] / ref_c.width
    s_moving = moving_p.shape[1] / moving.width
    S_moving = np.diag([s_moving, s_moving, 1])
    S_ref_inv = np.diag([1 / s_ref, 1 / s_ref, 1])
    M = S_ref_inv @ M_low @ S_moving

    if DUMP_THUMBNAIL:
        print(f"  rigid: {len(matched_ref_xy)} matches, M=\n{M}")
    M_inv=np.linalg.inv(M)
    moving = moving.affine(M_inv, oarea=(ref_c.width, ref_c.height))

    # ── micro-rigid refinement ──────────────────────────────────
    M = micro_rigid_refine(ref_c, moving, moving_channel, scale=scale, tile_wh=tile_wh)

    if DUMP_THUMBNAIL:
        print(f"M=\n{M}")
    out_w, out_h = ref_c.width, ref_c.height
    M_inv=np.linalg.inv(M)
    warped1 = moving.affine(M_inv, oarea=(out_w, out_h))

    return warped1


def micro_rigid_refine(ref_c, moving, moving_channel, scale=0.125, tile_wh=512):
    """Refine M by matching features on tiled full-resolution images.

    Warps both full-res images by current M, resizes, tiles the overlap,
    processes/normalizes/detects/matches per tile, combines matches,
    estimates a refined SimilarityTransform, and composes with M.
    """
    from feature_detectors import DiskFD
    from feature_matcher import LightGlueMatcher, filter_matches_tukey, filter_matches_ransac

    fd = DiskFD()
    matcher = LightGlueMatcher(fd)

    ref_warped = ref_c  # reference doesn't move — it IS the target space
    moving_warped = moving
    # Resize to working resolution
    moving_small = moving_warped.resize(scale)
    moving_small = extract_channel(moving_small, moving_channel)
    moving_small = np_img(moving_small, interpretation="b-w", background=0)
    ref_small = ref_warped.resize(scale)

    moving_sxy = (np.array([moving_small.height, moving_small.width])
                  / np.array([moving_warped.height, moving_warped.width]))[::-1]
    ref_sxy = (np.array([ref_small.height, ref_small.width])
               / np.array([ref_warped.height, ref_warped.width]))[::-1]

    # Determine overlap region as the bounding box of both images
    aligned_h = min(ref_small.height, moving_small.height)
    aligned_w = min(ref_small.width, moving_small.width)

    # Tile the overlap
    n_tiles_x = max(1, aligned_w // tile_wh)
    n_tiles_y = max(1, aligned_h // tile_wh)

    all_moving_xy = []
    all_ref_xy = []

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            print(f"  micro-rigid: processing tile ({tx}, {ty}) / ({n_tiles_x}, {n_tiles_y})")
            x = tx * tile_wh
            y = ty * tile_wh
            tw = min(tile_wh, aligned_w - x)
            th = min(tile_wh, aligned_h - y)

            # Extract tiles
            ref_tile = ref_small.extract_area(x, y, tw, th, level=0)
            moving_tile = moving_small.extract_area(x, y, tw, th, level=0)

            if ref_tile.max() == ref_tile.min() or moving_tile.max() == moving_tile.min():
                continue

            # Process tiles (channel, CLAHE, normalize)
            # ref_c is already single-channel, moving may be multi-channel
            ref_proc = ref_tile if ref_tile.ndim == 2 else ref_tile[..., 0]
            moving_proc = moving_tile if moving_tile.ndim == 2 else moving_tile[..., 0]

            ref_proc = clahe(ref_proc)
            moving_proc = clahe(moving_proc)
            ref_proc, moving_proc = normalize_pair(ref_proc, moving_proc)

            try:
                tkp1_xy, tdesc1 = fd.detect_and_compute(ref_proc)
                tkp2_xy, tdesc2 = fd.detect_and_compute(moving_proc)
                _, fm, _, _ = matcher.match_images(
                    img1=ref_proc, desc1=tdesc1, kp1_xy=tkp1_xy,
                    img2=moving_proc, desc2=tdesc2, kp2_xy=tkp2_xy,
                    rotation_deg=0
                )
                if fm.matched_kp1_xy.shape[0] < 3:
                    continue

                fm_moving, fm_ref, _ = filter_matches_tukey(
                    fm.matched_kp2_xy, fm.matched_kp1_xy,
                    tform=sktransform.EuclideanTransform()
                )
                if fm_ref.shape[0] < 3:
                    continue
            except Exception:
                continue

            # Add tile offset
            all_moving_xy.append(fm_moving + np.array([x, y]))
            all_ref_xy.append(fm_ref + np.array([x, y]))

    if len(all_moving_xy) < 1:
        return np.eye(3)  # micro-rigid failed to find matches; keep original M

    all_moving_xy = np.vstack(all_moving_xy)
    all_ref_xy = np.vstack(all_ref_xy)

    # Filter combined matches
    all_moving_xy, all_ref_xy, _ = filter_matches_ransac(
        all_moving_xy, all_ref_xy)
    all_moving_xy, all_ref_xy, _ = filter_matches_tukey(
        all_moving_xy, all_ref_xy,
        tform=sktransform.EuclideanTransform()
    )

    if all_ref_xy.shape[0] < 3:
        return np.eye(3)

    # Estimate refined transform in the small-image coordinate space
    small_tform = sktransform.SimilarityTransform()
    small_tform.estimate(dst=all_ref_xy, src=all_moving_xy)
    small_M = small_tform.params

    # Scale small_M to full-res coordinates
    S_moving = np.diag([moving_sxy[0], moving_sxy[1], 1])
    S_ref_inv = np.diag([1 / ref_sxy[0], 1 / ref_sxy[1], 1])
    full_M = S_ref_inv @ small_M @ S_moving

    # Compose: new_M = M @ full_M
    new_M = full_M

    if DUMP_THUMBNAIL:
        print(f"  micro-rigid: {all_ref_xy.shape[0]} combined matches")

    return new_M