"""
Non-rigid registration step: DeepFlow optical flow, displacement scaling,
and mapim warping.

Function extracted from valis_reg.py step3.
"""

import cv2
import numpy as np

from channel import extract_channel
from preprocessing import clahe, normalize_pair
from np_img import np_img


def step3_nonrigid(ref_c, moving,moving_channel, ref_p,scale_ref):
    """Non-rigid registration via optical flow.

    Compute DeepFlow backward displacement between ref_p (fixed) and a
    preprocessed version of moving. Scale displacement to full-res.
    Apply to moving via mapim → warped2.

    Returns warped2: np_img.
    """
    # 1. Resize moving to match ref_p size, then get single-channel data
    scale = scale_ref
    moving_small = moving.resize(scale)
    moving_small = extract_channel(moving_small, moving_channel)
    moving_small = np_img(moving_small, interpretation="b-w", background=0)
    moving_p = moving_small._data
    moving_p = clahe(moving_p)
    ref_norm, moving_norm = normalize_pair(ref_p.copy(), moving_p)

    # 2. Compute DeepFlow backward flow (fixed→moving)
    flow = cv2.optflow.createOptFlow_DeepFlow().calc(
        ref_norm, moving_norm, np.zeros(ref_norm.shape[0:2], dtype=np.float32)
    )
    bk_dx = flow[..., 0]
    bk_dy = flow[..., 1]
    
    # 3. Scale displacement to full-res
    s_x = moving.width / bk_dx.shape[1]
    s_y = moving.height / bk_dx.shape[0]
    full_w, full_h = moving.width, moving.height

    bk_dx = cv2.resize(bk_dx * s_x, (full_w, full_h), interpolation=cv2.INTER_CUBIC)
    bk_dy = cv2.resize(bk_dy * s_y, (full_w, full_h), interpolation=cv2.INTER_CUBIC)
    
    # 4. Build index: output pixel (x,y) → source at (x + dx, y + dy)
    index = np_img.xyz(full_w, full_h)
    
    idx_data = index._data
    idx_data[..., 0] += bk_dx
    idx_data[..., 1] += bk_dy

    # free displacement components — index already incorporated them
    del bk_dx, bk_dy, flow

    # 5. Apply via mapim
    warped2 = moving.mapim(index)
    return warped2
