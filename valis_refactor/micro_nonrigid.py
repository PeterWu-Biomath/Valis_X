"""
Micro non-rigid refinement step: tiled DeepFlow optical flow at full resolution.

Mirrors the tiling logic from NonRigidRegistrar.register():
  get_grid_bboxes → expand_bbox → per-tile register → stitch_tiles → mapim

Function extracted from valis_reg.py step4.
"""

import os

import cv2
import numpy as np

from channel import extract_channel
from preprocessing import clahe, normalize_pair
from np_img import np_img

from tile_registration import register_tiles

import psutil

def _mem(label=""):
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[MEM {label}] {rss:.2f} GiB", flush=True)
    return rss

TILE_WH = 512
TILE_BUFFER = 100

import gc

def step4_micro_nonrigid(ref_c, moving,moving_channel, micro_non_rigid_factor=2, n_cpu=10):
    """Micro non-rigid refinement: tiled optical flow at full resolution.

    1. Partition the image into overlapping tiles (get_grid_bboxes + expand_bbox)
    2. For each tile: extract, preprocess, downsample, DeepFlow, scale up
    3. Stitch tile displacements into full-res field
    4. Apply via mapim → warped3

    Returns warped3: np_img.
    """
    shape_rc = np.array([ref_c.height, ref_c.width])

    # ── 1. Build tile grid (mirrors NonRigidRegistrar.register lines 1763-1764) ──
    effective_tile_wh = TILE_WH * micro_non_rigid_factor
    effective_buffer = TILE_BUFFER * micro_non_rigid_factor

    temp_tile_bboxes = _get_grid_bboxes(shape_rc, effective_tile_wh, effective_tile_wh,
                                         inclusive=True)
    expanded_bboxes = np.array([
        _expand_bbox(bbox_xywh, effective_buffer, shape_rc)
        for bbox_xywh in temp_tile_bboxes
    ])

    n_tiles = len(temp_tile_bboxes)
    n_cols = len(np.unique(temp_tile_bboxes[:, 0]))
    n_rows = len(np.unique(temp_tile_bboxes[:, 1]))

    batch_size=32
    bk_dxdy=np.zeros((ref_c.height, ref_c.width, 2), dtype=np.float32)
    for i in range(0, n_tiles//batch_size+1):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,n_tiles)
        print(f"Processing tiles {start_id} to {end_id-1} / {n_tiles-1}...",flush=True)
        np_moving_list=[]
        np_fixed_list=[]
        np_mask_list=[]
        is_rgb=moving.is_rgb
        tile_downsample_rate=micro_non_rigid_factor
        for tile_id in range(start_id,end_id):
            box=expanded_bboxes[tile_id]
            x,y,w,h=box.astype(int)
            np_moving=moving.extract_area(x, y, w, h, level=0)
            np_moving=np_img(np_moving, interpretation="b-w", background=0)
            np_moving = extract_channel(np_moving, moving_channel)
            np_fixed=ref_c.extract_area(x, y, w, h, level=0)
            np_mask=None
            np_moving_list.append(np_moving)
            np_fixed_list.append(np_fixed)
            np_mask_list.append(np_mask)
        bk_dxdy_tile_batch,_=register_tiles(np_moving_list,np_fixed_list,np_mask_list,is_rgb,tile_downsample_rate,0,0,n_cpu)
        gc.collect()
        for tile_id in range(start_id,end_id):
            box=expanded_bboxes[tile_id]
            x,y,w,h=box.astype(int)
            bk_dxdy[y:y+h,x:x+w,:]+=bk_dxdy_tile_batch[tile_id-start_id]
        # force-free batch temporaries — prevent accumulation across batches
        del np_moving_list, np_fixed_list, np_mask_list, bk_dxdy_tile_batch, _
    _mem("after tile registration")
    gc.collect()

    #get overlap
    w_list=np.sort(np.unique(expanded_bboxes[:, 1]))
    h_list=np.sort(np.unique(expanded_bboxes[:, 0]))
    w_size=expanded_bboxes[0,3]
    h_size=expanded_bboxes[0,2]
    w_overlap_list=[]
    h_overlap_list=[]
    
    for i in range(len(w_list)-1):
        w_overlap_list.append((int(w_list[i+1]), int(w_list[i]+w_size)))

    for i in range(len(h_list)-1):
        h_overlap_list.append((int(h_list[i+1]), int(h_list[i]+h_size)))

    for w_overlap in w_overlap_list:
        bk_dxdy[w_overlap[0]:w_overlap[1],:,:] /= 2
    for h_overlap in h_overlap_list:
        bk_dxdy[:, h_overlap[0]:h_overlap[1], :] /= 2

    for x in range(bk_dxdy.shape[1]):
        bk_dxdy[:, x, 0] += x
    for y in range(bk_dxdy.shape[0]):
        bk_dxdy[y, :, 1] += y

    warped3 = moving.mapim(bk_dxdy)
    _mem("after warping")

    return warped3

# ── bbox helpers (mirrors warp_tools.get_grid_bboxes / expand_bbox) ──

def _get_grid_bboxes(shape_rc, bbox_w, bbox_h, inclusive=False):
    """Get list of bbox xywh for an image with shape shape_rc.
    Ordered left-to-right, top-to-bottom.

    Parameters
    ----------
    shape_rc : (n_row, n_col)
    bbox_w, bbox_h : int
    inclusive : bool
        If True, include edge boxes even if smaller than bbox_w/bbox_h.

    Returns
    -------
    bbox_list : [N, 4] ndarray (x, y, w, h)
    """
    temp_x = np.arange(0, shape_rc[1], bbox_w).astype(float)
    temp_y = np.arange(0, shape_rc[0], bbox_h).astype(float)

    if inclusive:
        if shape_rc[1] not in temp_x:
            temp_x = np.hstack([temp_x, shape_rc[1]])
        if shape_rc[0] not in temp_y:
            temp_y = np.hstack([temp_y, shape_rc[0]])

    tl_y, tl_x = np.meshgrid(temp_y, temp_x, indexing="ij")
    bbox_list = [[tl_x[i, j],
                  tl_y[i, j],
                  tl_x[i + 1, j + 1] - tl_x[i, j],
                  tl_y[i + 1, j + 1] - tl_y[i, j]]
                 for i in range(len(temp_y) - 1)
                 for j in range(len(temp_x) - 1)]

    return np.array(bbox_list)


def _expand_bbox(bbox_xywh, expand, shape_rc=None):
    """Expand bbox outward by `expand` pixels, clamped to shape_rc."""
    new_xy = bbox_xywh[0:2] - expand
    new_xy[new_xy < 0] = 0
    new_x, new_y = new_xy

    new_w, new_h = bbox_xywh[2:] + 2 * expand

    if shape_rc is not None:
        h, w = shape_rc
        if new_x + new_w >= w:
            new_w = w - new_x
        if new_y + new_h >= h:
            new_h = h - new_y

    return np.array([new_x, new_y, new_w, new_h])

