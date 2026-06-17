import numpy as np
from skimage import exposure
import cv2
from concurrent.futures import ThreadPoolExecutor
import time

import warp_tools
from _nonrigid_registrar import OpticalFlowWarper
from _preprocessing_extra import collect_img_stats, norm_img_stats
from channel import he_deconv
import os

def norm_img(img, stats, mask=None):
    normed_img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    normed_img = norm_img_stats(img=normed_img, target_stats=stats, mask=mask)
    normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)

    return normed_img

def norm_pair_tiles(np_moving,np_fixed,np_mask,total_stats):
    try:
        _, target_processing_stats = collect_img_stats([np_fixed, np_moving])
        fixed_normed = norm_img(np_fixed, target_processing_stats, np_mask)
        moving_normed = norm_img(np_moving, target_processing_stats, np_mask)
    except ValueError:
        try:
            fixed_normed = norm_img(np_fixed, total_stats, np_mask)
            moving_normed = norm_img(np_moving, total_stats, np_mask)
        except ValueError:
            fixed_normed = np_fixed
            moving_normed = np_moving
    return fixed_normed, moving_normed

def norm_tile(np_mat):
    np_mat = exposure.rescale_intensity(np_mat, in_range="image", out_range=(0.0, 1.0))
    np_mat = exposure.equalize_adapthist(np_mat)
    np_mat = exposure.rescale_intensity(np_mat, in_range="image", out_range=(0, 255)).astype(np.uint8)
    return np_mat

def reg_tile(tile_kwargs):
    np_moving,np_fixed,np_mask,total_stats,is_rgb,tile_downsample_rate,ref_channel,target_channel=tile_kwargs
    if is_rgb and np_moving.ndim == 3 and np_fixed.ndim == 3:
        edge_mask = 255*((np_moving.min(axis=2) != np_moving.max(axis=2)) & (np_fixed.min(axis=2) != np_fixed.max(axis=2))).astype(np.uint8)
        if np_mask is not None:
            np_mask = 255*((edge_mask > 0) & (np_mask > 0)).astype(np.uint8)
        else:
            np_mask = edge_mask
    is_empty = np_fixed.max() == np_fixed.min() or np_moving.max() == np_moving.min()
    if np_mask is not None:
        is_empty = is_empty or np_mask.max() == 0

    if is_empty:
        empty_dxdy = warp_tools.numpy2vips(np.zeros((np_moving.shape[0],np_moving.shape[1],2),dtype=np.float32))
        return empty_dxdy,empty_dxdy
    
    #import pdb;pdb.set_trace()
    
    if isinstance(ref_channel, int):
        if np_fixed.ndim == 3:
            np_fixed = np_fixed[..., ref_channel]
    else:
        if np_fixed.ndim == 3 and np_fixed.shape[-1] == 3:
            np_fixed = he_deconv(np_fixed)

    if isinstance(target_channel, int):
        if np_moving.ndim == 3:
            np_moving = np_moving[..., target_channel]
    else:
        if np_moving.ndim == 3 and np_moving.shape[-1] == 3:
            np_moving = he_deconv(np_moving)

    np_moving=norm_tile(np_moving)
    np_fixed=norm_tile(np_fixed)
    
    fixed_normed, moving_normed = norm_pair_tiles(np_moving,np_fixed,np_mask,total_stats)

    fixed_normed=cv2.resize(fixed_normed, (fixed_normed.shape[1]//tile_downsample_rate, fixed_normed.shape[0]//tile_downsample_rate))
    moving_normed=cv2.resize(moving_normed, (moving_normed.shape[1]//tile_downsample_rate, moving_normed.shape[0]//tile_downsample_rate))
    
    reg_obj = OpticalFlowWarper(optical_flow_obj=cv2.optflow.createOptFlow_DeepFlow())
    _, _, bk_dxdy = reg_obj.register(moving_normed, fixed_normed)
    bk_dxdy = tile_downsample_rate * bk_dxdy 
    fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy)
    bk_dxdy=np.dstack(bk_dxdy)
    fwd_dxdy=np.dstack(fwd_dxdy)

    bk_dxdy=cv2.resize(bk_dxdy, (np_fixed.shape[1], np_fixed.shape[0]))
    fwd_dxdy=cv2.resize(fwd_dxdy, (np_fixed.shape[1], np_fixed.shape[0]))

    vips_tile_bk_dxdy = warp_tools.numpy2vips(bk_dxdy.astype(np.float32))
    vips_tile_fwd_dxdy = warp_tools.numpy2vips(fwd_dxdy.astype(np.float32))

    return vips_tile_bk_dxdy,vips_tile_fwd_dxdy

def register_tiles(np_moving_list,np_fixed_list,np_mask_list,is_rgb,tile_downsample_rate,ref_channel,target_channel,n_cpu =10):
    t_start = time.time()
    _, total_stats = collect_img_stats(np_fixed_list+np_moving_list)
    print(f"collecting image stats using {time.time() - t_start} seconds")
    t_start = time.time()
    #import pdb;pdb.set_trace()
    args=[(np_moving,np_fixed,np_mask,total_stats,is_rgb,tile_downsample_rate,ref_channel,target_channel) for np_moving,np_fixed,np_mask in zip(np_moving_list,np_fixed_list,np_mask_list)]
    #print(f"Starting registration of tiles using {time.time() - t_start} seconds")
    t_start = time.time()
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_cpu) as pool:
        results = list(pool.map(reg_tile, args))
    #results = list(map(reg_tile, args))
    print(f"Registering tiles using {time.time() - t_start} seconds")
    return [result[0] for result in results], [result[1] for result in results]