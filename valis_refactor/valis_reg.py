"""
valis_reg: Register a moving slide to a reference slide.

All inputs and outputs are np_img objects. No disk I/O.
Each step warps the moving image and passes the result to the next step.
"""

import cv2
import numpy as np
import os
import gc

import psutil

from channel import extract_channel
from np_img import np_img
from preprocessing import step1_preprocess
from rigid import step2_rigid
from nonrigid import step3_nonrigid
from micro_nonrigid import step4_micro_nonrigid

DUMP_THUMBNAIL = True  # for debugging: save thumbnails at each step


def _mem(label=""):
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[MEM {label}] {rss:.2f} GiB", flush=True)
    return rss


def valis_reg(ref, moving, ref_channel=0, moving_channel=0, micro_non_rigid_factor=1, n_cpu=10):
    """Register moving slide to reference slide.

    Parameters
    ----------
    ref : np_img
        Reference slide (full-res in memory).
    moving : np_img
        Slide to register (full-res in memory).
    ref_channel : int or str
        Channel to use for reference. int = band index, "deconv" = H&E deconv.
    moving_channel : int or str
        Channel to use for moving slide.
    micro_non_rigid_factor : int
        Factor by which to increase resolution for micro non-rigid registration.
    n_cpu : int
        Number of CPU cores to use for parallel processing.

    Returns
    -------
    warped : np_img
        Moving slide after registration, warped to align with ref.
    """
    _mem("start")

    # Extract channeled data at full resolution for use in all registration steps
    ref_c = np_img(extract_channel(ref, ref_channel), interpretation="b-w", background=0)
    moving_c = np_img(extract_channel(moving, moving_channel), interpretation="b-w", background=0)

    # step1: preprocess — extract channel, resize, normalize
    ref_p, moving_p,scale_ref, scale_moving = step1_preprocess(ref_c, moving_c)

    
    # Delete original ref to save memory — ref_c has the single-channel data needed
    del ref, moving_c
    gc.collect()
    _mem("after channel extraction, deleted ref")

    if DUMP_THUMBNAIL:
        cv2.imwrite("ref_p.png", ref_p)
        cv2.imwrite("moving_p.png", moving_p)
    _mem("after step1")
    gc.collect()
    
    # step2: rigid — detect/match features, estimate M, micro-rigid refine, apply
    moving = step2_rigid(ref_c, moving, moving_channel, ref_p, moving_p)
    _mem("after step2")
    #gc.collect()
    print("gc collect over.", flush=True)
    
    # step3: non-rigid — optical flow on processed images, apply
    moving = step3_nonrigid(ref_c, moving,moving_channel, ref_p,scale_ref)
    _mem("after step3")
    gc.collect()

    
    # step4: micro non-rigid — higher-res optical flow, apply, crop
    moving = step4_micro_nonrigid(ref_c, moving,moving_channel, micro_non_rigid_factor, n_cpu=n_cpu)
    _mem("after step4")
    gc.collect()
    
    return moving

