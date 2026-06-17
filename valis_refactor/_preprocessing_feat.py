"""Vendored feature-detection preprocessing helpers."""

import numpy as np
from skimage import exposure
import torch
import einops


def img_to_tensor(img, return_rgb=True):
    """Convert numpy array to pytorch.Tensor.

    Parameters
    ----------
    img: np.ndarray
        Image with shape (H, W, C) for RGB or (H, W) for single channel.
    return_rgb: bool
        If True, greyscale images are converted to 3-channel RGB.

    Returns
    -------
    t_img: torch.Tensor
        Pytorch tensor with shape (B, C, H, W).
    """
    if np.issubdtype(img.dtype, np.integer):
        float_img = exposure.rescale_intensity(img, out_range=np.float32)
    elif img.max() > 1:
        float_img = img / img.max()
    else:
        float_img = img

    tensor_img = torch.from_numpy(float_img)
    if tensor_img.ndim == 2:
        if return_rgb:
            tensor_img = torch.stack(3 * [tensor_img])
        else:
            tensor_img.unsqueeze(0)
    else:
        tensor_img = einops.rearrange(tensor_img, "h w c -> c h w")

    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img
