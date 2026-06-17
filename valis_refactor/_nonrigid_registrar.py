"""Vendored NonRigidRegistrar + OpticalFlowWarper for tile_registration."""

import numpy as np
import cv2
from skimage import color, filters
import warp_tools


class NonRigidRegistrar:
    """Minimal vendored non-rigid registrar base class.

    Provides the register() method used by OpticalFlowWarper,
    depending on warp_tools (now local) for image operations.
    """

    def __init__(self, params=None, rgb=False):
        self.params = params
        self.rgb = rgb
        self.moving_img = None
        self.fixed_img = None
        self.mask = None
        self.shape = None
        self.grid_spacing = None
        self.method = None
        self.warped_image = None
        self.deformation_field_img = None
        self.backward_dx = None
        self.backward_dy = None
        self._params_provided = bool(params)

    def apply_mask(self, mask):
        masked_moving = warp_tools.apply_mask(self.moving_img, mask)
        masked_fixed = warp_tools.apply_mask(self.fixed_img, mask)
        return masked_moving, masked_fixed

    def calc(self, moving_img, fixed_img, mask, *args, **kwargs):
        return None

    def create_mask(self):
        temp_mask = np.zeros(self.shape, dtype=np.uint8)
        img_list = [self.moving_img, self.fixed_img]
        for img in img_list:
            temp_mask[img > 0] = 255
        mask = warp_tools.bbox2mask(
            *warp_tools.xy2bbox(warp_tools.mask2xy(temp_mask)),
            temp_mask.shape,
        )
        return mask

    def register(self, moving_img, fixed_img, mask=None, **kwargs):
        moving_shape = warp_tools.get_shape(moving_img)[0:2]
        fixed_shape = warp_tools.get_shape(fixed_img)[0:2]
        assert np.all(moving_shape == fixed_shape), "Images have different shapes"

        self.shape = moving_shape
        self.moving_img = moving_img
        self.fixed_img = fixed_img

        if mask is None:
            mask = np.full(self.shape, 255, dtype=np.uint8)

        self.mask = mask

        if self.mask is not None:
            _, masked_fixed = self.apply_mask(self.mask)
            masked_moving = self.moving_img.copy()

            mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(self.mask))
            min_c, min_r = mask_bbox[0:2]
            max_c, max_r = mask_bbox[0:2] + mask_bbox[2:]
            mask = self.mask[min_r:max_r, min_c:max_c]
            masked_moving = masked_moving[min_r:max_r, min_c:max_c]
            masked_fixed = masked_fixed[min_r:max_r, min_c:max_c]
        else:
            masked_moving = self.moving_img.copy()
            masked_fixed = self.fixed_img.copy()

        bk_dxdy = self.calc(moving_img=masked_moving, fixed_img=masked_fixed, mask=mask, **kwargs)

        if mask is not None:
            bk_dx = np.zeros(self.shape)
            bk_dx[min_r:max_r, min_c:max_c] = bk_dxdy[0]
            bk_dx[self.mask == 0] = 0

            bk_dy = np.zeros(self.shape)
            bk_dy[min_r:max_r, min_c:max_c] = bk_dxdy[1]
            bk_dy[self.mask == 0] = 0

            bk_dxdy = np.array([bk_dx, bk_dy])

        # Simplified grid: return zero images instead of actual warp/grid
        # (the callers in tile_registration only use bk_dxdy)
        warped_img = np.zeros_like(moving_img)
        warp_grid = np.zeros(self.shape[:2])

        self.backward_dx = bk_dxdy[..., 0]
        self.backward_dy = bk_dxdy[..., 1]
        self.deformation_field_img = warp_grid
        self.warped_image = warped_img

        return warped_img, warp_grid, bk_dxdy


class OpticalFlowWarper(NonRigidRegistrar):
    """Dense optical flow registration using OpenCV DeepFlow.

    Minimal vendored version — only the calc() path used by tile_registration.
    """

    def __init__(
        self,
        params=None,
        optical_flow_obj=None,
        n_grid_pts=50,
        sigma_ratio=0.005,
        paint_size=5000,
        fold_penalty=1e-6,
        smoothing_method=None,
    ):
        super().__init__(params)
        if optical_flow_obj is None:
            optical_flow_obj = cv2.optflow.createOptFlow_DeepFlow()
        self.smoothing_method = smoothing_method
        self.sigma_ratio = sigma_ratio
        self.paint_size = paint_size
        self.fold_penalty = fold_penalty
        self.n_grid_pts = n_grid_pts
        self.method = optical_flow_obj.__class__.__name__
        self.optical_flow_obj = optical_flow_obj

    def calc(self, moving_img, fixed_img, *args, **kwargs):
        if self.method in ["createOptFlow_DenseRLOF", "createOptFlow_SimpleFlow"]:
            if moving_img.ndim == 2:
                moving_img = color.gray2rgb(moving_img)
            if fixed_img.ndim == 2:
                fixed_img = color.gray2rgb(fixed_img)

        backward_flow = self.optical_flow_obj.calc(
            fixed_img, moving_img, np.zeros(moving_img.shape[0:2], dtype=np.float32)
        )

        backward_flow = np.array([backward_flow[..., 0], backward_flow[..., 1]])

        if self.smoothing_method == "gauss":
            sigma = self.sigma_ratio * np.max(backward_flow[0].shape)
            smooth_dx = filters.gaussian(backward_flow[0], sigma=sigma)
            smooth_dy = filters.gaussian(backward_flow[1], sigma=sigma)
            backward_flow = np.array([smooth_dx, smooth_dy])
        elif self.smoothing_method == "inpaint":
            backward_flow = warp_tools.remove_folds_in_dxdy(
                backward_flow,
                n_grid_pts=self.n_grid_pts,
                paint_size=self.paint_size,
                method=self.smoothing_method,
            )
        elif self.smoothing_method == "regularize":
            backward_flow = warp_tools.untangle(
                backward_flow,
                n_grid_pts=self.n_grid_pts,
                penalty=self.fold_penalty,
                mask=self.mask,
            )

        return np.array(backward_flow)
