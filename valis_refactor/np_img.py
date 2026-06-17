"""
np_img: numpy-backed image class replacing pyvips.Image for registration operations.

Always holds the highest-resolution (level 0) data in memory as a numpy array.
Supports lazy level-based cropping: when cropping at level > 0, the image is
resized to that level on demand.

pyvips-compatible method names for the subset used by the registration pipeline.
"""

import numpy as np
import cv2
import pyvips


class np_img:
    """Numpy-backed image with pyvips-compatible API for registration operations.

    Always eager: _data holds the full-resolution numpy array.
    level_dims records (width, height) for each pyramid level.
    """

    def __init__(self, data, level_dims=None, interpretation=None, background=0):
        """
        Parameters
        ----------
        data : np.ndarray
            Image data at full resolution (level 0). 2D for single-band, 3D for multi-band.
        level_dims : list of (int, int), optional
            (width, height) for each pyramid level. level_dims[0] = data dimensions.
            If None, defaults to [data.shape[1::-1]].
        interpretation : str, optional
            "srgb", "b-w", or "multiband". Auto-detected if None.
        background : int, optional
            Background fill value. 0 for dark-field/fluorescence, 255 for bright-field.
        """
        self._data = data
        self.background = background
        if level_dims is None:
            self.level_dims = [(data.shape[1], data.shape[0])]
        else:
            self.level_dims = list(level_dims)

        if interpretation is None:
            if data.ndim == 3 and data.shape[2] == 3:
                interpretation = "srgb"
            elif data.ndim == 3:
                interpretation = "multiband"
            else:
                interpretation = "b-w"
        self.interpretation = interpretation
        self._level_data=[None]*(len(self.level_dims)-1)
    # ── properties ──────────────────────────────────────────────

    @property
    def width(self):
        return self._data.shape[1]

    @property
    def height(self):
        return self._data.shape[0]

    @property
    def bands(self):
        if self._data.ndim == 2:
            return 1
        return self._data.shape[2]

    @property
    def is_rgb(self):
        return self.interpretation == "srgb"

    # ── static constructors ─────────────────────────────────────

    @classmethod
    def black(cls, w, h, bands=1):
        shape = (h, w) if bands == 1 else (h, w, bands)
        return cls(np.zeros(shape, dtype=np.float32), interpretation="b-w")

    @classmethod
    def xyz(cls, w, h):
        """Coordinate grid as (h, w, 2) float32 — no temporary mgrid/astype copies."""
        data = np.empty((h, w, 2), dtype=np.float32)
        # broadcasting writes — no full-size temporaries
        data[..., 0] = np.arange(w, dtype=np.float32)
        data[..., 1] = np.arange(h, dtype=np.float32)[:, np.newaxis]
        return cls(data, interpretation="multiband")

    # ── level-based lazy crop ───────────────────────────────────

    def extract_area(self, x, y, w, h, level=0):
        """Crop region (x, y, w, h) at given pyramid level.

        If level == 0: direct numpy slice (O(1), returns a view).
        If level > 0: lazily resizes to that level then crops.
        """
        x, y, w, h = int(x), int(y), int(w), int(h)
        if level == 0:
            roi = self._data[y:y + h, x:x + w]
            return roi
        else:
            if self._level_data[level-1] is None:
                level_w, level_h = self.level_dims[level]
                scale_x = level_w / self.width
                scale_y = level_h / self.height
                scaled_data = cv2.resize(self._data, (level_w, level_h), interpolation=cv2.INTER_AREA)
                self._level_data[level-1] = scaled_data

            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            roi = self._level_data[level-1][y_scaled:y_scaled + h_scaled, x_scaled:x_scaled + w_scaled]
            return roi

    # ── pyvips bridge ───────────────────────────────────────────

    def _to_vips(self):
        """Convert internal numpy array to pyvips Image."""
        vi = pyvips.Image.new_from_array(self._data)
        return vi

    # ── warp / transform operations (eager) ─────────────────────

    def resize(self, scale, interp="bicubic"):
        """Resize by scale factor. Returns new np_img."""
        new_w = max(1, int(self.width * scale))
        new_h = max(1, int(self.height * scale))
        inter = _get_cv2_interp(interp)
        resized = cv2.resize(self._data, (new_w, new_h), interpolation=inter)
        if resized.ndim == 2 and self._data.ndim == 2:
            pass  # keep 2D
        elif self._data.ndim == 2:
            resized = resized  # cv2 keeps shape
        return np_img(resized, interpretation=self.interpretation, background=self.background)

    def affine(self, matrix, oarea=None, interpolate="bicubic", background=None):
        """Apply affine transform via pyvips. Returns new np_img.

        Parameters
        ----------
        matrix : np.ndarray
            3x3 affine matrix (pyvips convention: maps dst → src).
        oarea : (w, h) or None
            Output area. If None, same as input.
        interpolate : str
            "bicubic", "bilinear", or "nearest".
        background : int or tuple
            Background fill value.
        """
        if background is None:
            background = self.background
        print("starting affine warp...")
        out_w, out_h = oarea if oarea else (self.width, self.height)
        interp_map = {"bicubic": "bicubic", "bilinear": "bilinear", "nearest": "nearest"}
        interpolator = pyvips.Interpolate.new(interp_map.get(interpolate, "bicubic"))

        vi = self._to_vips()
        M = np.asarray(matrix, dtype=np.float64)
        tx, ty = M[:2, 2]
        M_inv = np.linalg.inv(M)
        vips_M = M_inv[:2, :2].reshape(-1).tolist()

        vi = vi.affine(vips_M,
                       oarea=[0, 0, out_w, out_h],
                       interpolate=interpolator,
                       idx=-tx,
                       idy=-ty,
                       premultiplied=True,
                       background=background,
                       extend=pyvips.enums.Extend.BACKGROUND)
        print("finished affine warp.")
        return np_img(vi.numpy(), interpretation=self.interpretation, background=self.background)

    def mapim(self, index, interpolate="bicubic", background=None):
        """Remap by index/displacement field via pyvips. Returns new np_img.

        Parameters
        ----------
        index : np_img or np.ndarray
            2-band image where index[0] = x-coordinates, index[1] = y-coordinates
            mapping each output pixel to its source location in the input image.
        interpolate : str
        background : int or tuple
        """
        #print("begin warping.",flush=True)
        if background is None:
            background = self.background
        interp_map = {"bicubic": "bicubic", "bilinear": "bilinear", "nearest": "nearest"}
        interpolator = pyvips.Interpolate.new(interp_map.get(interpolate, "bicubic"))

        vi = self._to_vips()
        idx_data = index.numpy() if isinstance(index, np_img) else index
        if idx_data.dtype != np.float32:
            idx_data = idx_data.astype(np.float32)
        vip_idx = pyvips.Image.new_from_array(idx_data)

        try:
            warped = vi.mapim(vip_idx,
                              premultiplied=True,
                              background=background,
                              extend=pyvips.enums.Extend.BACKGROUND,
                              interpolate=interpolator)
        except pyvips.error.Error:
            warped = vi.mapim(vip_idx, interpolate=interpolator)
            if background is not None:
                warped = (warped == 0).ifthenelse(background, warped)
        
        #print("warping done.",flush=True)
        return np_img(warped.numpy(), interpretation=self.interpretation, background=self.background)

    def bandjoin(self, other):
        """Join with another np_img along the band axis."""
        a = self._data
        b = other._data if isinstance(other, np_img) else other
        if a.ndim == 2:
            a = a[..., np.newaxis]
        if b.ndim == 2:
            b = b[..., np.newaxis]
        joined = np.concatenate([a, b], axis=-1)
        return np_img(joined, interpretation=self.interpretation, background=self.background)

    def cast(self, format_str):
        """Cast to dtype. ``format_str`` can be "float" (float32) or "uchar" (uint8)."""
        dtype_map = {"float": np.float32, "uchar": np.uint8}
        dtype = dtype_map.get(format_str, np.float32)
        return np_img(self._data.astype(dtype), level_dims=self.level_dims,
                      interpretation=self.interpretation, background=self.background)

    # ── immediate / aggregate ───────────────────────────────────

    def max(self):
        return self._data.max()

    def min(self):
        return self._data.min()

    # ── numpy bridge ────────────────────────────────────────────

    def numpy(self):
        """Return the underlying numpy array."""
        return self._data

    def __array__(self, dtype=None):
        if dtype is None:
            return self._data
        return self._data.astype(dtype)

    # ── operators ───────────────────────────────────────────────

    def __getitem__(self, key):
        """Band indexing: img[0] returns band 0."""
        if isinstance(key, int):
            if self._data.ndim == 2:
                return np_img(self._data, interpretation=self.interpretation, background=self.background)
            return np_img(self._data[..., key], interpretation="b-w", background=self.background)
        return np_img(self._data[key], interpretation=self.interpretation, background=self.background)

    def __add__(self, other):
        if isinstance(other, np_img):
            return np_img(self._data + other._data, interpretation=self.interpretation, background=self.background)
        return np_img(self._data + other, interpretation=self.interpretation, background=self.background)

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if isinstance(other, np_img):
            return self._data == other._data
        return self._data == other

    def __repr__(self):
        return f"np_img({self.width}x{self.height}, bands={self.bands}, {self.interpretation})"

def _get_cv2_interp(name):
    return {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }.get(name, cv2.INTER_CUBIC)
