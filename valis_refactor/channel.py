"""
Channel extraction for registration preprocessing.

extract_channel(img, channel):
    int → img[..., channel]  (band index)
    "deconv" → he_deconv(img)  (H&E stain deconvolution → hematoxylin)
"""

import numpy as np
VERBOSE=True

def extract_channel(img, channel):
    """Extract the registration channel from an np_img.

    Parameters
    ----------
    img : np_img
        Image to extract channel from.
    channel : int or str
        int → band index: img._data[..., channel]
        "deconv" → H&E deconvolution to hematoxylin channel

    Returns
    -------
    np.ndarray (2D, single-channel)
    """
    data = img._data
    if isinstance(channel, int):
        if data.ndim == 2:
            return data
        return data[..., channel]
    elif channel == "deconv":
        return he_deconv(data)
    else:
        raise ValueError(f"Unknown channel spec: {channel!r}")


def he_deconv(np_mat):
    """H&E stain deconvolution — extract hematoxylin channel.

    Ported from valis/tile_registration.py.

    Parameters
    ----------
    np_mat : np.ndarray
        RGB image (H, W, 3).

    Returns
    -------
    np.ndarray (H, W)
        Hematoxylin channel (single-channel, same spatial dims).
    """
    if np_mat.ndim != 3 or np_mat.shape[-1] != 3:
        return np_mat[..., 0] if np_mat.ndim == 3 else np_mat
    return _deconvolution_he(np_mat)

def log_and_div(img, Io=240):
    """Optical density transform. Returns float32 to halve memory vs float64."""
    if img.dtype == np.uint8:
        max_num = 255
    elif img.dtype == np.uint16:
        max_num = 65535
    else:
        return -np.log((img.astype(np.float32) + 1) / Io)

    lookuptable = -np.log((np.arange(max_num + 1, dtype=np.float32) + 1) / Io)
    return lookuptable[img]

# ── internal: H&E deconvolution ─────────────────────────────────

def _deconvolution_he(img, Io=240, alpha=1, beta=0.15, sample_num=1000000,
                      tile_rows=10_000_000):
    """Macenko H&E stain deconvolution — memory-tiled for large WSIs.

    Samples pixels *before* computing the full optical-density array, then
    projects in tiles so the (W×H, 3) float array never fully exists in RAM.
    """
    if VERBOSE:
        import time
        start_time = time.time()

    max_conc_ref = np.array([1.9705, 1.0308], dtype=np.float32)
    orig_shape = img.shape
    img_flat = img.reshape((-1, 3))
    n_total = img_flat.shape[0]

    # ── 1. sample pixels for stain matrix (before any big allocation) ─
    if sample_num is not None and n_total > sample_num:
        rng = np.random.default_rng()
        idx = rng.choice(n_total, sample_num, replace=False)
    else:
        idx = np.arange(n_total)

    # Compute OD *only* for sampled pixels → at most sample_num×3 float32
    od_sampled = log_and_div(img_flat[idx], Io)
    od_sample = od_sampled[~np.any(od_sampled < beta, axis=1)]

    if od_sample.shape[0] < 3:
        # Not enough tissue — fall back to grayscale
        return np.mean(img, axis=-1).astype(np.uint8)

    # ── 2. eigenvectors of OD covariance (from samples only) ─────────
    try:
        _, eigvecs = np.linalg.eigh(np.cov(od_sample.T))
    except Exception:
        eigvecs = np.array([
            [0.19819253, -0.79567392, -0.57238337],
            [-0.39573746,  0.46929681, -0.78940001],
            [0.89672269,  0.38296672, -0.22186685],
        ], dtype=np.float32)
    eigvecs = eigvecs.astype(np.float32)

    # project sampled OD onto plane of 2 largest eigenvectors
    t_hat = od_sample.dot(eigvecs[:, 1:3])
    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eigvecs[:, 1:3].dot(
        np.array([[np.cos(min_phi), np.sin(min_phi)]], dtype=np.float32).T
    )
    v_max = eigvecs[:, 1:3].dot(
        np.array([[np.cos(max_phi), np.sin(max_phi)]], dtype=np.float32).T
    )

    # hematoxylin first, eosin second
    if v_min[0] > v_max[0]:
        h_e_vector = np.array([v_min[:, 0], v_max[:, 0]], dtype=np.float32).T
    else:
        h_e_vector = np.array([v_max[:, 0], v_min[:, 0]], dtype=np.float32).T

    # ── 3. project in tiles — never build the full (N,3) OD array ────
    h_e_pinv = np.linalg.pinv(h_e_vector)  # (2, 3) float32
    hema = np.empty(n_total, dtype=np.float32)

    for start in range(0, n_total, tile_rows):
        end = min(start + tile_rows, n_total)
        od_tile = log_and_div(img_flat[start:end], Io)  # (tile, 3) float32
        # h_e_pinv[0] is the hematoxylin row: (1, 3) @ (3, tile) → (tile,)
        hema[start:end] = h_e_pinv[0].astype(np.float32) @ od_tile.T

    # ── 4. normalize ──────────────────────────────────────────────────
    max_conc = np.percentile(hema, 99)
    if max_conc > 0:
        hema *= (max_conc_ref[0] / max_conc)

    hema_normed = (hema * 255).clip(0, 255).astype(np.uint8)
    return hema_normed.reshape(orig_shape[:2])


if __name__ == "__main__":
    # Quick smoke test: random 100x100x3 RGB, deconv should not crash
    N=50000
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, (N,N, 3), dtype=np.uint8)
    result = he_deconv(data)
    assert result.shape == (N, N), f"bad shape: {result.shape}"
    assert result.dtype == np.uint8, f"bad dtype: {result.dtype}"
    assert result.max() >= 0 and result.min() <= 255
    print(f"he_deconv: OK — shape={result.shape}, dtype={result.dtype}, range=[{result.min()},{result.max()}]")