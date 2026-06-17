"""
Register a moving slide to a reference slide using valis_reg.

Usage:
    python demo.py <ref_slide_path> <moving_slide_path> [options]

Examples:
    python demo.py ref.tif moving.tif
    python demo.py ref.tif moving.tif --crop -j 8
    python demo.py ref.svs moving.svs -r deconv -m deconv -o result.ome.tiff
"""

#OPENBLAS_NUM_THREADS=1 /usr/bin/time -v python demo.py /home/ecs-user/data/pathology_img/0_HE_test.tif /home/ecs-user/data/pathology_img/0_PGR_test.tif -r deconv -m deconv

import sys
import os
import argparse
import pyvips
import numpy as np

sys.path.insert(0, "./valis_refactor")

import np_img as npm
from valis_reg import valis_reg

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── crop helper ─────────────────────────────────────────────────

def _crop_center_third(img):
    """Return a new np_img cropped to the center 1/3 (by width and height)."""
    w, h = img.width, img.height
    x = w // 3
    y = h // 3
    crop_w = w // 3
    crop_h = h // 3
    roi = img.extract_area(x, y, crop_w, crop_h, level=0)
    # carry forward interpretation and background
    return npm.np_img(roi, interpretation=img.interpretation,
                      background=img.background)

# ── standalone IO (no valis dependency) ────────────────────────

def _get_level_dims(vips_img):
    """Return list of (width, height) for each pyramid level.

    Uses openslide metadata when available (e.g. SVS, MRXS), otherwise
    falls back to SubIFDs (generic pyramid tiff), then pages.
    """
    dims = [(vips_img.width, vips_img.height)]

    # openslide-backed image
    n_levels = vips_img.get("openslide.level-count")
    if n_levels is not None:
        n = int(n_levels)
        for i in range(1, n):
            w = int(vips_img.get(f"openslide.level[{i}].width"))
            h = int(vips_img.get(f"openslide.level[{i}].height"))
            dims.append((w, h))
        return dims

    # SubIFD pyramid
    n_subifds = vips_img.get("n-subifds")
    if n_subifds is not None:
        n = int(n_subifds)
        for i in range(1, n + 1):
            page = pyvips.Image.new_from_file(
                vips_img.get_string("vips-loader") == "tiffload"
                and vips_img.filename or vips_img.filename,
                subifd=i - 1, access="random",
            )
            dims.append((page.width, page.height))
        return dims

    # n-page fallback
    n_pages = vips_img.get("n-pages")
    if n_pages is not None:
        n = int(n_pages)
        for i in range(1, n):
            page = pyvips.Image.new_from_file(vips_img.filename, page=i, access="random")
            dims.append((page.width, page.height))
        return dims

    return dims


def _guess_interpretation(vips_img):
    """Map pyvips interpretation to np_img interpretation string."""
    interp = str(vips_img.interpretation)
    if interp == "srgb":
        return "srgb"
    if vips_img.bands >= 3:
        return "multiband"
    return "b-w"


def load_slide(path):
    """Load a slide as an np_img using pyvips/openslide directly."""
    # openslide handles SVS, MRXS, NDPI, etc.; tiffload handles generic/OME tiffs
    img = pyvips.Image.new_from_file(path, access="random", autocrop=True)

    # drop alpha channel if present
    if img.hasalpha():
        img = img.flatten()

    np_arr = img.numpy()
    level_dims = _get_level_dims(img)
    interp = _guess_interpretation(img)

    ni = npm.np_img(np_arr, level_dims=level_dims, interpretation=interp)

    # store metadata needed for saving later
    ni._src_vips = img  # keep the original vips handle for its metadata
    return ni


def _build_ome_xml(width, height, bands, dtype_str, channel_names=None,
                   pixel_size_x=None, pixel_size_y=None, pixel_size_unit="um"):
    """Build a minimal OME-XML string for OME-TIFF output."""
    dtype_map = {
        "uint8": "uint8", "uint16": "uint16", "int8": "int8",
        "int16": "int16", "uint32": "uint32", "int32": "int32",
        "float32": "float", "float64": "double",
    }
    bf_dtype = dtype_map.get(dtype_str, "uint8")

    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(bands)]

    px_attr = ""
    if pixel_size_x is not None and pixel_size_y is not None:
        px_attr = (f' PhysicalSizeX="{pixel_size_x}" PhysicalSizeXUnit="{pixel_size_unit}"'
                   f' PhysicalSizeY="{pixel_size_y}" PhysicalSizeYUnit="{pixel_size_unit}"')

    channels_xml = ""
    for i, name in enumerate(channel_names):
        channels_xml += (
            f'        <Channel ID="Channel:0:{i}" Name="{name}"'
            f' SamplesPerPixel="1"/>\n'
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06'
        ' http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">\n'
        '  <Image ID="Image:0" Name="warped">\n'
        f'    <Pixels ID="Pixels:0" DimensionOrder="XYCZT"'
        f' Type="{bf_dtype}" SizeX="{width}" SizeY="{height}"'
        f' SizeC="{bands}" SizeZ="1" SizeT="1"{px_attr}>\n'
        f'{channels_xml}'
        '    </Pixels>\n'
        '  </Image>\n'
        '</OME>'
    )
    return xml


def _compute_tile_size(width, height, max_tile=1024):
    """Pick a power-of-2 tile size that fits the image and minimises overhang."""
    tile = max_tile
    while tile > 16 and (tile > width or tile > height):
        tile //= 2
    # pick the largest power-of-2 <= tile that minimises waste
    best, best_waste = tile, (width % tile) + (height % tile)
    t = tile
    while t >= 16:
        waste = (width % t) + (height % t)
        if waste < best_waste:
            best, best_waste = t, waste
        t //= 2
    return max(best, 16)


def save_as_ome_tiff(np_data, dst_f, src_vips_img=None,
                     compression="jpeg", Q=90, pyramid=True):
    """Save a numpy array as a pyramid OME-TIFF using pyvips."""
    vips_img = pyvips.Image.new_from_array(np_data)

    h, w = np_data.shape[:2]
    bands = 1 if np_data.ndim == 2 else np_data.shape[2]

    # try to get pixel size from source metadata
    px_x, px_y, px_unit = None, None, "um"
    if src_vips_img is not None:
        px_x = src_vips_img.get("openslide.mpp-x")
        px_y = src_vips_img.get("openslide.mpp-y")
        if px_x is not None:
            px_x = float(px_x)
        if px_y is not None:
            px_y = float(px_y)

    channel_names = None
    if bands > 3 or (bands == 3 and str(vips_img.interpretation) != "srgb"):
        # multichannel: set interpretation so tiffsave writes each band as a page
        vips_img = vips_img.copy(interpretation="b-w")
        channel_names = [f"Channel {i}" for i in range(bands)]

    ome_xml = _build_ome_xml(
        w, h, bands, np_data.dtype.name,
        channel_names=channel_names,
        pixel_size_x=px_x, pixel_size_y=px_y, pixel_size_unit=px_unit,
    )

    vips_img = vips_img.copy()
    vips_img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml)
    vips_img.set_type(pyvips.GValue.gint_type, "page-height", h)

    tile_wh = _compute_tile_size(w, h)

    vips_img.tiffsave(
        dst_f,
        compression=compression,
        Q=Q,
        tile=True,
        tile_width=tile_wh,
        tile_height=tile_wh,
        pyramid=pyramid,
        subifd=pyramid,
        bigtiff=True,
    )


# ── main ───────────────────────────────────────────────────────

def _parse_channel(value):
    """Parse channel argument: 'deconv' → 'deconv', otherwise int."""
    if value == "deconv":
        return "deconv"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid channel: {value!r} (expected int or 'deconv')")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Register a moving slide to a reference slide (VALIS pair mode).")
    parser.add_argument("ref_path", help="Path to reference slide")
    parser.add_argument("moving_path", help="Path to moving slide")
    parser.add_argument("-r", "--ref-channel", type=_parse_channel, default=0,
                        help="Channel for reference slide (default: 0)")
    parser.add_argument("-m", "--moving-channel", type=_parse_channel, default=0,
                        help="Channel for moving slide (default: 0)")
    parser.add_argument("-o", "--output", default="warped.ome.tiff",
                        help="Output OME-TIFF path (default: warped.ome.tiff)")
    parser.add_argument("-j", "--n-cpu", type=int, default=16,
                        help="Number of CPU cores (default: 16)")
    parser.add_argument("--micro-factor", type=int, default=1,
                        help="Micro non-rigid resolution multiplier (default: 1)")
    parser.add_argument("--crop", action="store_true",
                        help="Crop centre 1/3 of both slides before registration (light-memory test)")
    args = parser.parse_args()

    # launch background RSS watcher — auto-exits when this process dies
    

    print(f"Loading ref:  {args.ref_path}")
    ref = load_slide(args.ref_path)
    print(f"  → {ref}")

    print(f"Loading moving: {args.moving_path}")
    moving = load_slide(args.moving_path)
    print(f"  → {moving}")
    moving.background = 255
    print(f"  moving background: {moving.background}")

    # ── optional centre-crop for quick memory-light test ────────
    if args.crop:
        print("\n--crop: cropping centre 1/3 of both slides")
        _src_vips = ref._src_vips  # keep for OME metadata at save time
        ref = _crop_center_third(ref)
        ref._src_vips = _src_vips
        print(f"  ref cropped → {ref}")
        moving = _crop_center_third(moving)
        print(f"  moving cropped → {moving}")

    # ── run registration ────────────────────────────────────────

    print("\nRunning valis_reg(ref, moving)...")
    warped = valis_reg(ref, moving, ref_channel=args.ref_channel,
                       moving_channel=args.moving_channel,
                       micro_non_rigid_factor=args.micro_factor,
                       n_cpu=args.n_cpu)
    print(f"\nResult: {warped}")

    # ── save OME-TIFF ───────────────────────────────────────────

    save_as_ome_tiff(warped._data, args.output, src_vips_img=ref._src_vips,
                     compression="jpeg", Q=90, pyramid=True)
    print(f"Saved {args.output}")