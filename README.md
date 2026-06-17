# valis_reg

A (over-)simplified, in-memory refactoring of [VALIS](https://github.com/MathOnco/valis) — the Virtual Alignment of pathoLogy Image Series pipeline — originally motivated by the need to deploy registration across different platforms that decouples the I/O and computation part.

Both the reference and moving slides are loaded into memory as `np_img` objects at the start, and the entire registration pipeline runs against those in-memory arrays. This trades higher peak memory usage for simpler code and no intermediate disk writes.

This is **not** the full VALIS package. We only implement the pair registration part that register **one moving slide** to **one reference slide**.  

> ⚠️ **Precision warning:** This pipeline has only been tested with **uint8** slides. Other dtypes (uint16, float32, etc.) may encounter precision issues — in particular, H&E deconvolution, intensity normalisation, and optical-flow steps contain hardcoded assumptions (e.g. `max_num=255`, 8-bit clipping) that are not valid for higher bit depths.

## Differences from upstream VALIS

| Original VALIS | This refactor |
|---|---|
| Full workflow: slide ordering, clustering, multi-slide I/O | Focused on a single pair: `ref` → `moving` |
| Reads and writes slides, thumbnails, and pickles at each stage | All operations in memory |
| Saves warped result as OME-TIFF to disk | Returns `np_img` in memory |
| Accumulates transformation matrices and displacement fields | Applies the transform immediately and passes the warped image forward |

### Format-agnostic by design

This repo only concerns itself with in-memory operations — the registration pipeline works on numpy arrays wrapped in `np_img` objects and returns the same. That means you pick your own loader and exporter:

- **Loading:** use pyvips, OpenSlide, tifffile, or any library that gives you a numpy array — `valis_reg` doesn't care how the data got there.
- **Exporting:** write the warped result with whatever format or library suits your workflow (OME-TIFF via pyvips, tifffile, Bio-Formats, etc.).

This keeps the codebase narrowly focused on the registration itself, while staying compatible with WSI formats that libvips and OpenSlide support.

## Pipeline

```
ref ──────────────────────────────────────┐
moving ─► preprocess ─► rigid ─► nonrigid ─► micro-nr ─► warped np_img
```

1. **Preprocess** — extract channel, resize to low-res, normalize intensities
2. **Rigid** — detect & match features (DISK + LightGlue), estimate similarity transform, micro-rigid refinement
3. **Non-rigid** — dense optical flow (DeepFlow) to correct local deformations
4. **Micro non-rigid** — higher-resolution optical flow for fine feature alignment, crop to reference frame

The pipeline returns **only the warped moving slide**, clipped to the reference's spatial extent. Regions of the moving slide that fall outside the reference bounds are discarded — the output has the same width and height as the reference, and every pixel maps to a location inside the reference coordinate frame.

## Quick start

```python
from valis_reg import valis_reg
from np_img import np_img
from demo import load_slide    # or use your own pyvips loader

ref = load_slide("reference.ome.tiff")
moving = load_slide("slide_to_register.svs")

warped = valis_reg(ref, moving, ref_channel="deconv", moving_channel="deconv")
# warped is the moving slide aligned to ref — same dimensions as ref
```

## Running the demo

Download the test data (two H&E whole-slide images from the [DORIS database](https://doris.snd.se/)) and run `demo.py`:

```bash
# 1. download and extract test slides
wget -O /tmp/test.zip "https://doris.snd.se//api/file/2022-190-1/1/data?filePath=test.zip"
unzip /tmp/test.zip -d /tmp/test_slides

# 2. run registration
OPENBLAS_NUM_THREADS=1 python demo.py \
    /tmp/test_slides/0_HE_test.tif \
    /tmp/test_slides/0_PGR_test.tif \
    -r deconv -m deconv
```

This produces `warped.ome.tiff` — the moving slide (PGR) warped into alignment with the reference (HE).

### Test slide dimensions

| Slide | Width | Height | Bands | Dtype | In-memory size |
|---|---|---|---|---|---|
| `0_HE_test.tif` (ref) | 31,744 | 26,880 | 3 | uint8 | ~2.4 GiB |
| `0_PGR_test.tif` (moving) | 50,176 | 27,776 | 3 | uint8 | ~3.9 GiB |

Peak RSS during registration is ~26 GiB. A machine with at least 48 GiB RAM is recommended, or use `--crop` to test with a smaller region first.

## API

### `valis_reg(ref, moving, ref_channel=0, moving_channel=0, micro_non_rigid_factor=1, n_cpu=10)`

| Parameter | Type | Description |
|---|---|---|
| `ref` | `np_img` | Reference slide (full-res in memory) |
| `moving` | `np_img` | Slide to register (full-res in memory) |
| `ref_channel` | `int` or `"deconv"` | Channel for reference. `int` = band index, `"deconv"` = H&E hematoxylin |
| `moving_channel` | `int` or `"deconv"` | Channel for moving slide |
| `micro_non_rigid_factor` | `int` | Resolution multiplier for micro-registration (default 1) |
| `n_cpu` | `int` | CPU cores for parallel processing (default 10) |

Returns: **`np_img`** — the moving slide warped to the reference coordinate frame.

### `np_img`

A numpy-backed image class that replaces `pyvips.Image` for registration operations. Holds full-resolution data in memory, supports lazy pyramid-level access, and exposes a pyvips-compatible method subset (`affine`, `mapim`, `crop`, etc.).

## Installation

### System dependencies

libvips and OpenSlide must be installed on your system:

**Ubuntu / Debian:**
```bash
sudo apt install libvips-dev openslide-tools
```

**macOS:**
```bash
brew install vips openslide
```

**Windows:**  
Download binaries from [libvips](https://www.libvips.org/install.html) and [OpenSlide](https://openslide.org/download/).

### Python packages

```bash
pip install -r requirements.txt
```

## Utilities

### `memlog.py` — pipe-based RSS tracker

Wraps a pipeline and annotates every line of stdout with the current RSS (resident set size) of the target process, plus a running peak. Useful for spotting where a registration pipeline OOMs without instrumenting the code itself.

```bash
python demo.py 2>&1 | python utils/memlog.py
```

Output looks like:

```
Loading ref:  np_img(31744x26880, bands=3, srgb)   [MEM  2.41 GiB | peak  2.41 GiB, 12345]
Loading moving: np_img(50176x27776, bands=3, srgb)  [MEM  6.87 GiB | peak  6.87 GiB, 12345]
Running valis_reg...                                  [MEM 20.12 GiB | peak 20.12 GiB, 12345]
```

Each `[MEM …]` annotation shows current RSS and the highest RSS seen so far. At exit the peak is also printed to stderr:

```
[memlog] peak RSS: 20.12 GiB
```

By default it auto-detects the PID of the upstream pipe writer. You can also target a specific PID:

```bash
python utils/memlog.py --pid <pid>
```

## License

MIT — see [LICENSE](LICENSE).

Original VALIS: Copyright (c) 2021-2025 Chandler Gatenbee ([github.com/MathOnco/valis](https://github.com/MathOnco/valis))
