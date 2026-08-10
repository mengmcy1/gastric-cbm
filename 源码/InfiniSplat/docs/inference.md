# Inference Guide

Run all commands from the repository root through the Python entry point:

```bash
python -m src.demo.infer_batch_images [OPTIONS]
```

## Modes

| Mode | Encoder configuration | Default checkpoint | Prompt depth |
| --- | --- | --- | --- |
| `rgb` | RGB-only | `checkpoints/infinisplat_rgb.ckpt` | Not used |
| `lidar` | RGB + prompt-conditioned InfiniDepth | `checkpoints/infinisplat_lidar.ckpt` | Required |

The selected checkpoint contains the full inference model. Encoder and decoder weights are not downloaded separately.

## Input discovery

`--input` accepts one image or a directory. Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. Directory entries are sorted to make batch selection deterministic.

Only files directly inside the selected directory are scanned. Use `--limit N` to process only the first `N` selected images. A limit of `0` means no cap.

When `--input` is omitted, each mode uses its bundled example directory:

```text
rgb   -> examples/data/rgb_demo
lidar -> examples/data/lidar_demo
```

## Depth pairing

By default, depth files are searched in the input directory and paired by filename stem. For a single-image input, this is the directory containing the image. For example:

```text
data/frame_001.jpg
data/frame_001.npy
```

Run this pair with:

```bash
python -m src.demo.infer_batch_images \
  --mode lidar \
  --input data/frame_001.jpg
```

or to process all images and depths in a directory:

```bash
python -m src.demo.infer_batch_images \
  --mode lidar \
  --input data
```

---

To explicitly select the depth file for a single image, use `--prompt-depth`:

```bash
python -m src.demo.infer_batch_images \
  --mode lidar \
  --input /path/to/frame_001.jpg \
  --prompt-depth /path/to/frame_001.npy
```

To keep depth files in a separate directory, set `--prompt-depth-dir`; each image is still paired with a depth file that has the same stem:

```bash
python -m src.demo.infer_batch_images \
  --mode lidar \
  --input /path/to/images \
  --prompt-depth-dir /path/to/depths
```

`--prompt-depth` applies the same file to every selected image, so it should normally be used only with a single-image input. Candidate extensions are tried in this order when pairing automatically: `.npz`, `.npy`, `.h5`, `.hdf5`, `.exr`.

## Depth input format

Prompt depth must be spatially aligned with the RGB image and use larger values for farther points. Metric scale is optional: scale-ambiguous depth maps are also supported, while metric input preserves the scene scale in the exported 3DGS. Disparity or inverse depth must be converted to depth first.

Depth arrays should use shape `[H, W]`. Plain arrays and dense maps are accepted; `.npz` files may instead store a sparse `mask` and `value` pair. Valid values must be finite and strictly between 1 and 100 after decoding, so relative depth in `[0, 1]` must be rescaled first. At most 1500 valid samples are used as prompts.

## Camera intrinsics

Camera parameters are resolved in the following order:

1. `--intrinsics-file`: a YAML or JSON file containing a 3 x 3 pixel-space matrix.
2. `--focal-px`: focal length in pixels; `fx = fy`, with a centered principal point.
3. `--focal-mm`: 35 mm full-frame-equivalent focal length.
4. Image EXIF focal length.
5. A fixed 30 mm full-frame-equivalent fallback.

The three command-line overrides are mutually exclusive. One override is reused for every image in a batch.

An intrinsics file may place the matrix at either `intrinsics_px` or `camera.intrinsics_px`:

```yaml
intrinsics_px:
  - [1200.0, 0.0, 768.0]
  - [0.0, 1200.0, 576.0]
  - [0.0, 0.0, 1.0]
```

Values must describe the original input image in pixels. The loader scales the matrix automatically during preprocessing.

## Optional video render resolution

When the optional `gsplat` package is installed, videos render at the original image resolution when possible. Very large frames are scaled down to a maximum long edge of 3840 pixels and a maximum area of `3840 x 2160` pixels. Each video contains 60 frames at 10 FPS.

## Outputs and resume behavior

The default output root is `outputs/demo/<mode>`. Every image receives its own directory:

```text
outputs/demo/rgb/example/
├── example.ply
├── example.mp4   # optional: requires gsplat
└── example.html  # optional: requires splat-transform
```

PLY export is always enabled and does not require `gsplat`. Video export is enabled by default when `gsplat` is installed; otherwise it is skipped with a warning. Pass `--no-video` to disable it explicitly. HTML export is independent of `gsplat` and is skipped with a warning when the optional `splat-transform` executable is unavailable.

If every requested artifact already exists, the case is skipped. If the PLY and requested video exist but HTML is missing, only HTML conversion runs. Pass `--overwrite` to recompute requested outputs.

The HTML converter reads `config/viewer_settings.json` directly. It does not generate a hidden viewer-settings file.

## Common options

| Option | Description |
| --- | --- |
| `--mode {rgb,lidar}` | Select the model mode. |
| `--checkpoint PATH` | Override the checkpoint selected by the mode. |
| `--input PATH` | Process one image or a directory. |
| `--output-dir PATH` | Override `outputs/demo/<mode>`. |
| `--limit N` | Process at most `N` selected images; `0` means all. |
| `--overwrite` | Recompute outputs that already exist. |
| `--device DEVICE` | Override automatic device selection, for example `cuda:0`. |
| `--intrinsics-file PATH` | Use a 3 x 3 pixel-space intrinsics matrix. |
| `--focal-px VALUE` | Use a focal length in pixels. |
| `--focal-mm VALUE` | Use a full-frame-equivalent focal length in millimeters. |
| `--prompt-depth PATH` | Use one prompt-depth file. |
| `--prompt-depth-dir PATH` | Pair depth files by image stem. |
| `--disable-floater-filter` | Keep Gaussians removed by the final floater filter. |
| `--no-video` | Skip MP4 rendering. |
| `--no-export-html` | Skip HTML viewer export. |

Print the authoritative option list with:

```bash
python -m src.demo.infer_batch_images --help
```
