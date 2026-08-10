from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.demo.infer_single_image import (
    _resolve_checkpoint_path,
    _resolve_device,
    filter_final_gaussian_floaters,
    load_demo_config,
    load_demo_image_bundle,
    load_demo_model,
    load_prompt_depth_tensors,
    patch_supersplat_html_auto_rotate,
    render_novel_view_video_from_single_view,
    run_single_image_inference,
    scale_intrinsics_px,
    validate_prompt_configuration,
)
from src.model.decoder.decoder_gsplat import is_gsplat_available
from src.utils.gaussians import save_ply

MODE_EXPERIMENTS = {
    "rgb": "infinisplat_hypersim_rgb",
    "lidar": "infinisplat_hypersim_lidar",
}
MODE_CHECKPOINTS = {
    "rgb": Path("checkpoints/infinisplat_rgb.ckpt"),
    "lidar": Path("checkpoints/infinisplat_lidar.ckpt"),
}
MODE_INPUT_DIRS = {
    "rgb": Path("examples/data/rgb_demo"),
    "lidar": Path("examples/data/lidar_demo"),
}
DEFAULT_OUTPUT_ROOT = Path("outputs/demo")
DEFAULT_MAX_RENDER_LONG_EDGE = 3840
DEFAULT_MAX_RENDER_PIXELS = 3840 * 2160
SPLAT_TRANSFORM = os.environ.get("SPLAT_TRANSFORM", "splat-transform")
VIEWER_SETTINGS = Path(__file__).resolve().parents[2] / "config" / "viewer_settings.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROMPT_DEPTH_EXTENSIONS = (".npz", ".npy", ".h5", ".hdf5", ".exr")


@dataclass(frozen=True)
class CasePaths:
    """Output paths for one image case.

    Args:
        case_dir: Directory containing all artifacts for this input image.
        scene_ply: Final Gaussian PLY path.
        video: Novel-view video path.
        html: SuperSplat HTML path.
    """

    case_dir: Path
    scene_ply: Path
    video: Path
    html: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run InfiniSplat inference on one image or a directory of images "
            "without reloading model weights for every image."
        )
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_EXPERIMENTS),
        default="rgb",
        help="Inference mode. Defaults to rgb.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint override. Defaults to the released checkpoint for --mode.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Input image or directory. Defaults to the bundled examples for --mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory. Defaults to outputs/demo/<mode>.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of selected images to process; 0 means no cap.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute outputs that already exist.")

    parser.add_argument("--device", type=str, default="auto")
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--intrinsics-file", type=Path, default=None)
    camera_group.add_argument("--focal-px", type=float, default=None)
    camera_group.add_argument("--focal-mm", type=float, default=None)

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-depth", type=Path, default=None)
    prompt_group.add_argument(
        "--prompt-depth-dir",
        type=Path,
        default=None,
        help="Directory containing per-image prompt depth files with matching stems.",
    )
    parser.add_argument(
        "--disable-floater-filter",
        action="store_true",
        help="Disable final Gaussian floater filtering.",
    )

    parser.add_argument("--no-video", action="store_true", help="Skip novel-view video rendering.")
    parser.add_argument(
        "--no-export-html",
        dest="export_html",
        action="store_false",
        default=True,
        help="Disable SuperSplat HTML export.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the root output directory for this batch run."""
    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_ROOT / args.mode


def _collect_images(args: argparse.Namespace) -> list[Path]:
    """Collect input image paths in deterministic order."""
    input_path = args.input_path or MODE_INPUT_DIRS[args.mode]
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image extension: {input_path}")
        images = [input_path]
    elif input_path.is_dir():
        images = [
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        raise FileNotFoundError(f"Input image or directory not found: {input_path}")

    images = sorted(images)
    if not images:
        raise FileNotFoundError("No input images were found.")

    stem_counts = Counter(path.stem for path in images)
    duplicate_stems = sorted(stem for stem, count in stem_counts.items() if count > 1)
    if duplicate_stems:
        raise ValueError(
            "Input images must have unique filename stems. Duplicates: "
            f"{', '.join(duplicate_stems)}"
        )

    if args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if args.limit > 0:
        images = images[: args.limit]
    return images


def _resolve_prompt_depth_path(args: argparse.Namespace, image_path: Path) -> Path | None:
    """Resolve the prompt depth path for one image.

    Args:
        args: Parsed batch arguments.
        image_path: RGB image path.

    Returns:
        Prompt depth path for this image, or None when prompt depth is disabled.
    """
    if args.prompt_depth is not None:
        if not args.prompt_depth.exists():
            raise FileNotFoundError(f"Prompt depth file not found: {args.prompt_depth}")
        if not args.prompt_depth.is_file():
            raise ValueError(f"Prompt depth path is not a file: {args.prompt_depth}")
        return args.prompt_depth
    prompt_depth_dir = args.prompt_depth_dir
    if prompt_depth_dir is None and args.mode == "lidar":
        input_path = args.input_path or MODE_INPUT_DIRS[args.mode]
        prompt_depth_dir = input_path if input_path.is_dir() else input_path.parent
    if prompt_depth_dir is None:
        return None

    candidates = [
        prompt_depth_dir / f"{image_path.stem}{ext}"
        for ext in PROMPT_DEPTH_EXTENSIONS
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != image_path.resolve():
            return candidate
    raise FileNotFoundError(
        "Prompt depth file not found for "
        f"{image_path}. Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def _case_paths(output_dir: Path, image_path: Path) -> CasePaths:
    """Create deterministic artifact paths for one image."""
    case_dir = output_dir / image_path.stem
    stem = image_path.stem
    return CasePaths(
        case_dir=case_dir,
        scene_ply=case_dir / f"{stem}.ply",
        video=case_dir / f"{stem}.mp4",
        html=case_dir / f"{stem}.html",
    )


def _resolve_video_render_geometry(
    intrinsics_px: torch.Tensor,
    original_width: int,
    original_height: int,
) -> tuple[tuple[int, int], torch.Tensor]:
    """Resolve capped video render shape and scaled camera intrinsics.

    Args:
        intrinsics_px: Pixel-space camera intrinsics with shape [3, 3].
        original_width: Original image width in pixels.
        original_height: Original image height in pixels.
    Returns:
        A tuple containing render image shape as (height, width) and pixel-space
        intrinsics with shape [3, 3].
    """
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid original image size: {original_width}x{original_height}")

    scale = min(
        1.0,
        float(DEFAULT_MAX_RENDER_LONG_EDGE) / float(max(original_width, original_height)),
        math.sqrt(float(DEFAULT_MAX_RENDER_PIXELS) / float(original_width * original_height)),
    )

    if scale >= 1.0:
        return (original_height, original_width), intrinsics_px

    render_width = max(2, int(math.floor(original_width * scale)))
    render_height = max(2, int(math.floor(original_height * scale)))
    # Keep video dimensions even for common yuv420 encoders.
    render_width = max(2, render_width - (render_width % 2))
    render_height = max(2, render_height - (render_height % 2))
    render_intrinsics_px = scale_intrinsics_px(
        intrinsics_px=intrinsics_px,
        src_width=original_width,
        src_height=original_height,
        dst_width=render_width,
        dst_height=render_height,
    )
    return (render_height, render_width), render_intrinsics_px


def _expected_artifacts_done(paths: CasePaths, args: argparse.Namespace) -> bool:
    """Return whether all requested artifacts already exist."""
    expected = [paths.scene_ply]
    if not args.no_video:
        expected.append(paths.video)
    if args.export_html:
        expected.append(paths.html)
    return all(path.exists() for path in expected)


def _needs_only_conversion(paths: CasePaths, args: argparse.Namespace) -> bool:
    """Return whether inference is done but requested converted outputs are missing."""
    if not paths.scene_ply.exists():
        return False
    if not args.no_video and not paths.video.exists():
        return False
    return not _expected_artifacts_done(paths, args)


def _disable_unavailable_optional_outputs(args: argparse.Namespace) -> None:
    """Skip optional outputs whose external dependencies are unavailable."""
    if not args.no_video and not is_gsplat_available():
        print("[batch] Skipping video rendering; optional gsplat is unavailable.")
        args.no_video = True
    if (
        args.export_html
        and shutil.which(SPLAT_TRANSFORM) is None
        and not Path(SPLAT_TRANSFORM).exists()
    ):
        print("[batch] Skipping HTML export; optional converter is unavailable.")
        args.export_html = False


def _build_splat_transform_command(
    scene_ply: Path,
    output_path: Path,
    viewer_settings: Path,
) -> list[str]:
    """Build the fixed SH0 HTML conversion command."""
    command = [
        SPLAT_TRANSFORM,
        "-w",
        "--viewer-settings",
        str(viewer_settings),
        str(scene_ply),
        "--filter-harmonics",
        "0",
        str(output_path),
    ]
    return command


def _run_splat_transform(
    scene_ply: Path,
    output_path: Path,
    viewer_settings: Path,
) -> None:
    """Convert one Gaussian PLY into a quiet, paused HTML viewer."""
    command = _build_splat_transform_command(scene_ply, output_path, viewer_settings)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    patch_supersplat_html_auto_rotate(output_path)


def _convert_scene_if_requested(
    export_html: bool,
    paths: CasePaths,
    viewer_settings: Path | None,
) -> None:
    """Convert scene.ply into the default HTML viewer when requested."""
    if not export_html:
        return
    if viewer_settings is None:
        raise RuntimeError("Viewer settings are required for HTML export.")
    _run_splat_transform(paths.scene_ply, paths.html, viewer_settings)


def _validate_batch_prompt_configuration(cfg: Any, args: argparse.Namespace) -> bool:
    """Validate prompt configuration for batch inference."""
    prompt_depth_enabled = (
        args.mode == "lidar"
        or args.prompt_depth is not None
        or args.prompt_depth_dir is not None
    )
    return validate_prompt_configuration(cfg, prompt_depth_enabled)


@torch.inference_mode()
def _run_one_image(
    args: argparse.Namespace,
    image_path: Path,
    paths: CasePaths,
    encoder: Any,
    decoder: Any,
    prompt_enabled: bool,
    device: torch.device,
    viewer_settings: Path | None,
) -> None:
    """Run inference and export artifacts for one input image."""
    paths.case_dir.mkdir(parents=True, exist_ok=True)

    image_bundle = load_demo_image_bundle(
        image_path=image_path,
        focal_length_px=args.focal_px,
        focal_length_mm=args.focal_mm,
        intrinsics_override_path=args.intrinsics_file,
    )
    original_height, original_width = image_bundle.original_image_shape
    _, inference_height, inference_width = image_bundle.inference_image.shape
    render_image_shape, render_intrinsics_px = _resolve_video_render_geometry(
        intrinsics_px=image_bundle.original_intrinsics.intrinsics_px,
        original_width=original_width,
        original_height=original_height,
    )
    prompt_inputs = None
    if prompt_enabled:
        prompt_depth_path = _resolve_prompt_depth_path(args, image_path)
        if prompt_depth_path is None:
            raise ValueError("Prompt-conditioned inference requires --prompt-depth or --prompt-depth-dir.")
        prompt_inputs = load_prompt_depth_tensors(
            prompt_depth_path=prompt_depth_path,
            image_shape=(inference_height, inference_width),
        )

    encoder_output = run_single_image_inference(
        encoder=encoder,
        image=image_bundle.inference_image,
        intrinsics_px=image_bundle.inference_intrinsics.intrinsics_px,
        device=device,
        prompt_inputs=prompt_inputs,
    )

    final_gaussians = encoder_output["gaussians"]
    if not args.disable_floater_filter:
        final_gaussians = filter_final_gaussian_floaters(final_gaussians)

    save_ply(
        gaussians=final_gaussians,
        f_px=image_bundle.inference_intrinsics.focal_length_px,
        image_shape=(inference_height, inference_width),
        path=paths.scene_ply,
    )

    if not args.no_video:
        render_novel_view_video_from_single_view(
            decoder=decoder,
            gaussians=final_gaussians,
            render_intrinsics_px=render_intrinsics_px,
            render_image_shape=render_image_shape,
            output_path=paths.video,
        )

    _convert_scene_if_requested(args.export_html, paths, viewer_settings)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run the batch inference pipeline."""
    images = _collect_images(args)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    _disable_unavailable_optional_outputs(args)
    viewer_settings = VIEWER_SETTINGS if args.export_html else None

    succeeded = 0
    skipped = 0
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36, complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
    )
    with progress:
        task = progress.add_task("Checking outputs", total=len(images))
        pending_inference: list[tuple[Path, CasePaths]] = []
        for image_path in images:
            paths = _case_paths(output_dir, image_path)
            if not args.overwrite and _expected_artifacts_done(paths, args):
                skipped += 1
                progress.advance(task)
                continue

            if not args.overwrite and _needs_only_conversion(paths, args):
                try:
                    _convert_scene_if_requested(
                        args.export_html,
                        paths,
                        viewer_settings,
                    )
                    succeeded += 1
                finally:
                    progress.advance(task)
                continue

            pending_inference.append((image_path, paths))

        if pending_inference:
            progress.update(task, description="Preparing model")
            cfg = load_demo_config(MODE_EXPERIMENTS[args.mode])
            prompt_enabled = _validate_batch_prompt_configuration(cfg, args)
            checkpoint_path = _resolve_checkpoint_path(
                args.checkpoint or MODE_CHECKPOINTS[args.mode]
            )
            device = _resolve_device(args.device)
            encoder, decoder = load_demo_model(
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            progress.update(task, description="Running inference")

            for image_path, paths in pending_inference:
                try:
                    _run_one_image(
                        args=args,
                        image_path=image_path,
                        paths=paths,
                        encoder=encoder,
                        decoder=decoder,
                        prompt_enabled=prompt_enabled,
                        device=device,
                        viewer_settings=viewer_settings,
                    )
                    succeeded += 1
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    progress.advance(task)

        progress.update(task, description="Complete")

    return {
        "status": "success",
        "succeeded": succeeded,
        "skipped": skipped,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = _parse_args()
    result = run_batch(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
