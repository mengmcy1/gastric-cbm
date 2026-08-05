"""Compare frozen and cross-hardware SHARP PLYs with identical uncompressed views.

The frozen PLY defines the camera model and trajectory. Both PLYs are rendered on
the same CUDA device with the same left, strict-zero center, and right cameras.
Only cropped PNG frames and machine-readable comparison metrics are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARP_SRC = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
if str(SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(SHARP_SRC))

# Make the Conda CUDA toolkit discoverable on Windows before gsplat is imported.
CONDA_PREFIX = Path(sys.prefix)
CUDA_BIN = CONDA_PREFIX / "Library" / "bin"
if (CUDA_BIN / "nvcc.exe").is_file():
    os.environ.setdefault("CUDA_HOME", str(CONDA_PREFIX / "Library"))
    os.environ["PATH"] = str(CUDA_BIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(CUDA_BIN))

from sharp.utils import camera, gsplat  # noqa: E402
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply  # noqa: E402


SSIM_MAX_SIDE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-ply", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-disparity", type=float, default=0.04)
    parser.add_argument("--crop-single-side-percent", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_crop(frame: np.ndarray, crop_percent: int) -> np.ndarray:
    if crop_percent == 0:
        return frame
    height, width = frame.shape[:2]
    ratio = crop_percent / 100.0
    left, top = round(width * ratio), round(height * ratio)
    right, bottom = round(width * (1 - ratio)), round(height * (1 - ratio))
    cropped = frame[top:bottom, left:right]
    return np.asarray(
        Image.fromarray(cropped).resize((width, height), Image.Resampling.LANCZOS)
    )


def resized_for_ssim(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, SSIM_MAX_SIDE / max(width, height))
    if scale == 1.0:
        return frame
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(frame).resize(target, Image.Resampling.BICUBIC))


def compare_frames(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"帧形状不一致：{left.shape} != {right.shape}")
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    absolute = np.abs(left_float - right_float)
    pixel_max = absolute.max(axis=2)
    mse = float(np.mean((left_float - right_float) ** 2))
    psnr = None if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    left_ssim = resized_for_ssim(left)
    right_ssim = resized_for_ssim(right)
    ssim = structural_similarity(
        left_ssim,
        right_ssim,
        data_range=255,
        channel_axis=2,
    )
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "psnr_db_full_resolution": psnr,
        "psnr_is_infinite": mse == 0,
        "ssim_max_side_512": float(ssim),
        "mean_absolute_channel_difference": float(absolute.mean()),
        "max_absolute_channel_difference": int(absolute.max()),
        "fraction_pixels_any_channel_gt_1": float(np.mean(pixel_max > 1)),
        "fraction_pixels_any_channel_gt_5": float(np.mean(pixel_max > 5)),
        "fraction_pixels_any_channel_gt_10": float(np.mean(pixel_max > 10)),
        "comparison_resolution": [int(left.shape[1]), int(left.shape[0])],
        "ssim_resolution": [int(left_ssim.shape[1]), int(left_ssim.shape[0])],
    }


def metadata_record(metadata: SceneMetaData) -> dict[str, Any]:
    return {
        "focal_length_px": float(metadata.focal_length_px),
        "resolution_px": [int(value) for value in metadata.resolution_px],
        "color_space": metadata.color_space,
    }


def render_frame(
    renderer: gsplat.GSplatRenderer,
    gaussians: Gaussians3D,
    camera_info: Any,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        result = renderer(
            gaussians,
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
    torch.cuda.synchronize(device)
    return (
        result.color[0]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def main() -> None:
    args = parse_args()
    frozen_path = args.frozen_ply.resolve()
    candidate_path = args.candidate_ply.resolve()
    for path in (frozen_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA，无法运行 gsplat 一致性检查。")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)

    frozen_cpu, frozen_metadata = load_ply(frozen_path)
    candidate_cpu, candidate_metadata = load_ply(candidate_path)
    if frozen_metadata != candidate_metadata:
        raise ValueError(
            "PLY 元数据不一致："
            f"{frozen_metadata!r} != {candidate_metadata!r}"
        )
    if frozen_cpu.mean_vectors.shape != candidate_cpu.mean_vectors.shape:
        raise ValueError(
            "高斯数量或形状不一致："
            f"{frozen_cpu.mean_vectors.shape} != {candidate_cpu.mean_vectors.shape}"
        )

    width, height = (int(value) for value in frozen_metadata.resolution_px)
    focal = float(frozen_metadata.focal_length_px)
    intrinsics = torch.tensor(
        [
            [focal, 0, (width - 1) / 2, 0],
            [0, focal, (height - 1) / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    camera_model = camera.create_camera_model(
        frozen_cpu,
        intrinsics,
        resolution_px=frozen_metadata.resolution_px,
    )
    trajectory = camera.create_eye_trajectory(
        frozen_cpu,
        camera.TrajectoryParams(
            type="swipe",
            max_disparity=args.max_disparity,
            num_steps=60,
        ),
        frozen_metadata.resolution_px,
        focal,
    )
    eyes = {
        "left": trajectory[0],
        "center": torch.zeros(3, dtype=torch.float32),
        "right": trajectory[-1],
    }

    frozen = frozen_cpu.to(device)
    candidate = candidate_cpu.to(device)
    renderer = gsplat.GSplatRenderer(color_space=frozen_metadata.color_space)

    view_results: dict[str, Any] = {}
    for label, eye in eyes.items():
        camera_info = camera_model.compute(eye)
        frozen_frame = apply_crop(
            render_frame(renderer, frozen, camera_info, device),
            args.crop_single_side_percent,
        )
        candidate_frame = apply_crop(
            render_frame(renderer, candidate, camera_info, device),
            args.crop_single_side_percent,
        )
        frozen_name = f"frozen_{label}_crop{args.crop_single_side_percent:02d}.png"
        candidate_name = (
            f"candidate_{label}_crop{args.crop_single_side_percent:02d}.png"
        )
        iio.imwrite(output_dir / frozen_name, frozen_frame)
        iio.imwrite(output_dir / candidate_name, candidate_frame)
        view_results[label] = {
            "eye": [float(value) for value in eye.tolist()],
            "frozen_png": frozen_name,
            "candidate_png": candidate_name,
            **compare_frames(frozen_frame, candidate_frame),
        }

    result = {
        "schema_version": "1.0-cross-hardware-render",
        "comparison": "same GPU, same camera, uncompressed cropped PNG",
        "frozen_ply": {
            "path": str(frozen_path),
            "bytes": frozen_path.stat().st_size,
            "sha256": sha256(frozen_path),
        },
        "candidate_ply": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256(candidate_path),
        },
        "metadata": metadata_record(frozen_metadata),
        "fixed_variables": {
            "max_disparity": args.max_disparity,
            "crop_single_side_percent": args.crop_single_side_percent,
            "gaussian_keep_percent": 100,
            "trajectory": "swipe",
            "trajectory_steps_used_to_define_endpoints": 60,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "views": view_results,
    }
    metrics_path = output_dir / "cross_hardware_render_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": str(metrics_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
