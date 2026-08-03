"""Run one isolated SHARP generation or resident-GPU rendering measurement."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on native Windows
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARP_SRC = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
if str(SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(SHARP_SRC))

from sharp.models import PredictorParams, create_predictor  # noqa: E402
from sharp.utils import camera, gsplat, io as sharp_io  # noqa: E402
from sharp.utils.gaussians import (  # noqa: E402
    load_ply,
    save_ply,
    unproject_gaussians,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generation", "render"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--max-disparity", type=float, default=0.04)
    parser.add_argument("--crop-single-side-percent", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--run-kind", choices=("warmup", "formal"), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    return parser.parse_args()


def ensure_new_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def synchronize() -> None:
    torch.cuda.synchronize(torch.device("cuda:0"))


def peak_rss_mb() -> float | None:
    if sys.platform.startswith("win"):
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        return round(float(peak) / (1024**2), 3) if peak is not None else None
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(usage) / 1024.0, 3)


def memory_snapshot() -> dict[str, Any]:
    process = psutil.Process()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "rss_mb": round(process.memory_info().rss / (1024**2), 3),
        "peak_rss_mb": peak_rss_mb(),
        "system_available_mb": round(psutil.virtual_memory().available / (1024**2), 3),
        "cuda_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 3),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 3),
        "cuda_free_mb": round(free_bytes / (1024**2), 3),
        "cuda_total_mb": round(total_bytes / (1024**2), 3),
    }


def environment_record(args: argparse.Namespace) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "profile": args.profile,
        "run_kind": args.run_kind,
        "repeat_index": args.repeat_index,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index_requested": args.physical_gpu_index,
        "process_device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_mb": round(props.total_memory / (1024**2), 3),
    }


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


def summarize_values(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": round(ordered[p95_index], 6),
        "max": round(max(values), 6),
    }


def run_generation(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.input_image is None or args.checkpoint is None:
        raise ValueError("generation 模式需要 --input-image 和 --checkpoint")
    for path in (args.input_image, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    timings: dict[str, float] = {}
    memory: dict[str, Any] = {"start": memory_snapshot()}
    total_start = time.perf_counter()

    start = time.perf_counter()
    state_dict = torch.load(args.checkpoint, weights_only=True)
    timings["checkpoint_load_s"] = time.perf_counter() - start
    memory["after_checkpoint_load"] = memory_snapshot()

    start = time.perf_counter()
    predictor = create_predictor(PredictorParams())
    timings["model_construct_s"] = time.perf_counter() - start
    memory["after_model_construct"] = memory_snapshot()

    start = time.perf_counter()
    predictor.load_state_dict(state_dict)
    predictor.eval()
    timings["state_dict_load_s"] = time.perf_counter() - start
    memory["after_state_dict_load"] = memory_snapshot()

    start = time.perf_counter()
    predictor.to(torch.device("cuda:0"))
    synchronize()
    timings["model_upload_s"] = time.perf_counter() - start
    memory["after_model_upload"] = memory_snapshot()

    start = time.perf_counter()
    image, _, focal_length_px = sharp_io.load_rgb(args.input_image)
    timings["input_load_s"] = time.perf_counter() - start
    height, width = image.shape[:2]

    start = time.perf_counter()
    image_pt = (
        torch.from_numpy(image.copy())
        .float()
        .to(torch.device("cuda:0"))
        .permute(2, 0, 1)
        / 255.0
    )
    disparity_factor = torch.tensor(
        [focal_length_px / width], dtype=torch.float32, device="cuda:0"
    )
    image_resized = F.interpolate(
        image_pt[None],
        size=(1536, 1536),
        mode="bilinear",
        align_corners=True,
    )
    synchronize()
    timings["preprocess_s"] = time.perf_counter() - start
    memory["after_preprocess"] = memory_snapshot()

    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    with torch.inference_mode():
        gaussians_ndc = predictor(image_resized, disparity_factor)
    synchronize()
    timings["network_forward_s"] = time.perf_counter() - start
    memory["after_network_forward"] = memory_snapshot()
    network_peak_allocated = torch.cuda.max_memory_allocated(0)
    network_peak_reserved = torch.cuda.max_memory_reserved(0)

    intrinsics = torch.tensor(
        [
            [focal_length_px, 0, width / 2, 0],
            [0, focal_length_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )
    intrinsics[0] *= 1536 / width
    intrinsics[1] *= 1536 / height

    start = time.perf_counter()
    with torch.inference_mode():
        gaussians = unproject_gaussians(
            gaussians_ndc,
            torch.eye(4, device="cuda:0"),
            intrinsics,
            (1536, 1536),
        )
    synchronize()
    timings["unproject_postprocess_s"] = time.perf_counter() - start
    memory["after_unproject"] = memory_snapshot()

    ply_path = output_dir / "scene_full.ply"
    start = time.perf_counter()
    save_ply(gaussians, focal_length_px, (height, width), ply_path)
    synchronize()
    timings["ply_write_s"] = time.perf_counter() - start
    memory["after_ply_write"] = memory_snapshot()
    timings["measured_pipeline_total_s"] = time.perf_counter() - total_start

    return {
        "mode": "generation",
        "timings_s": {key: round(value, 6) for key, value in timings.items()},
        "memory": memory,
        "network_peak_cuda_allocated_mb": round(network_peak_allocated / (1024**2), 3),
        "network_peak_cuda_reserved_mb": round(network_peak_reserved / (1024**2), 3),
        "peak_rss_mb": peak_rss_mb(),
        "input_resolution": [width, height],
        "internal_resolution": [1536, 1536],
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "ply_bytes": ply_path.stat().st_size,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "state_dict_retention": "Retained through the measured pipeline to match the current official CLI scope.",
    }


def run_render(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.ply is None or not args.ply.is_file():
        raise FileNotFoundError(args.ply)
    timings: dict[str, float] = {}
    memory: dict[str, Any] = {"start": memory_snapshot()}
    total_start = time.perf_counter()

    start = time.perf_counter()
    gaussians_cpu, metadata = load_ply(args.ply)
    timings["ply_load_s"] = time.perf_counter() - start
    memory["after_ply_load"] = memory_snapshot()

    start = time.perf_counter()
    gaussians = gaussians_cpu.to(torch.device("cuda:0"))
    synchronize()
    timings["gpu_upload_s"] = time.perf_counter() - start
    memory["after_gpu_upload"] = memory_snapshot()

    width, height = (int(value) for value in metadata.resolution_px)
    focal_length_px = float(metadata.focal_length_px)
    intrinsics = torch.tensor(
        [
            [focal_length_px, 0, (width - 1) / 2, 0],
            [0, focal_length_px, (height - 1) / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px
    )
    trajectory = camera.create_eye_trajectory(
        gaussians,
        camera.TrajectoryParams(
            type="swipe",
            max_disparity=args.max_disparity,
            num_steps=args.num_steps,
        ),
        metadata.resolution_px,
        focal_length_px,
    )
    cameras = [camera_model.compute(eye) for eye in trajectory]
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

    warm_camera = cameras[0]
    start = time.perf_counter()
    with torch.inference_mode():
        renderer(
            gaussians,
            extrinsics=warm_camera.extrinsics[None].to("cuda:0"),
            intrinsics=warm_camera.intrinsics[None].to("cuda:0"),
            image_width=warm_camera.width,
            image_height=warm_camera.height,
        )
    synchronize()
    timings["renderer_warmup_s"] = time.perf_counter() - start

    torch.cuda.reset_peak_memory_stats(0)
    frame_ms: list[float] = []
    pure_start = time.perf_counter()
    with torch.inference_mode():
        for camera_info in cameras:
            start = time.perf_counter()
            renderer(
                gaussians,
                extrinsics=camera_info.extrinsics[None].to("cuda:0"),
                intrinsics=camera_info.intrinsics[None].to("cuda:0"),
                image_width=camera_info.width,
                image_height=camera_info.height,
            )
            synchronize()
            frame_ms.append((time.perf_counter() - start) * 1000)
    pure_total = time.perf_counter() - pure_start
    pure_peak_allocated = torch.cuda.max_memory_allocated(0)
    pure_peak_reserved = torch.cuda.max_memory_reserved(0)
    memory["after_pure_render"] = memory_snapshot()

    frames: list[np.ndarray] = []
    transfer_start = time.perf_counter()
    with torch.inference_mode():
        for camera_info in cameras:
            result = renderer(
                gaussians,
                extrinsics=camera_info.extrinsics[None].to("cuda:0"),
                intrinsics=camera_info.intrinsics[None].to("cuda:0"),
                image_width=camera_info.width,
                image_height=camera_info.height,
            )
            color = (
                result.color[0].permute(1, 2, 0).clamp(0, 1).mul(255)
                .to(torch.uint8).cpu().numpy()
            )
            frames.append(apply_crop(color, args.crop_single_side_percent))
    synchronize()
    timings["render_transfer_crop_60_s"] = time.perf_counter() - transfer_start
    memory["after_frame_collection"] = memory_snapshot()

    video_path = output_dir / "color.mp4"
    encode_start = time.perf_counter()
    writer = iio.get_writer(
        video_path,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    timings["video_encode_60_s"] = time.perf_counter() - encode_start
    memory["after_video_encode"] = memory_snapshot()
    timings["measured_pipeline_total_s"] = time.perf_counter() - total_start

    reader = iio.get_reader(video_path)
    try:
        video_meta = reader.get_meta_data()
        encoded_frames = reader.count_frames()
    finally:
        reader.close()
    encoded_size = video_meta.get("size") or video_meta.get("source_size")

    return {
        "mode": "render",
        "timings_s": {key: round(value, 6) for key, value in timings.items()},
        "pure_render_60_s": round(pure_total, 6),
        "pure_render_fps": round(len(cameras) / pure_total, 6),
        "pure_render_frame_ms": summarize_values(frame_ms),
        "pure_render_peak_cuda_allocated_mb": round(
            pure_peak_allocated / (1024**2), 3
        ),
        "pure_render_peak_cuda_reserved_mb": round(
            pure_peak_reserved / (1024**2), 3
        ),
        "memory": memory,
        "peak_rss_mb": peak_rss_mb(),
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "ply_bytes": args.ply.stat().st_size,
        "render_resolution": [width, height],
        "num_steps": len(cameras),
        "encoded_video": {
            "path": video_path.name,
            "bytes": video_path.stat().st_size,
            "frame_count": int(encoded_frames),
            "resolution": [int(encoded_size[0]), int(encoded_size[1])],
            "fps": args.fps,
            "codec": "libx264",
            "pixel_format": "yuv420p",
        },
        "encoding_boundary": "Pre-rendered cropped RGB frames in CPU memory to H.264; rendering and GPU-to-CPU transfer are excluded.",
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_new_output_dir(args.output_dir)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "性能 worker 要求 CUDA_VISIBLE_DEVICES 只暴露一张已选 GPU。"
        )
    torch.cuda.set_device(0)
    synchronize()
    environment = environment_record(args)
    torch.cuda.reset_peak_memory_stats(0)
    process_start = time.perf_counter()
    if args.mode == "generation":
        result = run_generation(args, output_dir)
    else:
        result = run_render(args, output_dir)
    result["worker_wall_s"] = round(time.perf_counter() - process_start, 6)
    result["environment"] = environment
    result["fixed_variables"] = {
        "max_disparity": args.max_disparity,
        "crop_single_side_percent": args.crop_single_side_percent,
        "gaussian_keep_percent": 100,
        "trajectory": "swipe",
        "num_steps": args.num_steps,
        "fps": args.fps,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": str(output_dir / "metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
