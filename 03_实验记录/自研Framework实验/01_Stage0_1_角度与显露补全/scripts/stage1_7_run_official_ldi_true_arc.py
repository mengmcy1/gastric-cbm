#!/usr/bin/env python3
"""Run the official 3D Photo LDI construction with the frozen P01 true-arc path.

The official repository is imported read-only. Compatibility aliases live here so
the pinned source commit remains unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--center-depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focal-px-original", type=float, default=3195.073)
    parser.add_argument("--reference-depth", type=float, default=5.203962802886963)
    parser.add_argument("--angle-total-deg", type=float, default=30.0)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--longer-side", type=int, default=768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-mesh", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def true_arc_poses(reference_depth: float, angle_total_deg: float, count: int) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for angle_deg in np.linspace(-angle_total_deg / 2.0, angle_total_deg / 2.0, count):
        angle = math.radians(float(angle_deg))
        pose = np.eye(4, dtype=np.float64)
        # Official mesh/render coordinates look toward -Z.  These C2W poses orbit
        # around (0, 0, -reference_depth) while keeping that point centered.
        pose[:3, :3] = rotation_y(angle)
        pose[:3, 3] = np.array(
            [reference_depth * math.sin(angle), 0.0, reference_depth * (math.cos(angle) - 1.0)]
        )
        poses.append(pose)
    return poses


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    model_root = args.model_root.resolve()
    image_path = args.image.resolve()
    depth_path = args.center_depth.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{output}")
    required_source = [source_root / name for name in ("mesh.py", "networks.py", "argument.yml")]
    required_models = [model_root / name for name in ("edge-model.pth", "depth-model.pth", "color-model.pth")]
    for path in [*required_source, *required_models, image_path, depth_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    reuse_mesh = args.reuse_mesh.resolve() if args.reuse_mesh is not None else None
    if reuse_mesh is not None and not reuse_mesh.is_file():
        raise FileNotFoundError(reuse_mesh)
    if args.num_frames < 3 or args.num_frames % 2 == 0:
        raise ValueError("--num-frames 必须是至少3的奇数，确保存在原图中心帧。")
    if args.longer_side < 256:
        raise ValueError("--longer-side 必须不小于256。")

    output.mkdir(parents=True)
    mesh_dir = output / "mesh"
    video_dir = output / "video"
    review_dir = output / "review"
    mesh_dir.mkdir()
    video_dir.mkdir()
    review_dir.mkdir()

    # Removed aliases are required by the frozen 2020 source.  Do not patch the
    # external checkout itself.
    np.int = int  # type: ignore[attr-defined]
    np.bool = bool  # type: ignore[attr-defined]
    np.float = float  # type: ignore[attr-defined]
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(source_root))
    import vispy

    vispy.use(app="egl")
    from bilateral_filtering import sparse_bilateral_filtering
    from mesh import output_3d_photo, read_ply, write_ply
    from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但当前进程不可用。")
    if device.type == "cuda" and device.index not in (None, 0):
        raise ValueError("请用CUDA_VISIBLE_DEVICES选择物理GPU，脚本内--device固定为cuda:0。")
    official_gpu_id = 0 if device.type == "cuda" else -1

    config = yaml.safe_load((source_root / "argument.yml").read_text(encoding="utf-8"))
    config.update(
        {
            "depth_edge_model_ckpt": str(model_root / "edge-model.pth"),
            "depth_feat_model_ckpt": str(model_root / "depth-model.pth"),
            "rgb_feat_model_ckpt": str(model_root / "color-model.pth"),
            "fps": args.fps,
            "num_frames": args.num_frames,
            "longer_side_len": args.longer_side,
            "save_ply": True,
            "load_ply": False,
            "inference_video": True,
            # Frozen official helpers only recognize a non-negative integer as
            # CUDA; a torch.device/string silently routes their tensors to CPU.
            "gpu_ids": official_gpu_id,
            "offscreen_rendering": True,
            "crop_border": [0.0, 0.0, 0.0, 0.0],
        }
    )

    original = np.asarray(Image.open(image_path).convert("RGB"))
    original_h, original_w = original.shape[:2]
    scale = args.longer_side / max(original_h, original_w)
    work_w, work_h = round(original_w * scale), round(original_h * scale)
    image = cv2.resize(original, (work_w, work_h), interpolation=cv2.INTER_AREA)
    source_depth = np.load(depth_path).astype(np.float32)
    invalid = ~np.isfinite(source_depth) | (source_depth <= 0)
    if invalid.all():
        raise ValueError("中心深度没有有效正值。")
    source_depth[invalid] = float(np.median(source_depth[~invalid]))
    depth = cv2.resize(source_depth, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("缩放后的中心深度非法。")

    intrinsic = np.array(
        [
            [args.focal_px_original / original_w, 0.0, 0.5],
            [0.0, args.focal_px_original / original_h, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    start = time.perf_counter()
    filtered_rgb, filtered_depth = sparse_bilateral_filtering(
        depth.copy(), image.copy(), config, num_iter=config["sparse_iter"], spdb=False
    )
    image = filtered_rgb[-1]
    depth = filtered_depth[-1]
    preprocessing_seconds = time.perf_counter() - start
    reference_depth = float(args.reference_depth)
    if not math.isfinite(reference_depth) or reference_depth <= 0:
        raise ValueError("--reference-depth 必须是有限正值。")

    mesh_path = mesh_dir / "p01_official_ldi.ply"
    if reuse_mesh is None:
        load_start = time.perf_counter()
        edge_model = Inpaint_Edge_Net(init_weights=False).to(device)
        depth_model = Inpaint_Depth_Net().to(device)
        color_model = Inpaint_Color_Net().to(device)
        model_specs = [
            (edge_model, model_root / "edge-model.pth"),
            (depth_model, model_root / "depth-model.pth"),
            (color_model, model_root / "color-model.pth"),
        ]
        for model, checkpoint in model_specs:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            model.eval()
        load_seconds = time.perf_counter() - load_start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        mesh_start = time.perf_counter()
        result = write_ply(
            image,
            depth,
            intrinsic,
            str(mesh_path),
            config,
            color_model,
            edge_model,
            edge_model,
            depth_model,
        )
        if result is False or not mesh_path.is_file():
            raise RuntimeError("官方write_ply未生成完整LDI mesh。")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        mesh_seconds = time.perf_counter() - mesh_start
        peak_cuda_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        del edge_model, depth_model, color_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        load_seconds = 0.0
        mesh_start = time.perf_counter()
        shutil.copyfile(reuse_mesh, mesh_path)
        mesh_seconds = time.perf_counter() - mesh_start
        peak_cuda_memory = 0
    verts, colors, faces, height, width, hfov, vfov = read_ply(str(mesh_path))
    poses = true_arc_poses(reference_depth, args.angle_total_deg, args.num_frames)
    render_start = time.perf_counter()
    output_3d_photo(
        verts,
        colors,
        faces,
        height,
        width,
        hfov,
        vfov,
        np.eye(4),
        ["true-arc-30"],
        np.eye(4),
        str(video_dir),
        image,
        intrinsic,
        config,
        image,
        [poses],
        ["p01"],
        work_h,
        work_w,
        border=[0, work_h, 0, work_w],
        depth=depth,
        mean_loc_depth=reference_depth,
    )
    render_seconds = time.perf_counter() - render_start
    moviepy_video_path = video_dir / "p01_true-arc-30.mp4"
    if not moviepy_video_path.is_file():
        raise FileNotFoundError(moviepy_video_path)
    # MoviePy 1.0.3 can encode one duplicate tail frame for an exact frame list.
    # Preserve that raw diagnostic and produce an exact frozen 61-frame artifact.
    video_path = video_dir / f"p01_true-arc-30_{args.num_frames}f.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(moviepy_video_path),
            "-frames:v",
            str(args.num_frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: dict[int, np.ndarray] = {}
    wanted = {0, args.num_frames // 2, args.num_frames - 1}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    if frame_count != args.num_frames or index != args.num_frames:
        raise RuntimeError(
            f"canonical视频帧数错误：metadata={frame_count}, decoded={index}, expected={args.num_frames}"
        )
    if wanted != set(frames):
        raise RuntimeError(f"视频帧不完整：期望{sorted(wanted)}，实际{sorted(frames)}")
    labels = {0: "left_-15deg", args.num_frames // 2: "center_0deg", args.num_frames - 1: "right_+15deg"}
    for idx, frame in frames.items():
        Image.fromarray(frame).save(review_dir / f"{idx:03d}_{labels[idx]}.png")
    strip = np.concatenate([frames[idx] for idx in sorted(frames)], axis=1)
    Image.fromarray(strip).save(review_dir / "P01_official_LDI_true_arc30_endpoints.png")

    record = {
        "schema_version": "stage1.7-official-ldi-true-arc-v1",
        "status": "awaiting_manual_review",
        "producer_machine_id": "linux5080",
        "method_boundary": "official full LDI construction and iterative local edge/depth/color inpainting; project wrapper supplies frozen P01 depth, calibrated intrinsics, and true_arc poses",
        "official_source": {
            "root": str(source_root),
            "commit": subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "mesh_reuse": None
        if reuse_mesh is None
        else {"source_path": str(reuse_mesh), "source_sha256": sha256(reuse_mesh)},
        "inputs": {
            "image": str(image_path),
            "image_sha256": sha256(image_path),
            "center_depth": str(depth_path),
            "center_depth_sha256": sha256(depth_path),
            "focal_px_original": args.focal_px_original,
        },
        "protocol": {
            "trajectory": "true_arc",
            "angle_total_deg": args.angle_total_deg,
            "endpoint_angles_deg": [-args.angle_total_deg / 2.0, args.angle_total_deg / 2.0],
            "num_frames": args.num_frames,
            "fps": args.fps,
            "working_resolution_wh": [work_w, work_h],
            "crop_border": [0.0, 0.0, 0.0, 0.0],
            "reference_depth": reference_depth,
        },
        "models": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in required_models},
        "runtime_seconds": {
            "preprocessing": preprocessing_seconds,
            "model_loading": load_seconds,
            "ldi_mesh": mesh_seconds,
            "render_and_encode": render_seconds,
        },
        "peak_cuda_memory_bytes_mesh_stage": peak_cuda_memory,
        "artifacts": {
            "mesh": {"path": str(mesh_path), "bytes": mesh_path.stat().st_size, "sha256": sha256(mesh_path)},
            "video": {"path": str(video_path), "bytes": video_path.stat().st_size, "sha256": sha256(video_path), "decoded_frames": frame_count},
            "moviepy_raw_video": {
                "path": str(moviepy_video_path),
                "bytes": moviepy_video_path.stat().st_size,
                "sha256": sha256(moviepy_video_path),
            },
            "review_strip": {
                "path": str(review_dir / "P01_official_LDI_true_arc30_endpoints.png"),
                "bytes": (review_dir / "P01_official_LDI_true_arc30_endpoints.png").stat().st_size,
                "sha256": sha256(review_dir / "P01_official_LDI_true_arc30_endpoints.png"),
            },
        },
        "quality_boundary": "manual visual review pending; no claim of Stage 1 or 30-degree quality pass",
    }
    (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
