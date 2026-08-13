from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image

from stage01_common import REPO_ROOT, file_hash, prepare_output_dir, read_json, write_json


EXPECTED_SOURCE_COMMIT = "de0446740a3726f3de76c32e78b43bd985d987f9"
EXPECTED_MODEL_SHA256 = {
    "edge": "B1D768BD008AD5FE9F540004F870B8C3D355E4939B2009AA4DB493FD313217C9",
    "depth": "2D0E63E89A22762DDFA8BC8C9F8C992E5532B140123274FFC6E4171BAA1B76F8",
    "color": "383C9B1DB70097907A6F9C8ABB0303E7056F50D5456A36F34AB784592B8B2C20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 3D Photo Inpainting 官方三网络执行冻结 P01 端点的结构/深度/RGB 适配实验。"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--context-width", type=int, default=256)
    parser.add_argument("--crop-margin", type=int, default=256)
    parser.add_argument("--background-quantile", type=float, default=0.40)
    parser.add_argument("--edge-quantile", type=float, default=0.88)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_manifest_file(entry: dict, label: str) -> Path:
    path = resolve_repo_path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    observed = file_hash(path)
    if observed != entry["sha256"].upper():
        raise ValueError(f"{label} SHA256 不匹配：{observed} != {entry['sha256']}")
    return path


def verify_models(model_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, expected in EXPECTED_MODEL_SHA256.items():
        path = model_root / f"{name}-model.pth"
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = file_hash(path)
        if observed != expected:
            raise ValueError(f"{name} 权重 SHA256 不匹配：{observed} != {expected}")
        paths[name] = path
    return paths


def crop_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("冻结 accepted mask 为空。")
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + 1 + margin),
        min(height, int(ys.max()) + 1 + margin),
    )


def resize_working(
    rgb: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    height, width = mask.shape
    scale = min(1.0, max_side / max(height, width))
    new_width = max(128, int(round(width * scale / 128)) * 128)
    new_height = max(128, int(round(height * scale / 128)) * 128)
    return (
        cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA),
        cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0,
        cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST),
        scale,
    )


def make_context_and_edges(
    mask: np.ndarray,
    depth: np.ndarray,
    context_width: int,
    background_quantile: float,
    edge_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    finite = np.isfinite(depth) & (depth > 0)
    kernel3 = np.ones((3, 3), np.uint8)
    boundary = (cv2.dilate(mask.astype(np.uint8), kernel3, iterations=3) > 0) & ~mask & finite
    boundary_values = depth[boundary]
    if boundary_values.size < 32:
        boundary = ~mask & finite
        boundary_values = depth[boundary]
    threshold = float(np.quantile(boundary_values, background_quantile))

    radius = max(1, int(round(context_width / 2)))
    kernel_size = radius * 2 + 1
    local = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)) > 0
    background = finite & (depth >= threshold)
    context = local & ~mask & background
    # A too-aggressive depth split can starve partial convolution. Keep the
    # local known ring as a deterministic fallback and record that fallback.
    fallback = False
    if int(context.sum()) < max(1024, int(mask.sum() * 0.25)):
        context = local & ~mask & finite
        fallback = True

    safe_depth = np.where(finite, depth, threshold).astype(np.float32)
    log_depth = np.log(np.maximum(safe_depth, 1e-6))
    gy, gx = np.gradient(log_depth)
    magnitude = np.hypot(gx, gy)
    edge_values = magnitude[context]
    edge_threshold = float(np.quantile(edge_values, edge_quantile)) if edge_values.size else 0.0
    edge = (magnitude >= edge_threshold) & context
    return context, edge, log_depth, {
        "background_depth_threshold": threshold,
        "background_quantile": background_quantile,
        "edge_gradient_threshold": edge_threshold,
        "edge_quantile": edge_quantile,
        "context_fallback_to_all_local_known": fallback,
    }


def preview_float(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    sample = values[valid & np.isfinite(values)]
    if sample.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.quantile(sample, [0.02, 0.98])
    normalized = np.clip((values - low) / max(float(high - low), 1e-6), 0, 1)
    return np.round(normalized * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.max_side < 128 or args.max_side % 128:
        raise ValueError("--max-side 必须是不小于128的128整数倍。")
    if not 0 <= args.background_quantile <= 1 or not 0 <= args.edge_quantile <= 1:
        raise ValueError("quantile 参数必须位于[0,1]。")

    manifest = read_json(args.manifest)
    if manifest.get("experiment_id") != "P01_ang30_static_inpaint_ab_v1":
        raise ValueError("拒绝使用未冻结的静态 A/B 清单。")
    endpoint = manifest["endpoints"][args.side]
    image_path = verify_manifest_file(endpoint["image"], f"{args.side} image")
    mask_path = verify_manifest_file(endpoint["mask"], f"{args.side} mask")
    depth_path = args.depth.resolve()
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    depth_sha256 = file_hash(depth_path)

    source_root = args.source_root.resolve()
    if not (source_root / "networks.py").is_file():
        raise FileNotFoundError(f"3D Photo Inpainting 源码不完整：{source_root}")
    model_paths = verify_models(args.model_root.resolve())
    sys.path.insert(0, str(source_root))
    from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用。")

    source = np.asarray(Image.open(image_path).convert("RGB"))
    hole = np.asarray(Image.open(mask_path).convert("L")) > 0
    depth = np.load(depth_path).astype(np.float32)
    if source.shape[:2] != hole.shape or depth.shape != hole.shape:
        raise ValueError("RGB、accepted mask 与端点深度的分辨率不一致。")

    left, top, right, bottom = crop_box(hole, args.crop_margin)
    rgb_crop = source[top:bottom, left:right].astype(np.float32) / 255.0
    mask_crop = hole[top:bottom, left:right]
    depth_crop = depth[top:bottom, left:right]
    rgb_work, mask_work, depth_work, scale = resize_working(
        rgb_crop, mask_crop, depth_crop, args.max_side
    )
    effective_context_width = max(8, int(round(args.context_width * scale)))
    context, known_edge, log_depth, geometry = make_context_and_edges(
        mask_work,
        depth_work,
        effective_context_width,
        args.background_quantile,
        args.edge_quantile,
    )
    known_log = log_depth[context]
    if known_log.size == 0:
        raise ValueError("背景侧上下文为空。")
    mean_log_depth = float(known_log.mean())
    zero_mean_log = (log_depth - mean_log_depth) * context
    disparity = np.zeros_like(depth_work, dtype=np.float32)
    valid_context_depth = context & np.isfinite(depth_work) & (depth_work > 0)
    disparity[valid_context_depth] = 1.0 / depth_work[valid_context_depth]

    tensor = lambda value: torch.from_numpy(value.copy()).float()[None, None].to(device)
    rgb_tensor = torch.from_numpy((rgb_work * context[..., None]).copy()).float()
    rgb_tensor = rgb_tensor.permute(2, 0, 1)[None].to(device)
    mask_tensor = tensor(mask_work.astype(np.float32))
    context_tensor = tensor(context.astype(np.float32))
    disparity_tensor = tensor(disparity)
    edge_tensor = tensor(known_edge.astype(np.float32))
    log_tensor = tensor(zero_mean_log.astype(np.float32))

    constructors = {
        "edge": lambda: Inpaint_Edge_Net(init_weights=False),
        "depth": Inpaint_Depth_Net,
        "color": Inpaint_Color_Net,
    }
    models: dict[str, torch.nn.Module] = {}
    load_start = time.perf_counter()
    for name, constructor in constructors.items():
        model = constructor()
        model = model.to(device)
        # The official color model's train()/eval() override does not return self,
        # so these calls intentionally remain unchained.
        model.eval()
        state = torch.load(model_paths[name], map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        models[name] = model
    load_seconds = time.perf_counter() - load_start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    infer_start = time.perf_counter()
    with torch.inference_mode():
        predicted_edge = models["edge"].forward_3P(
            mask_tensor,
            context_tensor,
            rgb_tensor,
            disparity_tensor,
            edge_tensor,
            unit_length=128,
            cuda=device,
        )
        completed_edge = (predicted_edge > 0.5).float() * mask_tensor + edge_tensor
        predicted_log_depth = models["depth"].forward_3P(
            mask_tensor,
            context_tensor,
            log_tensor,
            completed_edge,
            unit_length=128,
            cuda=device,
        )
        predicted_color = models["color"].forward_3P(
            mask_tensor,
            context_tensor,
            rgb_tensor,
            completed_edge,
            unit_length=128,
            cuda=device,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - infer_start

    edge_work = completed_edge[0, 0].detach().cpu().numpy()
    depth_pred_work = np.exp(predicted_log_depth[0, 0].detach().cpu().numpy() + mean_log_depth)
    color_work = predicted_color[0].detach().cpu().permute(1, 2, 0).numpy()
    if not np.isfinite(depth_pred_work).all() or not np.isfinite(color_work).all():
        raise FloatingPointError("网络输出包含 NaN 或 Inf。")

    crop_height, crop_width = mask_crop.shape
    raw_crop = cv2.resize(color_work, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
    raw_crop = np.round(np.clip(raw_crop, 0, 1) * 255).astype(np.uint8)
    depth_pred_crop = cv2.resize(depth_pred_work, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
    edge_crop = cv2.resize(edge_work, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)
    known_edge_crop = cv2.resize(known_edge.astype(np.uint8), (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)
    context_crop = cv2.resize(context.astype(np.uint8), (crop_width, crop_height), interpolation=cv2.INTER_NEAREST) > 0

    raw_full = source.copy()
    raw_full[top:bottom, left:right] = raw_crop
    composited = source.copy()
    composited[hole] = raw_full[hole]
    if not np.array_equal(composited[~hole], source[~hole]):
        raise AssertionError("适配器输出改写了 accepted mask 外像素。")

    completed_depth = depth.copy()
    depth_prediction_full = np.full(depth.shape, np.nan, dtype=np.float32)
    depth_prediction_full[top:bottom, left:right] = depth_pred_crop.astype(np.float32)
    completed_depth[hole] = depth_prediction_full[hole]
    if not np.isfinite(completed_depth[hole]).all() or not (completed_depth[hole] > 0).all():
        raise FloatingPointError("洞区补全深度不是全部有限正数。")

    output_dir = prepare_output_dir(args.output_dir)
    iio.imwrite(output_dir / "context_background_side.png", context_crop.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "structure_known.png", known_edge_crop * 255)
    iio.imwrite(output_dir / "structure_completed.png", np.round(edge_crop * 255).astype(np.uint8))
    np.save(output_dir / "depth_completed_float32.npy", completed_depth)
    iio.imwrite(output_dir / "depth_completed_preview.png", preview_float(completed_depth, np.isfinite(completed_depth) & (completed_depth > 0)))
    iio.imwrite(output_dir / "inpaint_raw.png", raw_full)
    iio.imwrite(output_dir / "inpaint_composited.png", composited)

    write_json(
        output_dir / "adapter_run.json",
        {
            "schema_version": "1.0-stage1-3dphoto-adapter",
            "status": "success",
            "result_label": "3D Photo network fixed-endpoint adapter experiment; not an official full-pipeline reproduction",
            "experiment_id": manifest["experiment_id"],
            "side": args.side,
            "source": {
                "repository": "https://github.com/vt-vl-lab/3d-photo-inpainting",
                "commit": EXPECTED_SOURCE_COMMIT,
                "local_root": str(source_root),
            },
            "models": {
                name: {"path": str(path), "sha256": EXPECTED_MODEL_SHA256[name]}
                for name, path in model_paths.items()
            },
            "model_provenance": {
                "official_primary_host": "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/",
                "primary_host_status": "TLS/connect timeout from linux5080 on 2026-08-13",
                "download_mirror": "https://huggingface.co/ai-minamo/3d-photo-inpainting",
                "cross_check_mirror": "https://huggingface.co/camenduru/3d-photo-inpainting",
                "cross_check": "all three filenames, byte sizes, and LFS SHA256 values match across the two independent mirrors",
            },
            "inputs": {
                "image": str(image_path),
                "image_sha256": endpoint["image"]["sha256"],
                "mask": str(mask_path),
                "mask_sha256": endpoint["mask"]["sha256"],
                "depth": str(depth_path),
                "depth_sha256": depth_sha256,
                "hole_fraction": float(hole.mean()),
            },
            "adapter": {
                "crop_xyxy": [left, top, right, bottom],
                "working_resolution_hw": list(mask_work.shape),
                "max_side": args.max_side,
                "context_width_original_pixels": args.context_width,
                "context_width_working_pixels": effective_context_width,
                "context_fraction_in_crop": float(context.mean()),
                "known_edge_fraction_in_crop": float(known_edge.mean()),
                "mean_log_depth": mean_log_depth,
                **geometry,
            },
            "runtime": {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "model_load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            },
            "checks": {
                "known_pixels_exact": True,
                "hole_depth_finite_positive_fraction": float((np.isfinite(completed_depth[hole]) & (completed_depth[hole] > 0)).mean()),
            },
            "outputs": {
                name: {"path": str(output_dir / name), "sha256": file_hash(output_dir / name)}
                for name in (
                    "context_background_side.png",
                    "structure_known.png",
                    "structure_completed.png",
                    "depth_completed_float32.npy",
                    "depth_completed_preview.png",
                    "inpaint_raw.png",
                    "inpaint_composited.png",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
