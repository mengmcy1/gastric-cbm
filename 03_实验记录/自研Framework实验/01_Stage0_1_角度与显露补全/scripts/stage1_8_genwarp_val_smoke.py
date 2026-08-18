#!/usr/bin/env python3
"""Run one frozen GenWarp multi2 validation endpoint smoke without test/P01 access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--adaptive3dgs-src", type=Path, required=True)
    parser.add_argument("--genwarp-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def resize_chw(value: np.ndarray, size: tuple[int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(value)
    if value.ndim == 2:
        tensor = tensor[None, None].float()
    else:
        tensor = tensor.permute(2, 0, 1)[None].float()
    kwargs: dict[str, object] = {"size": size, "mode": mode}
    if mode == "bilinear":
        kwargs.update({"align_corners": False, "antialias": True})
    return F.interpolate(tensor, **kwargs)


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().clamp(0, 1)[0, 0].cpu().numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="L").save(path)


def projection_from_intrinsics(k: np.ndarray, width: int, height: int, near: float, far: float) -> np.ndarray:
    """OpenGL projection matching GenWarp's positive-screen-y viewport convention."""
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    result = np.zeros((4, 4), dtype=np.float64)
    result[0, 0] = 2.0 * fx / width
    result[0, 2] = 1.0 - 2.0 * cx / width
    result[1, 1] = -2.0 * fy / height
    result[1, 2] = 1.0 - 2.0 * cy / height
    result[2, 2] = -(far + near) / (far - near)
    result[2, 3] = -2.0 * far * near / (far - near)
    result[3, 2] = -1.0
    return result


def viewport(width: int, height: int) -> np.ndarray:
    return np.array(
        [[width / 2, 0, 0, width / 2], [0, height / 2, 0, height / 2],
         [0, 0, 0.5, 0.5], [0, 0, 0, 1]], dtype=np.float64
    )


def geometry_gate(
    depth: np.ndarray,
    source_k: np.ndarray,
    target_k: np.ndarray,
    source_w2c_cv: np.ndarray,
    target_w2c_cv: np.ndarray,
    source_projection: np.ndarray,
    target_projection: np.ndarray,
) -> dict[str, float | int]:
    """Compare exact GenWarp matrix projection with canonical OpenCV projection."""
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(0, height, 8), np.arange(0, width, 8), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3).astype(np.float64)
    sampled_depth = depth[yy, xx].reshape(-1).astype(np.float64)
    source_xyz_cv = (pixels @ np.linalg.inv(source_k).T) * sampled_depth[:, None]
    rel_cv = target_w2c_cv @ np.linalg.inv(source_w2c_cv)
    target_xyz_cv = source_xyz_cv @ rel_cv[:3, :3].T + rel_cv[:3, 3]
    canonical_h = target_xyz_cv @ target_k.T
    canonical_xy = canonical_h[:, :2] / canonical_h[:, 2:3]

    screen = np.concatenate((pixels[:, :2], np.zeros((len(pixels), 1)), np.ones((len(pixels), 1))), axis=1)
    eye_gl = screen @ np.linalg.inv(viewport(width, height) @ source_projection).T
    eye_gl *= sampled_depth[:, None]
    eye_gl[:, 3] = 1.0
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    source_w2c_gl = cv_to_gl @ source_w2c_cv
    target_w2c_gl = cv_to_gl @ target_w2c_cv
    rel_gl = target_w2c_gl @ np.linalg.inv(source_w2c_gl)
    clip = eye_gl @ (target_projection @ rel_gl).T
    ndc = clip / clip[:, 3:4]
    screen_target = ndc @ viewport(width, height).T
    genwarp_xy = screen_target[:, :2]
    valid = np.isfinite(canonical_xy).all(axis=1) & np.isfinite(genwarp_xy).all(axis=1)
    valid &= np.isfinite(sampled_depth) & (sampled_depth > 0) & (target_xyz_cv[:, 2] > 0)
    error = np.linalg.norm(genwarp_xy[valid] - canonical_xy[valid], axis=1)
    if not len(error):
        raise RuntimeError("geometry gate found no valid comparison points")
    return {
        "compared_points": int(len(error)),
        "median_px": float(np.median(error)),
        "p99_px": float(np.quantile(error, 0.99)),
        "maximum_px": float(np.max(error)),
    }


def main() -> None:
    args = parse_args()
    for path in (args.config, args.dataset_root, args.camera_parameters, args.adaptive3dgs_src, args.genwarp_repo):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite smoke output or record")
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    sys.path.insert(0, str(args.adaptive3dgs_src.resolve()))
    sys.path.insert(0, str(args.genwarp_repo.resolve()))
    from adaptive3dgs.datasets import load_hypersim_training_triplets
    from genwarp import GenWarp

    triplets = load_hypersim_training_triplets(args.config, args.dataset_root, args.camera_parameters)
    if len(triplets) != 1:
        raise RuntimeError("smoke config must load exactly one triplet")
    triplet = triplets[0]
    side = config["selected_target"]["side"]
    side_index = {"left": 0, "right": 1}[side]
    target = triplet.target_observations[side_index]
    expected_frame = int(config["selected_target"]["target_frame"])
    if int(target.observation_id.rsplit("-", 1)[1]) != expected_frame:
        raise RuntimeError("selected target frame mismatch")

    base_path = Path(config["frozen_base_depth"]["arrays"])
    visibility_path = Path(config["selected_target"]["visibility_arrays"])
    for path, expected in ((base_path, config["frozen_base_depth"]["arrays_sha256"]),
                           (visibility_path, config["selected_target"]["visibility_arrays_sha256"])):
        if sha256(path) != expected:
            raise RuntimeError(f"input hash mismatch: {path}")
    with np.load(base_path, allow_pickle=False) as values:
        base_depth = values[config["frozen_base_depth"]["array_key"]].astype(np.float32)
    source_rgb = triplet.source_rgb_uint8.astype(np.float32) / 255.0
    target_rgb = target.rgb_uint8.astype(np.float32) / 255.0
    height, width = base_depth.shape
    side_length = min(height, width)
    crop_y = (height - side_length) // 2
    crop_x = (width - side_length) // 2
    crop = (slice(crop_y, crop_y + side_length), slice(crop_x, crop_x + side_length))
    output_height, output_width = map(int, config["adapter"]["input_size_hw"])
    output_size = (output_height, output_width)
    source_tensor = resize_chw(source_rgb[crop], output_size, "bilinear")
    target_tensor = resize_chw(target_rgb[crop], output_size, "bilinear")
    depth_tensor = resize_chw(base_depth[crop], output_size, "bilinear")
    resized_depth = depth_tensor[0, 0].numpy()

    scale_x, scale_y = output_width / side_length, output_height / side_length
    source_k = triplet.source_camera.intrinsics_3x3_float64.copy()
    target_k = target.intrinsics_3x3_float64.copy()
    for k in (source_k, target_k):
        k[0, 2] -= crop_x
        k[1, 2] -= crop_y
        k[0] *= scale_x
        k[1] *= scale_y
    near, far = float(config["adapter"]["near_m"]), float(config["adapter"]["far_m"])
    source_projection = projection_from_intrinsics(source_k, output_width, output_height, near, far)
    target_projection = projection_from_intrinsics(target_k, output_width, output_height, near, far)
    geometry = geometry_gate(
        resized_depth, source_k, target_k,
        triplet.source_camera.world_to_camera_4x4_float64,
        target.world_to_camera_4x4_float64,
        source_projection, target_projection,
    )
    gates = config["gates"]
    if geometry["compared_points"] < gates["minimum_compared_geometry_points"]:
        raise RuntimeError(f"too few geometry points: {geometry}")
    if geometry["p99_px"] > gates["geometry_projection_p99_px_max"] or geometry["maximum_px"] > gates["geometry_projection_max_px_max"]:
        raise RuntimeError(f"camera adapter geometry gate failed: {geometry}")

    torch.manual_seed(int(config["model"]["seed"]))
    torch.cuda.manual_seed_all(int(config["model"]["seed"]))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model_config = {
        "pretrained_model_path": str((args.genwarp_repo / "checkpoints").resolve()),
        "checkpoint_name": config["model"]["checkpoint_name"],
        "half_precision_weights": bool(config["model"]["half_precision_weights"]),
        "height": output_height,
        "width": output_width,
        "num_inference_steps": int(config["model"]["num_inference_steps"]),
        "guidance_scale": float(config["model"]["guidance_scale"]),
    }
    started = time.monotonic()
    model = GenWarp(cfg=model_config, device="cuda:0")
    dtype = torch.float16 if model_config["half_precision_weights"] else torch.float32
    cv_to_gl = np.asarray(config["adapter"]["opencv_to_opengl_left_multiply"], dtype=np.float64)
    source_w2c_gl = cv_to_gl @ triplet.source_camera.world_to_camera_4x4_float64
    target_w2c_gl = cv_to_gl @ target.world_to_camera_4x4_float64
    rel_gl = target_w2c_gl @ np.linalg.inv(source_w2c_gl)
    with torch.inference_mode():
        output = model(
            source_tensor.to(device=device, dtype=dtype),
            depth_tensor.to(device=device, dtype=dtype),
            torch.from_numpy(rel_gl)[None].to(device=device, dtype=dtype),
            torch.from_numpy(source_projection)[None].to(device=device, dtype=dtype),
            torch.from_numpy(target_projection)[None].to(device=device, dtype=dtype),
        )
    torch.cuda.synchronize(device)
    runtime_seconds = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    tensor_stats = {}
    for name, value in output.items():
        finite_map = torch.isfinite(value)
        finite = bool(finite_map.all())
        finite_values = value[finite_map].float()
        tensor_stats[name] = {
            "shape": list(value.shape), "dtype": str(value.dtype), "all_finite": finite,
            "nonfinite_values": int((~finite_map).sum()),
            "finite_minimum": float(finite_values.min()), "finite_maximum": float(finite_values.max()),
            "finite_mean": float(finite_values.mean()), "finite_std": float(finite_values.std()),
        }
        if name in {"synthesized", "warped", "mask"} and not finite:
            raise RuntimeError(f"non-finite output: {name}")
    correspondence_visible = output["mask"].bool().expand_as(output["correspondence"])
    correspondence_visible_finite = bool(torch.isfinite(output["correspondence"])[correspondence_visible].all())
    tensor_stats["correspondence"]["visible_region_all_finite"] = correspondence_visible_finite
    tensor_stats["correspondence"]["visible_region_values"] = int(correspondence_visible.sum())
    if not correspondence_visible_finite:
        raise RuntimeError("non-finite correspondence inside visible warp support")
    synth = output["synthesized"]
    low, high = map(float, gates["synthesized_range"])
    if float(synth.min()) < low - 1e-5 or float(synth.max()) > high + 1e-5:
        raise RuntimeError("synthesized output outside frozen range")
    if float(synth.float().std()) < float(gates["minimum_synthesized_std"]):
        raise RuntimeError("synthesized output is effectively constant")
    peak_reserved_gib = peak_reserved / 1024 ** 3
    if peak_reserved_gib > float(gates["peak_reserved_gib_max"]):
        raise RuntimeError(f"peak reserved memory gate failed: {peak_reserved_gib:.3f} GiB")

    save_rgb(args.output_dir / "source.png", source_tensor)
    save_rgb(args.output_dir / "target_truth_val_only.png", target_tensor)
    save_rgb(args.output_dir / "warped.png", output["warped"])
    save_rgb(args.output_dir / "synthesized.png", synth)
    save_mask(args.output_dir / "warp_valid.png", output["mask"])
    comparison = torch.cat((source_tensor, output["warped"].float().cpu(), synth.float().cpu(), target_tensor), dim=3)
    save_rgb(args.output_dir / "comparison_source_warp_synth_truth.png", comparison)

    with np.load(visibility_path, allow_pickle=False) as values:
        visibility_keys = sorted(values.files)
    record = {
        "schema_version": config["schema_version"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass",
        "producer_machine_id": args.machine_id,
        "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": [0],
        "inputs": {
            "config": portable(args.config), "config_sha256": sha256(args.config),
            "dataset_root": portable(args.dataset_root),
            "camera_parameters": portable(args.camera_parameters), "camera_parameters_sha256": sha256(args.camera_parameters),
            "base_depth": portable(base_path), "base_depth_sha256": sha256(base_path),
            "visibility": portable(visibility_path), "visibility_sha256": sha256(visibility_path),
            "visibility_keys_read_after_inference_for_audit_only": visibility_keys,
            "held_out_test_read": False, "p01_read": False,
        },
        "target": {
            "split": triplet.split, "scene": triplet.scene, "camera": triplet.camera_name,
            "source_frame": triplet.source_frame_index, "side": side, "target_frame": expected_frame,
            "yaw_deg": config["triplets"][0][f"{side}_yaw_deg"],
        },
        "adapter": {
            "original_shape_hw": [height, width], "crop_xywh": [crop_x, crop_y, side_length, side_length],
            "output_shape_hw": [output_height, output_width],
            "source_intrinsics_after_crop_resize": source_k.tolist(),
            "target_intrinsics_after_crop_resize": target_k.tolist(),
            "coordinate_conversion": "OpenCV right/down/forward to OpenGL right/up/back via diag(1,-1,-1,1)",
            "geometry_gate": geometry,
        },
        "model": {
            "source_commit": config["model"]["source_commit"], "checkpoint_name": model_config["checkpoint_name"],
            "half_precision_weights": model_config["half_precision_weights"],
            "num_inference_steps": model_config["num_inference_steps"], "guidance_scale": model_config["guidance_scale"],
            "seed": config["model"]["seed"],
        },
        "runtime": {
            "seconds_including_model_load": runtime_seconds,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": peak_allocated / 1024 ** 3, "peak_reserved_gib": peak_reserved_gib,
        },
        "outputs": tensor_stats,
        "artifacts": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "source": args.output_dir / "source.png", "target_truth_val_only": args.output_dir / "target_truth_val_only.png",
                "warped": args.output_dir / "warped.png", "synthesized": args.output_dir / "synthesized.png",
                "warp_valid": args.output_dir / "warp_valid.png",
                "comparison": args.output_dir / "comparison_source_warp_synth_truth.png",
            }.items()
        },
        "gates": {"frozen": gates, "result": "pass"},
        "quality_boundary": config["quality_boundary"],
    }
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "geometry": geometry, "runtime": record["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
