#!/usr/bin/env python3
"""Evaluate frozen GenWarp RGB on the 14 qualified Stage 1.8 val targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

import stage1_8_genwarp_val_smoke as common


def args_parser() -> argparse.Namespace:
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


def resolve_and_verify(item: dict[str, str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key in ("triplet_config", "qualification_pool", "base_depth_record", "passed_smoke_record"):
        path = Path(item[key])
        if not path.exists() or common.sha256(path) != item[f"{key}_sha256"]:
            raise RuntimeError(f"missing or changed frozen input: {path}")
        result[key] = path
    return result


def adjusted_k(old: np.ndarray, crop_x: int, crop_y: int, side: int, output: int) -> np.ndarray:
    k = old.copy()
    k[0, 2] -= crop_x
    k[1, 2] -= crop_y
    k[0] *= output / side
    k[1] *= output / side
    return k


def region_metrics(prediction: torch.Tensor, warp: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    error = torch.abs(prediction.float().cpu() - truth).mean(dim=1, keepdim=True) * 255
    warp_error = torch.abs(warp.float().cpu() - truth).mean(dim=1, keepdim=True) * 255
    selected = mask.bool()
    count = int(selected.sum())
    if count == 0:
        raise RuntimeError("qualified target region became empty after frozen crop")
    candidate_mae = float(error[selected].mean())
    baseline_mae = float(warp_error[selected].mean())
    return {
        "pixels": count,
        "genwarp_rgb_mae_0_255": candidate_mae,
        "warp_rgb_mae_0_255": baseline_mae,
        "relative_improvement_over_warp": 1.0 - candidate_mae / max(baseline_mae, 1e-12),
        "genwarp_error_sum": float(error[selected].sum()),
        "warp_error_sum": float(warp_error[selected].sum()),
    }


def main() -> int:
    args = args_parser()
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite val output or record")
    for path in (args.config, args.dataset_root, args.camera_parameters, args.adaptive3dgs_src, args.genwarp_repo):
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = resolve_and_verify(config["inputs"])
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.adaptive3dgs_src.resolve()))
    sys.path.insert(0, str(args.genwarp_repo.resolve()))
    from adaptive3dgs.datasets import load_hypersim_training_triplets
    from genwarp import GenWarp

    pool = json.loads(frozen["qualification_pool"].read_text(encoding="utf-8"))
    qualified = [item for item in pool["eligible_targets"] if item["split"] == "val"]
    expected = int(config["expected_validation_targets"])
    if len(qualified) != expected:
        raise RuntimeError(f"expected {expected} qualified val targets, got {len(qualified)}")
    selected = {(item["scene"], item["side"]): item for item in qualified}
    if len(selected) != expected:
        raise RuntimeError("qualified val target keys are not unique")
    base_record = json.loads(frozen["base_depth_record"].read_text(encoding="utf-8"))
    base_by_key = {(item["scene"], int(item["source_frame"])): item for item in base_record["results"]}
    full_triplet_config = json.loads(frozen["triplet_config"].read_text(encoding="utf-8"))
    selected_scenes = {scene for scene, _ in selected}
    val_only_config = dict(full_triplet_config)
    val_only_config["triplets"] = [
        item for item in full_triplet_config["triplets"] if item["scene"] in selected_scenes
    ]
    with tempfile.TemporaryDirectory(prefix="stage1_8_genwarp_val_") as temporary:
        val_only_path = Path(temporary) / "qualified_val_triplets.json"
        val_only_path.write_text(json.dumps(val_only_config), encoding="utf-8")
        triplets = load_hypersim_training_triplets(val_only_path, args.dataset_root, args.camera_parameters)

    size = int(config["adapter"]["input_size_hw"][0])
    if config["adapter"]["input_size_hw"] != [size, size]:
        raise RuntimeError("GenWarp frozen adapter requires square input")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model_cfg = {
        "pretrained_model_path": str((args.genwarp_repo / "checkpoints").resolve()),
        "checkpoint_name": config["model"]["checkpoint_name"],
        "half_precision_weights": True,
        "height": size, "width": size,
        "num_inference_steps": int(config["model"]["num_inference_steps"]),
        "guidance_scale": float(config["model"]["guidance_scale"]),
    }
    started = time.monotonic()
    model = GenWarp(cfg=model_cfg, device="cuda:0")
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    results: list[dict[str, object]] = []
    artifact_files: list[Path] = []

    for triplet in triplets:
        if triplet.split != "val":
            continue
        base_item = base_by_key.get((triplet.scene, int(triplet.source_frame_index)))
        if base_item is None:
            raise RuntimeError(f"missing BaseDepth for {triplet.scene}")
        base_path = Path(base_item["arrays"])
        if common.sha256(base_path) != base_item["arrays_sha256"]:
            raise RuntimeError(f"BaseDepth hash mismatch: {base_path}")
        with np.load(base_path, allow_pickle=False) as archive:
            depth = archive["base_depth_z_float32"].astype(np.float32)
        source_rgb = triplet.source_rgb_uint8.astype(np.float32) / 255.0
        height, width = depth.shape
        side_length = min(height, width)
        crop_y, crop_x = (height - side_length) // 2, (width - side_length) // 2
        crop = (slice(crop_y, crop_y + side_length), slice(crop_x, crop_x + side_length))
        source = common.resize_chw(source_rgb[crop], (size, size), "bilinear")
        depth_tensor = common.resize_chw(depth[crop], (size, size), "bilinear")
        source_k = adjusted_k(triplet.source_camera.intrinsics_3x3_float64, crop_x, crop_y, side_length, size)
        source_projection = common.projection_from_intrinsics(source_k, size, size, config["adapter"]["near_m"], config["adapter"]["far_m"])

        for side_name, observation in zip(("left", "right"), triplet.target_observations):
            key = (triplet.scene, side_name)
            if key not in selected:
                continue
            qualification = selected[key]
            target_frame = int(observation.observation_id.rsplit("-", 1)[1])
            if target_frame != int(qualification["target_frame"]):
                raise RuntimeError(f"target frame mismatch: {key}")
            visibility_path = Path(qualification["arrays"])
            if common.sha256(visibility_path) != qualification["arrays_sha256"]:
                raise RuntimeError(f"visibility hash mismatch: {visibility_path}")
            target = common.resize_chw(observation.rgb_uint8.astype(np.float32)[crop] / 255.0, (size, size), "bilinear")
            masks: dict[str, torch.Tensor] = {}
            with np.load(visibility_path, allow_pickle=False) as archive:
                for region in config["rgb_gates"]["regions"]:
                    masks[region] = common.resize_chw(archive[f"{region}_uint8"][crop].astype(np.float32), (size, size), "nearest") > 0.5
            target_k = adjusted_k(observation.intrinsics_3x3_float64, crop_x, crop_y, side_length, size)
            target_projection = common.projection_from_intrinsics(target_k, size, size, config["adapter"]["near_m"], config["adapter"]["far_m"])
            geometry = common.geometry_gate(
                depth_tensor[0, 0].numpy(), source_k, target_k,
                triplet.source_camera.world_to_camera_4x4_float64, observation.world_to_camera_4x4_float64,
                source_projection, target_projection,
            )
            if geometry["p99_px"] > 0.05 or geometry["maximum_px"] > 0.1:
                raise RuntimeError(f"geometry gate failed: {key}: {geometry}")
            source_gl = cv_to_gl @ triplet.source_camera.world_to_camera_4x4_float64
            target_gl = cv_to_gl @ observation.world_to_camera_4x4_float64
            rel_gl = target_gl @ np.linalg.inv(source_gl)
            sample_index = len(results)
            seed = int(config["model"]["seed_base"]) + sample_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sample_started = time.monotonic()
            with torch.inference_mode():
                output = model(
                    source.to(device=device, dtype=torch.float16),
                    depth_tensor.to(device=device, dtype=torch.float16),
                    torch.from_numpy(rel_gl)[None].to(device=device, dtype=torch.float16),
                    torch.from_numpy(source_projection)[None].to(device=device, dtype=torch.float16),
                    torch.from_numpy(target_projection)[None].to(device=device, dtype=torch.float16),
                )
            torch.cuda.synchronize(device)
            for name, value in output.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"non-finite {name}: {key}")
            regions = {
                name: region_metrics(output["synthesized"], output["warped"], target, mask)
                for name, mask in masks.items()
            }
            sample_dir = args.output_dir / f"{triplet.scene}-{side_name}-{target_frame:04d}"
            sample_dir.mkdir()
            synth_path = sample_dir / "synthesized.png"
            comparison_path = sample_dir / "comparison_source_warp_synth_truth.png"
            common.save_rgb(synth_path, output["synthesized"])
            common.save_rgb(comparison_path, torch.cat((source, output["warped"].float().cpu(), output["synthesized"].float().cpu(), target), dim=3))
            artifact_files.extend((synth_path, comparison_path))
            results.append({
                "scene": triplet.scene, "camera": triplet.camera_name,
                "source_frame": int(triplet.source_frame_index), "side": side_name,
                "target_frame": target_frame, "yaw_deg": float(qualification["yaw_deg"]), "seed": seed,
                "geometry_gate": geometry, "regions": regions,
                "warp_coverage": float(output["mask"].float().mean()),
                "runtime_seconds": time.monotonic() - sample_started,
                "inputs": {
                    "base_depth": common.portable(base_path), "base_depth_sha256": common.sha256(base_path),
                    "visibility": common.portable(visibility_path), "visibility_sha256": common.sha256(visibility_path),
                },
                "artifacts": {
                    "synthesized": common.portable(synth_path), "synthesized_sha256": common.sha256(synth_path),
                    "comparison": common.portable(comparison_path), "comparison_sha256": common.sha256(comparison_path),
                },
            })

    if len(results) != expected:
        raise RuntimeError(f"evaluated {len(results)} targets, expected {expected}")
    aggregate: dict[str, dict[str, float | int | bool]] = {}
    gate_pass = True
    for region in config["rgb_gates"]["regions"]:
        pixel_count = sum(int(item["regions"][region]["pixels"]) for item in results)
        candidate_sum = sum(float(item["regions"][region]["genwarp_error_sum"]) for item in results)
        baseline_sum = sum(float(item["regions"][region]["warp_error_sum"]) for item in results)
        mae = candidate_sum / pixel_count
        baseline_mae = baseline_sum / pixel_count
        improvement = 1.0 - candidate_sum / baseline_sum
        passed_mae = mae <= float(config["rgb_gates"]["mae_0_255_maximum"])
        passed_improvement = improvement >= float(config["rgb_gates"]["relative_improvement_over_warp_minimum"])
        gate_pass &= passed_mae and passed_improvement
        aggregate[region] = {
            "pixels": pixel_count, "genwarp_rgb_mae_0_255": mae,
            "warp_rgb_mae_0_255": baseline_mae, "relative_improvement_over_warp": improvement,
            "mae_gate_pass": passed_mae, "improvement_gate_pass": passed_improvement,
        }
    record = {
        "schema_version": config["schema_version"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass_rgb_gate" if gate_pass else "failed_rgb_gate",
        "producer_machine_id": args.machine_id, "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": [0],
        "inputs": {
            key: {"path": common.portable(path), "sha256": common.sha256(path)} for key, path in frozen.items()
        } | {"held_out_test_read": False, "p01_read": False},
        "adapter": config["adapter"], "model": config["model"],
        "results": results, "aggregate": aggregate,
        "runtime": {
            "total_seconds_including_model_load": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3,
        },
        "gates": {"frozen": config["rgb_gates"], "result": "pass" if gate_pass else "failed"},
        "quality_boundary": config["quality_boundary"],
    }
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "aggregate": aggregate, "runtime": record["runtime"]}, indent=2))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
