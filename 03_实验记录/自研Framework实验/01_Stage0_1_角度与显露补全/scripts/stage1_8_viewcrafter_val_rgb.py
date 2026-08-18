#!/usr/bin/env python3
"""Evaluate frozen ViewCrafter RGB on the 14 qualified Stage 1.8 val targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

import stage1_8_viewcrafter_val_smoke as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--adaptive3dgs-src", type=Path, required=True)
    parser.add_argument("--viewcrafter-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    return parser.parse_args()


def resolve_inputs(config: dict[str, object]) -> dict[str, Path]:
    result = {}
    items = config["inputs"]
    for key in ("triplet_config", "qualification_pool", "base_depth_record", "passed_smoke_record"):
        path = Path(items[key])
        if not path.exists() or common.sha256(path) != items[f"{key}_sha256"]:
            raise RuntimeError(f"missing or changed frozen input: {path}")
        result[key] = path
    return result


def region_metrics(prediction: torch.Tensor, baseline: torch.Tensor, truth: torch.Tensor,
                   mask: torch.Tensor) -> dict[str, float | int]:
    candidate_error = torch.abs(prediction.float().cpu() - truth).mean(dim=1, keepdim=True) * 255
    baseline_error = torch.abs(baseline.float().cpu() - truth).mean(dim=1, keepdim=True) * 255
    selected = mask.bool()
    pixels = int(selected.sum())
    if pixels == 0:
        raise RuntimeError("qualified target region became empty after frozen crop")
    candidate_mae = float(candidate_error[selected].mean())
    baseline_mae = float(baseline_error[selected].mean())
    return {
        "pixels": pixels,
        "viewcrafter_rgb_mae_0_255": candidate_mae,
        "point_render_rgb_mae_0_255": baseline_mae,
        "relative_improvement_over_point_render": 1.0 - candidate_mae / max(baseline_mae, 1e-12),
        "viewcrafter_error_sum": float(candidate_error[selected].sum()),
        "point_render_error_sum": float(baseline_error[selected].sum()),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite val output or record")
    for path in (args.config, args.dataset_root, args.camera_parameters, args.adaptive3dgs_src,
                 args.viewcrafter_repo, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = resolve_inputs(config)
    if args.checkpoint.stat().st_size != int(config["model"]["checkpoint_bytes"]):
        raise RuntimeError("checkpoint byte count mismatch")
    if common.sha256(args.checkpoint) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.adaptive3dgs_src.resolve()))
    sys.path.insert(0, str(args.viewcrafter_repo.resolve()))
    from adaptive3dgs.datasets import load_hypersim_training_triplets
    from omegaconf import OmegaConf
    from pytorch3d.renderer import AlphaCompositor, PointsRasterizationSettings, PointsRasterizer, PointsRenderer
    from pytorch3d.structures import Pointclouds
    from torch.utils import checkpoint as _torch_checkpoint  # noqa: F401
    from utils.diffusion_utils import image_guided_synthesis, instantiate_from_config, load_model_checkpoint

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
    triplet_config = json.loads(frozen["triplet_config"].read_text(encoding="utf-8"))
    selected_scenes = {scene for scene, _ in selected}
    val_config = dict(triplet_config)
    val_config["triplets"] = [item for item in triplet_config["triplets"] if item["scene"] in selected_scenes]
    with tempfile.TemporaryDirectory(prefix="stage1_8_viewcrafter_val_") as temporary:
        val_path = Path(temporary) / "qualified_val_triplets.json"
        val_path.write_text(json.dumps(val_config), encoding="utf-8")
        triplets = load_hypersim_training_triplets(val_path, args.dataset_root, args.camera_parameters)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    common.install_sdpa_spatial_attention()
    model_cfg = OmegaConf.load(args.viewcrafter_repo / config["model"]["config"])["model"]
    model_cfg["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model_cfg["params"]["cond_stage_config"]["params"]["version"] = None
    model_cfg["params"]["img_cond_stage_config"]["params"]["version"] = None
    started = time.monotonic()
    model = instantiate_from_config(model_cfg).to(device)
    model.cond_stage_model.device = str(device)
    model.perframe_ae = bool(config["model"]["perframe_ae"])
    model = load_model_checkpoint(model, str(args.checkpoint)).eval()
    height, width = map(int, config["adapter"]["input_size_hw"])
    frames = int(config["adapter"]["video_frames"])
    noise_shape = [1, model.model.diffusion_model.out_channels, frames, height // 8, width // 8]
    results = []

    for triplet in triplets:
        if triplet.split != "val":
            continue
        base_item = base_by_key.get((triplet.scene, int(triplet.source_frame_index)))
        if base_item is None:
            raise RuntimeError(f"missing BaseDepth: {triplet.scene}")
        base_path = Path(base_item["arrays"])
        if common.sha256(base_path) != base_item["arrays_sha256"]:
            raise RuntimeError(f"BaseDepth hash mismatch: {base_path}")
        with np.load(base_path, allow_pickle=False) as archive:
            depth_full = archive["base_depth_z_float32"].astype(np.float32)
        source_rgb_full = triplet.source_rgb_uint8.astype(np.float32) / 255.0
        original_height, original_width = depth_full.shape
        crop = common.crop_for_aspect(original_height, original_width, height, width)
        crop_x, crop_y, crop_width, crop_height = crop
        slices = (slice(crop_y, crop_y + crop_height), slice(crop_x, crop_x + crop_width))
        source = common.resize_chw(source_rgb_full[slices], (height, width), "bilinear")
        depth_tensor = common.resize_chw(depth_full[slices], (height, width), "bilinear")
        depth = depth_tensor[0, 0].numpy()
        source_k = common.adjusted_intrinsics(triplet.source_camera.intrinsics_3x3_float64, crop, (height, width))
        points = common.unproject(depth, source_k)
        valid_points = np.isfinite(points).all(axis=2) & np.isfinite(depth) & (depth > 0)

        for side_name, observation in zip(("left", "right"), triplet.target_observations):
            key = (triplet.scene, side_name)
            if key not in selected:
                continue
            qualification = selected[key]
            target_frame = int(observation.observation_id.rsplit("-", 1)[1])
            if target_frame != int(qualification["target_frame"]):
                raise RuntimeError(f"target mismatch: {key}")
            visibility_path = Path(qualification["arrays"])
            if common.sha256(visibility_path) != qualification["arrays_sha256"]:
                raise RuntimeError(f"visibility hash mismatch: {visibility_path}")
            target = common.resize_chw(
                observation.rgb_uint8.astype(np.float32)[slices] / 255.0, (height, width), "bilinear"
            )
            masks = {}
            with np.load(visibility_path, allow_pickle=False) as archive:
                for region in config["rgb_gates"]["regions"]:
                    masks[region] = common.resize_chw(
                        archive[f"{region}_uint8"][slices].astype(np.float32), (height, width), "nearest"
                    ) > 0.5
            target_k = common.adjusted_intrinsics(observation.intrinsics_3x3_float64, crop, (height, width))
            c2w, relative_w2c = common.interpolate_c2w(
                triplet.source_camera.world_to_camera_4x4_float64,
                observation.world_to_camera_4x4_float64, frames,
            )
            cameras_cpu = common.make_pytorch3d_cameras(
                c2w, source_k, target_k, (height, width), torch.device("cpu")
            )
            geometry = common.geometry_gate(points, relative_w2c, target_k, cameras_cpu, (height, width))
            geometry_gates = config["geometry_gates"]
            minimum_all = geometry_gates.get(
                "minimum_all_positive_depth_points", geometry_gates.get("minimum_compared_points", 0)
            )
            minimum_raster = geometry_gates.get("minimum_target_raster_points", 0)
            p99_maximum = geometry_gates.get(
                "all_positive_depth_p99_px_maximum", geometry_gates.get("p99_px_maximum")
            )
            raster_maximum = geometry_gates.get(
                "target_raster_maximum_px_maximum", geometry_gates.get("maximum_px_maximum")
            )
            if (geometry["compared_points"] < minimum_all or
                    geometry["raster_compared_points"] < minimum_raster or
                    geometry["p99_px"] > p99_maximum or
                    geometry["maximum_px"] > raster_maximum):
                raise RuntimeError(f"geometry gate failed: {key}: {geometry}")

            sample_index = len(results)
            seed = int(config["model"]["seed_base"]) + sample_index
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            sample_started = time.monotonic()
            cameras = common.make_pytorch3d_cameras(c2w, source_k, target_k, (height, width), device)
            settings = PointsRasterizationSettings(
                image_size=(height, width), radius=float(config["adapter"]["point_radius_ndc"]),
                points_per_pixel=int(config["adapter"]["points_per_pixel"]), bin_size=0,
            )
            renderer = PointsRenderer(PointsRasterizer(cameras=cameras, raster_settings=settings), AlphaCompositor())
            point_tensor = torch.from_numpy(points[valid_points]).float().to(device)
            color_tensor = source[0].permute(1, 2, 0)[torch.from_numpy(valid_points)].float().to(device)
            cloud = Pointclouds(points=[point_tensor], features=[color_tensor]).extend(frames)
            with torch.inference_mode():
                renderings = renderer(cloud)[..., :3]
                renderings[0] = source[0].permute(1, 2, 0).to(device)
            if not bool(torch.isfinite(renderings).all()):
                raise RuntimeError(f"non-finite render: {key}")
            videos = (renderings * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                samples = image_guided_synthesis(
                    model, [config["model"]["prompt"]], videos, noise_shape,
                    n_samples=1, ddim_steps=int(config["model"]["ddim_steps"]),
                    ddim_eta=float(config["model"]["ddim_eta"]),
                    unconditional_guidance_scale=float(config["model"]["unconditional_guidance_scale"]),
                    cfg_img=None, fs=int(config["model"]["frame_stride"]),
                    text_input=bool(config["model"]["text_input"]), multiple_cond_cfg=False,
                    timestep_spacing=config["model"]["timestep_spacing"],
                    guidance_rescale=float(config["model"]["guidance_rescale"]), condition_index=[0],
                )
            generated_raw = samples[0, 0]
            if not bool(torch.isfinite(generated_raw).all()):
                raise RuntimeError(f"non-finite generation: {key}")
            generated = torch.clamp(generated_raw, -1, 1)
            prediction = generated[:, -1].add(1).div(2)[None]
            baseline = renderings[-1].permute(2, 0, 1)[None]
            regions = {name: region_metrics(prediction, baseline, target, mask) for name, mask in masks.items()}
            sample_dir = args.output_dir / f"{triplet.scene}-{side_name}-{target_frame:04d}"
            sample_dir.mkdir()
            generated_path = sample_dir / "generated_target.png"
            comparison_path = sample_dir / "comparison_source_render_generated_truth.png"
            common.save_rgb(generated_path, prediction)
            common.save_rgb(comparison_path, torch.cat((source, baseline.float().cpu(), prediction.float().cpu(), target), dim=3))
            torch.cuda.synchronize(device)
            results.append({
                "scene": triplet.scene, "camera": triplet.camera_name,
                "source_frame": int(triplet.source_frame_index), "side": side_name,
                "target_frame": target_frame, "yaw_deg": float(qualification["yaw_deg"]), "seed": seed,
                "geometry_gate": geometry, "regions": regions,
                "point_render_exact_black_fraction": float((baseline == 0).all(dim=1).float().mean()),
                "raw_generated_minimum": float(generated_raw.min()), "raw_generated_maximum": float(generated_raw.max()),
                "runtime_seconds": time.monotonic() - sample_started,
                "inputs": {
                    "base_depth": common.portable(base_path), "base_depth_sha256": common.sha256(base_path),
                    "visibility": common.portable(visibility_path), "visibility_sha256": common.sha256(visibility_path),
                },
                "artifacts": {
                    "generated": common.portable(generated_path), "generated_sha256": common.sha256(generated_path),
                    "comparison": common.portable(comparison_path), "comparison_sha256": common.sha256(comparison_path),
                },
            })
            del cameras, settings, renderer, point_tensor, color_tensor, cloud, renderings, videos
            del samples, generated_raw, generated, prediction, baseline
            torch.cuda.empty_cache()

    if len(results) != expected:
        raise RuntimeError(f"evaluated {len(results)} targets, expected {expected}")
    aggregate = {}
    gate_pass = True
    for region in config["rgb_gates"]["regions"]:
        pixels = sum(int(item["regions"][region]["pixels"]) for item in results)
        candidate_sum = sum(float(item["regions"][region]["viewcrafter_error_sum"]) for item in results)
        baseline_sum = sum(float(item["regions"][region]["point_render_error_sum"]) for item in results)
        mae = candidate_sum / pixels
        baseline_mae = baseline_sum / pixels
        improvement = 1.0 - candidate_sum / baseline_sum
        mae_pass = mae <= float(config["rgb_gates"]["mae_0_255_maximum"])
        improvement_pass = improvement >= float(config["rgb_gates"]["relative_improvement_over_warp_minimum"])
        gate_pass &= mae_pass and improvement_pass
        aggregate[region] = {
            "pixels": pixels, "viewcrafter_rgb_mae_0_255": mae,
            "point_render_rgb_mae_0_255": baseline_mae,
            "relative_improvement_over_point_render": improvement,
            "mae_gate_pass": mae_pass, "improvement_gate_pass": improvement_pass,
            "targets_passing_absolute_mae": sum(
                item["regions"][region]["viewcrafter_rgb_mae_0_255"] <= config["rgb_gates"]["mae_0_255_maximum"]
                for item in results
            ),
        }
    record = {
        "schema_version": config["schema_version"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass_rgb_gate" if gate_pass else "failed_rgb_gate",
        "producer_machine_id": args.machine_id, "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": [0],
        "inputs": {key: {"path": common.portable(path), "sha256": common.sha256(path)} for key, path in frozen.items()} | {
            "checkpoint": {"path": common.portable(args.checkpoint), "bytes": args.checkpoint.stat().st_size,
                           "sha256": common.sha256(args.checkpoint)},
            "target_rgb_model_input": False, "target_depth_model_input": False,
            "held_out_test_read": False, "p01_read": False,
        },
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
