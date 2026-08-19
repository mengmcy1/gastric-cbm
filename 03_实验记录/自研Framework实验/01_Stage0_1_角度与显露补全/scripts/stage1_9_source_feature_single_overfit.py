#!/usr/bin/env python3
"""Stage 1.9-2 single directed-pair overfit gate for source-feature conditioning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from adaptive3dgs.models import SourceFeatureTargetViewNet, compute_target_view_loss
from stage1_9_source_feature_common import portable, prepare_directed_samples, sha256, target_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--rgb-record", type=Path, action="append", required=True)
    parser.add_argument("--base-record", type=Path, action="append", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def save_rgb(path: Path, value: torch.Tensor) -> None:
    image = value[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray(np.uint8(np.rint(image * 255))).save(path)


def save_mask(path: Path, sample) -> None:
    shown = np.zeros((*sample.observed.shape[-2:], 3), dtype=np.uint8)
    shown[sample.observed[0, 0].cpu().numpy()] = [50, 200, 80]
    shown[sample.occluded[0, 0].cpu().numpy()] = [230, 60, 60]
    shown[sample.outside[0, 0].cpu().numpy()] = [60, 100, 230]
    Image.fromarray(shown).save(path)


def main() -> int:
    args = parse_args()
    for name in ("training_config", "pair_config", "dataset_root", "camera_parameters", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    args.rgb_record = [path.resolve() for path in args.rgb_record]
    args.base_record = [path.resolve() for path in args.base_record]
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    config = json.loads(args.training_config.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_run":
        raise RuntimeError("training config must be frozen_before_run")
    sample_cfg = config["sample"]
    primary_key = (str(sample_cfg["scene"]), int(sample_cfg["source_frame"]), int(sample_cfg["target_frame"]))
    control_key = (
        str(sample_cfg["scene"]), int(sample_cfg["source_frame"]), int(sample_cfg["control_target_frame"])
    )
    device = torch.device("cpu")
    if not args.prepare_only:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with physical GPU index")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda:0")
    samples = prepare_directed_samples(
        args.pair_config, args.dataset_root, args.camera_parameters, tuple(args.rgb_record),
        tuple(args.base_record), tuple(config["input_size_hw"]), device, selected={primary_key, control_key},
    )
    sample = next(value for value in samples if (value.scene, value.source_frame, value.target_frame) == primary_key)
    control = next(value for value in samples if (value.scene, value.source_frame, value.target_frame) == control_key)
    if sample.split != "train" or control.split != "train":
        raise RuntimeError("single-target overfit and camera control must use train split")
    preparation = {
        "sample": primary_key, "control": control_key,
        "shape": list(sample.target_rgb.shape),
        "region_pixels": {
            "observed": int(sample.observed.sum()), "occlusion_hidden": int(sample.occluded.sum()),
            "outside_source_fov": int(sample.outside.sum()),
        },
        "warp_coverage": float(sample.warp_valid.mean()),
        "source_rgb_minmax": [float(sample.source_rgb.min()), float(sample.source_rgb.max())],
        "base_depth_scale": float(sample.source_depth_scale),
        "all_inputs_finite": all(torch.isfinite(value).all() for value in sample.inputs()),
        "input_hashes": sample.input_hashes,
    }
    if args.prepare_only:
        print(json.dumps(preparation, indent=2))
        return 0

    seed = int(config["optimization"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = SourceFeatureTargetViewNet(base_channels=int(config["architecture"]["base_channels"])).to(device)
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt["steps"], eta_min=opt["learning_rate"] / 100
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        initial_output = model(*sample.inputs())
        initial = target_metrics(initial_output, sample)
    history = []
    for step in range(1, int(opt["steps"]) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(*sample.inputs())
        loss = compute_target_view_loss(
            output, sample.target_rgb, sample.target_depth, sample.observed, sample.occluded, sample.outside,
            depth_weight=opt["depth_weight"], support_weight=opt["support_weight"],
            confidence_weight=opt["confidence_weight"],
        )
        if not torch.isfinite(loss.total):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 50 == 0 or step == opt["steps"]:
            model.eval()
            with torch.no_grad():
                current = target_metrics(model(*sample.inputs()), sample)
            history.append({
                "step": step, "learning_rate": scheduler.get_last_lr()[0], "total": float(loss.total.detach()),
                "rgb": float(loss.rgb.detach()), "log_depth": float(loss.log_depth.detach()),
                "occlusion_support": float(loss.occlusion_support.detach()),
                "outside_support": float(loss.outside_support.detach()), "metrics": current,
            })
    model.eval()
    with torch.no_grad():
        final_output = model(*sample.inputs())
        control_output = model(*control.inputs())
        final = target_metrics(final_output, sample)
        control_rgb_difference = float(torch.abs(final_output.rgb - control_output.rgb).mean())
    gate_cfg = config["gates"]
    gates: dict[str, bool] = {}
    for region_name in ("occlusion_hidden", "outside_source_fov"):
        before, after = initial["regions"][region_name], final["regions"][region_name]
        improvement = 1 - after["rgb_mae_0_255"] / before["rgb_mae_0_255"]
        gates[f"{region_name}_rgb_improvement"] = improvement >= gate_cfg["both_novel_rgb_mae_relative_improvement_minimum"]
        gates[f"{region_name}_rgb_absolute"] = after["rgb_mae_0_255"] <= gate_cfg["both_novel_rgb_mae_0_255_maximum"]
        gates[f"{region_name}_depth"] = after["depth_abs_rel"] <= gate_cfg["both_novel_depth_abs_rel_maximum"]
        gates[f"{region_name}_positive_support"] = after["branch_positive_support_mean"] >= gate_cfg["branch_positive_support_minimum"]
        gates[f"{region_name}_negative_support"] = after["branch_negative_support_mean"] <= gate_cfg["branch_negative_support_maximum"]
    gates["control_target_changes_prediction"] = control_rgb_difference >= gate_cfg["control_target_rgb_mean_difference_minimum_0_1"]
    gates["all_outputs_finite"] = all(torch.isfinite(value).all() for value in (
        final_output.rgb, final_output.depth_z, final_output.occlusion_support_probability,
        final_output.outside_support_probability, final_output.geometry_confidence,
        final_output.appearance_confidence,
    ))
    gates = {name: bool(value) for name, value in gates.items()}
    passed = all(gates.values())
    args.output_dir.mkdir(parents=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    checkpoint, arrays, history_path = (
        args.output_dir / "model.pt", args.output_dir / "prediction.npz", args.output_dir / "history.json"
    )
    torch.save({
        "model": model.state_dict(), "architecture": config["architecture"],
        "input_size_hw": config["input_size_hw"], "sample": list(primary_key),
    }, checkpoint)
    np.savez_compressed(
        arrays,
        rgb_float32=final_output.rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
        depth_z_float32=final_output.depth_z[0, 0].cpu().numpy().astype(np.float32),
        occlusion_support_float32=final_output.occlusion_support_probability[0, 0].cpu().numpy().astype(np.float32),
        outside_support_float32=final_output.outside_support_probability[0, 0].cpu().numpy().astype(np.float32),
        geometry_confidence_float32=final_output.geometry_confidence[0, 0].cpu().numpy().astype(np.float32),
        appearance_confidence_float32=final_output.appearance_confidence[0, 0].cpu().numpy().astype(np.float32),
    )
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    previews = {}
    for name, value in (
        ("source", sample.source_rgb), ("warped", sample.warped_rgb),
        ("target", sample.target_rgb), ("prediction", final_output.rgb),
    ):
        path = args.output_dir / f"{name}.png"; save_rgb(path, value); previews[name] = path
    mask_path = args.output_dir / "regions.png"; save_mask(mask_path, sample); previews["regions"] = mask_path
    record = {
        "schema_version": "stage1.9-source-feature-single-overfit-v1",
        "status": "pass" if passed else "failed", "producer_machine_id": "linux5080",
        "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sample": {
            "split": sample.split, "scene": sample.scene, "camera": sample.camera,
            "source_role": sample.source_role, "target_role": sample.target_role,
            "source_frame": sample.source_frame, "target_frame": sample.target_frame,
            "yaw_deg": sample.yaw_deg, "control_target_frame": control.target_frame,
        },
        "preparation": preparation,
        "inputs": {
            "training_config": portable(args.training_config), "training_config_sha256": sha256(args.training_config),
            "pair_config": portable(args.pair_config), "pair_config_sha256": sha256(args.pair_config),
            "rgb_records": [{"path": portable(p), "sha256": sha256(p)} for p in args.rgb_record],
            "base_records": [{"path": portable(p), "sha256": sha256(p)} for p in args.base_record],
            **sample.input_hashes, "model_inputs": config["architecture"]["inputs"],
            "source_truth_depth_as_model_input": False, "target_rgb_depth_as_model_input": False,
            "truth_masks_as_model_input": False, "held_out_test_used": False,
        },
        "initial_metrics": initial, "final_metrics": final,
        "control_target_rgb_mean_difference_0_1": control_rgb_difference, "gates": gates,
        "artifacts": {
            "checkpoint": portable(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "arrays": portable(arrays), "arrays_sha256": sha256(arrays),
            "history": portable(history_path), "history_sha256": sha256(history_path),
            "previews": {name: {"path": portable(path), "sha256": sha256(path)} for name, path in previews.items()},
        },
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "quality_boundary": "single train directed-pair memorization only; not validation, held-out test, P01, CLB or Gaussian quality",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "initial": initial, "final": final, "gates": gates}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
