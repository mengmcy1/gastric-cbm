#!/usr/bin/env python3
"""Train and validate a Stage 1.9 source-feature target-view candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from adaptive3dgs.models import (
    FrozenResNetSourceFeatureTargetViewNet,
    FrozenResNetSpatialProxyTargetViewNet,
    SourceFeatureTargetViewNet,
    compute_target_view_loss,
)
from stage1_9_source_feature_common import portable, prepare_directed_samples, sha256, target_metrics


REGIONS = ("occlusion_hidden", "outside_source_fov")


def run_model(model, sample):
    inputs = sample.spatial_inputs() if isinstance(model, FrozenResNetSpatialProxyTargetViewNet) else sample.inputs()
    return model(*inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--pool-record", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--rgb-record", type=Path, action="append", required=True)
    parser.add_argument("--base-record", type=Path, action="append", required=True)
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def aggregate(model: SourceFeatureTargetViewNet, samples) -> dict[str, object]:
    keys = (
        "rgb_mae_0_255", "depth_abs_rel", "branch_positive_support_mean",
        "branch_negative_support_mean", "geometry_confidence_mean", "appearance_confidence_mean",
    )
    totals = {name: {key: 0.0 for key in keys} for name in REGIONS}
    pixels = {name: 0 for name in REGIONS}
    coverage = 0.0; finite = True
    model.eval()
    with torch.no_grad():
        for sample in samples:
            output = run_model(model, sample); metrics = target_metrics(output, sample)
            coverage += metrics["warp_coverage"]
            finite &= all(torch.isfinite(value).all() for value in (
                output.rgb, output.depth_z, output.occlusion_support_probability,
                output.outside_support_probability, output.geometry_confidence, output.appearance_confidence,
            ))
            for name in REGIONS:
                region, count = metrics["regions"][name], metrics["regions"][name]["pixels"]
                pixels[name] += count
                for key in keys:
                    totals[name][key] += region[key] * count
    regions = {}
    for name in REGIONS:
        regions[name] = {"pixels": pixels[name]}
        regions[name].update({key: value / pixels[name] for key, value in totals[name].items()})
        regions[name]["support_separation"] = (
            regions[name]["branch_positive_support_mean"] - regions[name]["branch_negative_support_mean"]
        )
    score = sum(
        regions[name]["rgb_mae_0_255"] / 255 + regions[name]["depth_abs_rel"]
        + (1 - regions[name]["branch_positive_support_mean"])
        + regions[name]["branch_negative_support_mean"] for name in REGIONS
    ) / len(REGIONS)
    return {
        "samples": len(samples), "mean_warp_coverage": coverage / len(samples),
        "regions": regions, "selection_score": score, "all_outputs_finite": finite,
    }


def warp_baseline(samples) -> dict[str, object]:
    totals = {name: {"rgb_mae_0_255": 0.0, "depth_abs_rel": 0.0} for name in REGIONS}
    pixels = {name: 0 for name in REGIONS}
    for sample in samples:
        for name, mask in (("occlusion_hidden", sample.occluded), ("outside_source_fov", sample.outside)):
            valid = mask & torch.isfinite(sample.target_depth) & (sample.target_depth > 0)
            count = int(valid.sum())
            rgb = torch.abs(sample.warped_rgb - sample.target_rgb).mean(dim=1, keepdim=True)[valid]
            depth = torch.abs(sample.warped_depth[valid] - sample.target_depth[valid]) / sample.target_depth[valid]
            pixels[name] += count
            totals[name]["rgb_mae_0_255"] += float(rgb.sum() * 255)
            totals[name]["depth_abs_rel"] += float(depth.sum())
    return {"regions": {
        name: {"pixels": pixels[name], **{key: value / pixels[name] for key, value in totals[name].items()}}
        for name in REGIONS
    }}


def save_rgb(path: Path, value: torch.Tensor) -> None:
    image = value[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray(np.uint8(np.rint(image * 255))).save(path)


def main() -> int:
    args = parse_args()
    for name in ("training_config", "pool_record", "dataset_root", "camera_parameters", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    args.rgb_record = [path.resolve() for path in args.rgb_record]
    args.base_record = [path.resolve() for path in args.base_record]
    if args.backbone_weights is not None:
        args.backbone_weights = args.backbone_weights.resolve()
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    config = json.loads(args.training_config.read_text(encoding="utf-8"))
    pool = json.loads(args.pool_record.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_run" or pool.get("status") != "pass":
        raise RuntimeError("training config/pool must be frozen and passed")
    selected = {(p["scene"], int(p["source_frame"]), int(p["target_frame"])) for p in pool["pairs"]}
    device = torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.prepare_only:
        if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with physical GPU index")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda:0")
    samples = prepare_directed_samples(
        args.pool_record, args.dataset_root, args.camera_parameters, tuple(args.rgb_record),
        tuple(args.base_record), tuple(config["input_size_hw"]), device, selected=selected,
    )
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "val"]
    if len(train) != config["expected_train_targets"] or len(validation) != config["expected_validation_targets"]:
        raise RuntimeError(f"sample count mismatch: {len(train)}/{len(validation)}")
    preparation = {
        "train_targets": len(train), "validation_targets": len(validation),
        "train_scenes": len({s.scene for s in train}), "validation_scenes": len({s.scene for s in validation}),
        "train_region_pixels": {
            "occlusion_hidden": sum(int(s.occluded.sum()) for s in train),
            "outside_source_fov": sum(int(s.outside.sum()) for s in train),
        },
        "validation_region_pixels": {
            "occlusion_hidden": sum(int(s.occluded.sum()) for s in validation),
            "outside_source_fov": sum(int(s.outside.sum()) for s in validation),
        },
        "source_plane_proxy": {
            "valid_fraction": float(torch.stack([s.source_plane_proxy_valid.mean() for s in samples]).mean()),
            "in_source_fov_fraction": float(torch.stack([
                (
                    (s.source_plane_proxy_grid[..., 0].abs() <= 1)
                    & (s.source_plane_proxy_grid[..., 1].abs() <= 1)
                    & (s.source_plane_proxy_valid[:, 0] > 0.5)
                ).float().mean() for s in samples
            ]).mean()),
            "derived_from_target_labels": False,
        },
        "all_inputs_finite": all(all(torch.isfinite(v).all() for v in s.spatial_inputs()) for s in samples),
    }
    if args.prepare_only:
        print(json.dumps(preparation, indent=2)); return 0

    opt = config["optimization"]; seed = int(opt["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    architecture_name = config["architecture"]["name"]
    if architecture_name == "SourceFeatureTargetViewNet":
        if args.backbone_weights is not None:
            raise RuntimeError("from-scratch architecture must not receive backbone weights")
        model = SourceFeatureTargetViewNet(base_channels=int(config["architecture"]["base_channels"]))
    elif architecture_name == "FrozenResNetSourceFeatureTargetViewNet":
        if args.backbone_weights is None:
            raise RuntimeError("frozen ResNet architecture requires --backbone-weights")
        from torchvision.models import resnet50
        backbone = resnet50(weights=None)
        state = torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        model = FrozenResNetSourceFeatureTargetViewNet(
            backbone, base_channels=int(config["architecture"]["base_channels"])
        )
    elif architecture_name == "FrozenResNetSpatialProxyTargetViewNet":
        if args.backbone_weights is None:
            raise RuntimeError("frozen ResNet architecture requires --backbone-weights")
        from torchvision.models import resnet50
        backbone = resnet50(weights=None)
        state = torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        model = FrozenResNetSpatialProxyTargetViewNet(
            backbone, base_channels=int(config["architecture"]["base_channels"])
        )
    else:
        raise RuntimeError(f"unsupported architecture: {architecture_name}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt["optimizer_updates"], eta_min=opt["learning_rate"] / 100
    )
    args.output_dir.mkdir(parents=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
    initial_train = aggregate(model, train); initial_validation = aggregate(model, validation)
    train_warp, validation_warp = warp_baseline(train), warp_baseline(validation)
    history = []; best_score = float("inf"); best_update = -1; best_state = None
    order = list(range(len(train))); rng = random.Random(seed); rng.shuffle(order); cursor = 0
    for update in range(1, int(opt["optimizer_updates"]) + 1):
        if cursor + opt["batch_size"] > len(order):
            rng.shuffle(order); cursor = 0
        indices = order[cursor:cursor + opt["batch_size"]]; cursor += opt["batch_size"]
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for index in indices:
            sample = train[index]; output = run_model(model, sample)
            loss = compute_target_view_loss(
                output, sample.target_rgb, sample.target_depth, sample.observed, sample.occluded, sample.outside,
                depth_weight=opt["depth_weight"], support_weight=opt["support_weight"],
                confidence_weight=opt["confidence_weight"],
            )
            if not torch.isfinite(loss.total):
                raise RuntimeError(f"non-finite loss at update {update}: {sample.scene}/{sample.target_frame}")
            (loss.total / len(indices)).backward(); losses.append(float(loss.total.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); scheduler.step()
        if update % opt["validation_interval_updates"] == 0 or update == opt["optimizer_updates"]:
            metrics = aggregate(model, validation)
            history.append({
                "update": update, "learning_rate": scheduler.get_last_lr()[0],
                "mean_train_batch_loss": float(np.mean(losses)), "validation": metrics,
            })
            if metrics["selection_score"] < best_score:
                best_score, best_update = metrics["selection_score"], update
                best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state); model.eval()
    best_train, best_validation = aggregate(model, train), aggregate(model, validation)
    gate_cfg = config["gates"]; gates = {}
    for name in REGIONS:
        val_region, train_region = best_validation["regions"][name], best_train["regions"][name]
        val_improvement = 1 - val_region["rgb_mae_0_255"] / validation_warp["regions"][name]["rgb_mae_0_255"]
        train_improvement = 1 - train_region["rgb_mae_0_255"] / train_warp["regions"][name]["rgb_mae_0_255"]
        gates[f"validation_{name}_rgb_relative_improvement"] = val_improvement >= gate_cfg["validation_novel_rgb_relative_improvement_over_warp_minimum"]
        gates[f"validation_{name}_rgb_absolute"] = val_region["rgb_mae_0_255"] <= gate_cfg["validation_novel_rgb_mae_0_255_maximum"]
        gates[f"validation_{name}_depth"] = val_region["depth_abs_rel"] <= gate_cfg["validation_novel_depth_abs_rel_maximum"]
        gates[f"validation_{name}_support_separation"] = val_region["support_separation"] >= gate_cfg["validation_branch_support_separation_minimum"]
        gates[f"train_{name}_rgb_relative_improvement"] = train_improvement >= gate_cfg["train_novel_rgb_relative_improvement_over_warp_minimum"]
    gates["all_best_outputs_finite"] = best_train["all_outputs_finite"] and best_validation["all_outputs_finite"]
    gates = {key: bool(value) for key, value in gates.items()}; passed = all(gates.values())
    checkpoint, history_path = args.output_dir / "best_model.pt", args.output_dir / "history.json"
    torch.save({
        "model": best_state, "architecture": config["architecture"],
        "input_size_hw": config["input_size_hw"], "best_update": best_update,
        "selection_score": best_score,
    }, checkpoint)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    predictions = []
    with torch.no_grad():
        for sample in validation:
            output = run_model(model, sample)
            prefix = f"val-{sample.scene}-{sample.source_frame:04d}-to-{sample.target_frame:04d}"
            pred_path, target_path = args.output_dir / f"{prefix}-prediction.png", args.output_dir / f"{prefix}-target.png"
            save_rgb(pred_path, output.rgb); save_rgb(target_path, sample.target_rgb)
            predictions.append({
                "scene": sample.scene, "source_frame": sample.source_frame,
                "target_frame": sample.target_frame, "source_role": sample.source_role,
                "target_role": sample.target_role, "metrics": target_metrics(output, sample),
                **sample.input_hashes, "prediction": portable(pred_path),
                "prediction_sha256": sha256(pred_path), "target_preview": portable(target_path),
                "target_preview_sha256": sha256(target_path),
            })
    record = {
        "schema_version": config["schema_version"],
        "status": "pass_candidate_frozen" if passed else "failed", "producer_machine_id": "linux5080",
        "physical_gpu_index": args.physical_gpu_index, "visible_cuda_devices": visible,
        "inputs": {
            "training_config": portable(args.training_config), "training_config_sha256": sha256(args.training_config),
            "pool_record": portable(args.pool_record), "pool_record_sha256": sha256(args.pool_record),
            "rgb_records": [{"path": portable(p), "sha256": sha256(p)} for p in args.rgb_record],
            "base_records": [{"path": portable(p), "sha256": sha256(p)} for p in args.base_record],
            "backbone_weights": None if args.backbone_weights is None else {
                "path": portable(args.backbone_weights), "sha256": sha256(args.backbone_weights),
            },
            "model_inputs": config["architecture"]["inputs"], "source_truth_depth_as_model_input": False,
            "target_rgb_depth_as_model_input": False, "truth_masks_as_model_input": False,
            "held_out_test_used": False, "p01_used": False,
        },
        "preparation": preparation, "sample_counts": {"train": len(train), "validation": len(validation)},
        "warp_baselines": {"train": train_warp, "validation": validation_warp},
        "initial_train": initial_train, "initial_validation": initial_validation,
        "best_update": best_update, "best_train": best_train, "best_validation": best_validation,
        "gates": gates, "validation_predictions": predictions,
        "artifacts": {
            "checkpoint": portable(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "history": portable(history_path), "history_sha256": sha256(history_path),
        },
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "quality_boundary": "8-train-scene/2-validation-scene short-circuit only; held-out test, P01 and Gaussian quality excluded",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": record["status"], "best_update": best_update,
        "warp_validation": validation_warp, "best_validation": best_validation, "gates": gates,
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
