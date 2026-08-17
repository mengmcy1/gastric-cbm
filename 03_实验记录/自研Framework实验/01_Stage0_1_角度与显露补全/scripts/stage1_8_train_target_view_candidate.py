#!/usr/bin/env python3
"""Train and validate the first explicit target-view-conditioned RGB-D candidate."""

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

from adaptive3dgs.models import TargetViewUNet, compute_target_view_loss
from stage1_8_target_view_common import portable, prepare_target_samples, sha256, target_metrics


REGIONS = ("occlusion_hidden", "outside_source_fov")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--pool-record", type=Path, required=True)
    parser.add_argument("--triplet-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--visibility-root", type=Path, required=True)
    parser.add_argument("--visibility-record", type=Path, required=True)
    parser.add_argument("--base-record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def aggregate(model: TargetViewUNet, samples) -> dict[str, object]:
    totals = {name: {key: 0.0 for key in (
        "rgb_mae_0_255", "depth_abs_rel", "branch_positive_support_mean", "branch_negative_support_mean",
        "geometry_confidence_mean", "appearance_confidence_mean",
    )} for name in REGIONS}
    pixels = {name: 0 for name in REGIONS}
    warp_coverage = 0.0
    finite = True
    model.eval()
    with torch.no_grad():
        for sample in samples:
            output = model(*sample.inputs())
            metrics = target_metrics(output, sample)
            warp_coverage += metrics["warp_coverage"]
            finite &= all(torch.isfinite(value).all() for value in (
                output.rgb, output.depth_z, output.occlusion_support_probability,
                output.outside_support_probability, output.geometry_confidence, output.appearance_confidence,
            ))
            for name in REGIONS:
                region = metrics["regions"][name]
                count = region["pixels"]
                pixels[name] += count
                for key in totals[name]:
                    totals[name][key] += region[key] * count
    regions = {}
    for name in REGIONS:
        regions[name] = {"pixels": pixels[name]}
        regions[name].update({key: value / pixels[name] for key, value in totals[name].items()})
        regions[name]["support_separation"] = (
            regions[name]["branch_positive_support_mean"] - regions[name]["branch_negative_support_mean"]
        )
    score = sum(
        regions[name]["rgb_mae_0_255"] / 255
        + regions[name]["depth_abs_rel"]
        + (1 - regions[name]["branch_positive_support_mean"])
        + regions[name]["branch_negative_support_mean"]
        for name in REGIONS
    ) / len(REGIONS)
    return {
        "samples": len(samples), "mean_warp_coverage": warp_coverage / len(samples),
        "regions": regions, "selection_score": score, "all_outputs_finite": finite,
    }


def warp_baseline(samples) -> dict[str, object]:
    totals = {name: {"rgb_mae_0_255": 0.0, "depth_abs_rel": 0.0} for name in REGIONS}
    pixels = {name: 0 for name in REGIONS}
    for sample in samples:
        for name, mask in (("occlusion_hidden", sample.occluded), ("outside_source_fov", sample.outside)):
            valid = mask & torch.isfinite(sample.target_depth) & (sample.target_depth > 0)
            count = int(valid.sum())
            rgb_error = torch.abs(sample.warped_rgb - sample.target_rgb).mean(dim=1, keepdim=True)[valid]
            depth_error = torch.abs(sample.warped_depth[valid] - sample.target_depth[valid]) / sample.target_depth[valid]
            pixels[name] += count
            totals[name]["rgb_mae_0_255"] += float(rgb_error.sum() * 255)
            totals[name]["depth_abs_rel"] += float(depth_error.sum())
    return {
        "regions": {
            name: {"pixels": pixels[name], **{key: value / pixels[name] for key, value in totals[name].items()}}
            for name in REGIONS
        }
    }


def save_rgb(path: Path, value: torch.Tensor) -> None:
    image = value[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray(np.uint8(np.rint(image * 255))).save(path)


def main() -> int:
    args = parse_args()
    for name in (
        "training_config", "pool_record", "triplet_config", "dataset_root", "camera_parameters",
        "visibility_root", "visibility_record", "base_record", "output_dir", "record",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with physical GPU index")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = json.loads(args.training_config.read_text(encoding="utf-8"))
    pool = json.loads(args.pool_record.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_run" or pool.get("status") != "pass":
        raise RuntimeError("training config/pool must be frozen and passed")
    selected = {(item["scene"], item["side"]) for item in pool["eligible_targets"]}
    opt = config["optimization"]; seed = int(opt["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0"); size = tuple(config["input_size_hw"])
    samples = prepare_target_samples(
        args.triplet_config, args.dataset_root, args.camera_parameters, args.visibility_root,
        args.base_record, size, device, selected=selected,
    )
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "val"]
    if len(train) != config["expected_train_targets"] or len(validation) != config["expected_validation_targets"]:
        raise RuntimeError(f"qualified sample count mismatch: {len(train)}/{len(validation)}")
    model = TargetViewUNet(base_channels=config["architecture"]["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt["optimizer_updates"], eta_min=opt["learning_rate"] / 100
    )
    args.output_dir.mkdir(parents=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
    initial_train = aggregate(model, train); initial_validation = aggregate(model, validation)
    train_warp = warp_baseline(train); val_warp = warp_baseline(validation)
    history = []; best_score = float("inf"); best_update = -1; best_state = None; best_validation = None
    order = list(range(len(train))); rng = random.Random(seed); rng.shuffle(order); cursor = 0
    for update in range(1, int(opt["optimizer_updates"]) + 1):
        if cursor + opt["batch_size"] > len(order):
            rng.shuffle(order); cursor = 0
        indices = order[cursor : cursor + opt["batch_size"]]; cursor += opt["batch_size"]
        model.train(); optimizer.zero_grad(set_to_none=True); loss_values = []
        for index in indices:
            sample = train[index]; output = model(*sample.inputs())
            loss = compute_target_view_loss(
                output, sample.target_rgb, sample.target_depth, sample.observed, sample.occluded, sample.outside,
                depth_weight=opt["depth_weight"], support_weight=opt["support_weight"],
                confidence_weight=opt["confidence_weight"],
            )
            if not torch.isfinite(loss.total):
                raise RuntimeError(f"non-finite loss at update {update}: {sample.target_id}")
            (loss.total / len(indices)).backward(); loss_values.append(float(loss.total.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); scheduler.step()
        if update % opt["validation_interval_updates"] == 0 or update == opt["optimizer_updates"]:
            validation_metrics = aggregate(model, validation)
            history.append({
                "update": update, "learning_rate": scheduler.get_last_lr()[0],
                "mean_train_batch_loss": float(np.mean(loss_values)), "validation": validation_metrics,
            })
            if validation_metrics["selection_score"] < best_score:
                best_score = validation_metrics["selection_score"]; best_update = update
                best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                best_validation = validation_metrics
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state); model.eval()
    best_train = aggregate(model, train); best_validation = aggregate(model, validation)
    cfg_gates = config["gates"]; gates = {}
    for name in REGIONS:
        val_region = best_validation["regions"][name]; train_region = best_train["regions"][name]
        val_improvement = 1 - val_region["rgb_mae_0_255"] / val_warp["regions"][name]["rgb_mae_0_255"]
        train_improvement = 1 - train_region["rgb_mae_0_255"] / train_warp["regions"][name]["rgb_mae_0_255"]
        gates[f"validation_{name}_rgb_relative_improvement"] = val_improvement >= cfg_gates["validation_novel_rgb_relative_improvement_over_warp_minimum"]
        gates[f"validation_{name}_rgb_absolute"] = val_region["rgb_mae_0_255"] <= cfg_gates["validation_novel_rgb_mae_0_255_maximum"]
        gates[f"validation_{name}_depth"] = val_region["depth_abs_rel"] <= cfg_gates["validation_novel_depth_abs_rel_maximum"]
        gates[f"validation_{name}_support_separation"] = val_region["support_separation"] >= cfg_gates["validation_branch_support_separation_minimum"]
        gates[f"train_{name}_rgb_relative_improvement"] = train_improvement >= cfg_gates["train_novel_rgb_relative_improvement_over_warp_minimum"]
    gates["all_best_outputs_finite"] = best_train["all_outputs_finite"] and best_validation["all_outputs_finite"]
    gates = {key: bool(value) for key, value in gates.items()}; passed = all(gates.values())
    checkpoint = args.output_dir / "best_model.pt"; history_path = args.output_dir / "history.json"
    torch.save({
        "model": best_state, "architecture": config["architecture"], "input_size_hw": list(size),
        "best_update": best_update, "selection_score": best_score,
    }, checkpoint)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    predictions = []
    with torch.no_grad():
        for sample in validation:
            output = model(*sample.inputs()); prefix = f"val-{sample.scene}-{sample.side}-{sample.target_frame:04d}"
            pred_path = args.output_dir / f"{prefix}-prediction.png"; target_path = args.output_dir / f"{prefix}-target.png"
            save_rgb(pred_path, output.rgb); save_rgb(target_path, sample.target_rgb)
            predictions.append({
                "scene": sample.scene, "side": sample.side, "target_frame": sample.target_frame,
                "metrics": target_metrics(output, sample), **sample.input_hashes,
                "prediction": portable(pred_path), "prediction_sha256": sha256(pred_path),
                "target_preview": portable(target_path), "target_preview_sha256": sha256(target_path),
            })
    record = {
        "schema_version": "stage1.8-target-view-candidate-v1",
        "status": "pass_candidate_frozen" if passed else "failed",
        "producer_machine_id": "linux5080", "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": visible,
        "inputs": {
            "training_config": portable(args.training_config), "training_config_sha256": sha256(args.training_config),
            "pool_record": portable(args.pool_record), "pool_record_sha256": sha256(args.pool_record),
            "triplet_config": portable(args.triplet_config), "triplet_config_sha256": sha256(args.triplet_config),
            "visibility_record": portable(args.visibility_record), "visibility_record_sha256": sha256(args.visibility_record),
            "base_record": portable(args.base_record), "base_record_sha256": sha256(args.base_record),
            "model_inputs": config["architecture"]["inputs"], "target_rgb_depth_as_model_input": False,
            "truth_masks_as_model_input": False, "held_out_test_used": False, "p01_used": False,
        },
        "sample_counts": {"train": len(train), "validation": len(validation)},
        "warp_baselines": {"train": train_warp, "validation": val_warp},
        "initial_train": initial_train, "initial_validation": initial_validation,
        "best_update": best_update, "best_train": best_train, "best_validation": best_validation,
        "gates": gates, "validation_predictions": predictions,
        "artifacts": {
            "checkpoint": portable(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "history": portable(history_path), "history_sha256": sha256(history_path),
        },
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "quality_boundary": "64-train/14-val target candidate selected without held-out test or P01; pass is required before test",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": record["status"], "best_update": best_update,
        "warp_validation": val_warp, "best_validation": best_validation, "gates": gates,
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
