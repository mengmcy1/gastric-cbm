#!/usr/bin/env python3
"""Stage 1.8-1 single-target overfit gate for explicit target-camera conditioning."""

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

from adaptive3dgs.models import TargetViewUNet, compute_target_view_loss
from stage1_8_target_view_common import portable, prepare_target_samples, sha256, target_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
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
    for name in (
        "training_config", "triplet_config", "dataset_root", "camera_parameters", "visibility_root",
        "visibility_record", "base_record", "output_dir", "record",
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
    if config.get("status") != "frozen_before_run":
        raise RuntimeError("training config must be frozen_before_run")
    seed = int(config["optimization"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    size = tuple(config["input_size_hw"])
    scene = config["sample"]["scene"]
    side = config["sample"]["target_side"]
    samples = prepare_target_samples(
        args.triplet_config, args.dataset_root, args.camera_parameters, args.visibility_root,
        args.base_record, size, device, selected={(scene, side), (scene, "right" if side == "left" else "left")},
    )
    sample = next(value for value in samples if value.scene == scene and value.side == side)
    opposite = next(value for value in samples if value.scene == scene and value.side != side)
    if sample.split != "train":
        raise RuntimeError("single-target overfit must use train split")
    model = TargetViewUNet(base_channels=int(config["architecture"]["base_channels"])).to(device)
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
            with torch.no_grad(): current = target_metrics(model(*sample.inputs()), sample)
            history.append({
                "step": step, "learning_rate": scheduler.get_last_lr()[0], "total": float(loss.total.detach()),
                "rgb": float(loss.rgb.detach()), "log_depth": float(loss.log_depth.detach()),
                "occlusion_support": float(loss.occlusion_support.detach()), "outside_support": float(loss.outside_support.detach()),
                "metrics": current,
            })
    model.eval()
    with torch.no_grad():
        final_output = model(*sample.inputs())
        opposite_output = model(*opposite.inputs())
        final = target_metrics(final_output, sample)
        opposite_rgb_difference = float(torch.abs(final_output.rgb - opposite_output.rgb).mean())
    gates_cfg = config["gates"]
    gates = {}
    for region_name in ("occlusion_hidden", "outside_source_fov"):
        initial_region = initial["regions"][region_name]
        final_region = final["regions"][region_name]
        improvement = 1 - final_region["rgb_mae_0_255"] / initial_region["rgb_mae_0_255"]
        gates[f"{region_name}_rgb_improvement"] = improvement >= gates_cfg["both_novel_rgb_mae_relative_improvement_minimum"]
        gates[f"{region_name}_rgb_absolute"] = final_region["rgb_mae_0_255"] <= gates_cfg["both_novel_rgb_mae_0_255_maximum"]
        gates[f"{region_name}_depth"] = final_region["depth_abs_rel"] <= gates_cfg["both_novel_depth_abs_rel_maximum"]
        gates[f"{region_name}_positive_support"] = final_region["branch_positive_support_mean"] >= gates_cfg["branch_positive_support_minimum"]
        gates[f"{region_name}_negative_support"] = final_region["branch_negative_support_mean"] <= gates_cfg["branch_negative_support_maximum"]
    gates["opposite_target_changes_prediction"] = opposite_rgb_difference >= gates_cfg["opposite_target_rgb_mean_difference_minimum_0_1"]
    gates["all_outputs_finite"] = all(torch.isfinite(value).all() for value in (
        final_output.rgb, final_output.depth_z, final_output.occlusion_support_probability,
        final_output.outside_support_probability, final_output.geometry_confidence,
        final_output.appearance_confidence,
    ))
    gates = {name: bool(value) for name, value in gates.items()}
    passed = all(gates.values())
    args.output_dir.mkdir(parents=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    arrays = args.output_dir / "prediction.npz"
    history_path = args.output_dir / "history.json"
    torch.save({
        "model": model.state_dict(), "architecture": config["architecture"], "input_size_hw": list(size),
        "scene": scene, "side": side,
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
    for name, value in (("warped", sample.warped_rgb), ("target", sample.target_rgb), ("prediction", final_output.rgb)):
        path = args.output_dir / f"{name}.png"; save_rgb(path, value); previews[name] = path
    mask_path = args.output_dir / "regions.png"; save_mask(mask_path, sample); previews["regions"] = mask_path
    record = {
        "schema_version": "stage1.8-target-view-single-overfit-v1",
        "status": "pass" if passed else "failed",
        "producer_machine_id": "linux5080", "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": visible,
        "sample": {"split": sample.split, "scene": scene, "source_frame": sample.source_frame, "side": side, "target_frame": sample.target_frame},
        "inputs": {
            "training_config": portable(args.training_config), "training_config_sha256": sha256(args.training_config),
            "triplet_config": portable(args.triplet_config), "triplet_config_sha256": sha256(args.triplet_config),
            "visibility_record": portable(args.visibility_record), "visibility_record_sha256": sha256(args.visibility_record),
            "base_record": portable(args.base_record), "base_record_sha256": sha256(args.base_record),
            **sample.input_hashes,
            "model_inputs": config["architecture"]["inputs"],
            "target_rgb_depth_as_model_input": False, "truth_masks_as_model_input": False,
            "held_out_test_used": False,
        },
        "initial_metrics": initial, "final_metrics": final,
        "opposite_target_rgb_mean_difference_0_1": opposite_rgb_difference,
        "gates": gates,
        "artifacts": {
            "checkpoint": portable(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "arrays": portable(arrays), "arrays_sha256": sha256(arrays),
            "history": portable(history_path), "history_sha256": sha256(history_path),
            "previews": {name: {"path": portable(path), "sha256": sha256(path)} for name, path in previews.items()},
        },
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "quality_boundary": "single train-target memorization only; not validation, held-out test, P01, CLB or Gaussian quality",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "initial": initial, "final": final, "gates": gates}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
