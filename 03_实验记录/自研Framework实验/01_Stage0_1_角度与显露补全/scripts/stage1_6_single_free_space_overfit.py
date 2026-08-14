#!/usr/bin/env python3
"""Single-train-sample overfit gate for verified free-space support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from adaptive3dgs.datasets import load_hypersim_training_triplets
from adaptive3dgs.models import (
    OcclusionHiddenUNet,
    compute_hidden_geometry_loss,
    positive_verified_negative_logistic_loss,
)
from stage1_6_train_small_candidate import Sample, portable, sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-record", type=Path, required=True)
    parser.add_argument("--base-record", type=Path, required=True)
    parser.add_argument("--triplet-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--scene", default="ai_001_001")
    parser.add_argument("--source-frame", type=int, default=99)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def metrics(output, sample):
    positive = sample.positive
    negative = sample.verified_free_space_negative(output.delta_ratio)
    pred, target = output.delta_ratio[positive], sample.target_ratio[positive]
    return {
        "ratio_abs_rel": float(((pred - target).abs() / target.clamp_min(1e-4)).mean()),
        "ratio_mae": float((pred - target).abs().mean()),
        "positive_support_mean": float(output.support_probability[positive].mean()),
        "verified_negative_support_mean": (
            float(output.support_probability[negative].mean()) if negative.any() else None
        ),
        "verified_negative_pixels": int(negative.sum()),
        "positive_confidence_mean": float(output.geometry_confidence[positive].mean()),
        "positive_confidence_std": float(output.geometry_confidence[positive].std()),
    }


def main():
    args = parse_args()
    for name in (
        "evidence_record", "base_record", "triplet_config", "dataset_root",
        "camera_parameters", "output_dir", "record",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with physical GPU index")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    evidence_record = json.loads(args.evidence_record.read_text())
    base_record = json.loads(args.base_record.read_text())
    key = (args.scene, args.source_frame)
    evidence_item = next(x for x in evidence_record["results"] if (x["scene"], int(x["source_frame"])) == key)
    base_item = next(x for x in base_record["results"] if (x["scene"], int(x["source_frame"])) == key)
    triplet = next(
        x for x in load_hypersim_training_triplets(args.triplet_config, args.dataset_root, args.camera_parameters)
        if (x.scene, int(x.source_frame_index)) == key
    )
    if evidence_item["split"] != "train" or triplet.split != "train":
        raise RuntimeError("overfit sample must be train")
    sample = Sample(evidence_item, base_item, (384, 512), device, triplet)
    model = OcclusionHiddenUNet(base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate / 100
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        initial = metrics(model(sample.rgb, sample.base, sample.camera_rays_xy), sample)
    history = []
    for step in range(1, args.steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(sample.rgb, sample.base, sample.camera_rays_xy)
        geometry = compute_hidden_geometry_loss(
            output, sample.source, sample.hidden, sample.positive,
            class_prior=sample.class_prior, support_weight=0, uncertainty_weight=0.02,
        )
        negative = sample.verified_free_space_negative(output.delta_ratio)
        support = positive_verified_negative_logistic_loss(
            output.support_logits, sample.positive, negative
        )
        total = geometry.total + args.support_weight * support
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(): current = metrics(model(sample.rgb, sample.base, sample.camera_rays_xy), sample)
            history.append({"step": step, "total": float(total), "support_loss": float(support), **current})
    model.eval()
    with torch.no_grad():
        final_output = model(sample.rgb, sample.base, sample.camera_rays_xy)
        final = metrics(final_output, sample)
    improvement = 1 - final["ratio_abs_rel"] / initial["ratio_abs_rel"]
    separation = final["positive_support_mean"] - final["verified_negative_support_mean"]
    gates = {
        "ratio_abs_rel_improved_by_90_percent": improvement >= 0.9,
        "final_ratio_abs_rel_below_0_10": final["ratio_abs_rel"] < 0.1,
        "positive_minus_verified_negative_support_above_0_50": separation > 0.5,
        "verified_negative_pixels_nonzero": final["verified_negative_pixels"] > 0,
    }
    passed = all(gates.values())
    args.output_dir.mkdir(parents=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"; arrays = args.output_dir / "prediction.npz"; curve = args.output_dir / "history.json"
    torch.save({"model": model.state_dict(), "scene": args.scene, "source_frame": args.source_frame}, checkpoint)
    np.savez_compressed(
        arrays,
        delta_ratio_float32=final_output.delta_ratio[0, 0].detach().cpu().numpy().astype(np.float32),
        support_float32=final_output.support_probability[0, 0].detach().cpu().numpy().astype(np.float32),
        confidence_float32=final_output.geometry_confidence[0, 0].detach().cpu().numpy().astype(np.float32),
        positive_mask_uint8=sample.positive[0, 0].cpu().numpy().astype(np.uint8),
        verified_negative_mask_uint8=sample.verified_free_space_negative(final_output.delta_ratio)[0, 0].cpu().numpy().astype(np.uint8),
    )
    curve.write_text(json.dumps(history, indent=2) + "\n")
    record = {
        "schema_version": "stage1.6-single-free-space-overfit-v1",
        "status": "pass" if passed else "failed",
        "producer_machine_id": "linux5080",
        "physical_gpu_index": args.physical_gpu_index,
        "sample": {"split": "train", "scene": args.scene, "source_frame": args.source_frame},
        "model_inputs": ["source_rgb", "frozen_base_depth", "source_camera_rays_xy"],
        "source_truth_usage": "label-side ratio and free-space projection only",
        "initial": initial,
        "final": final,
        "ratio_abs_rel_relative_improvement": improvement,
        "support_separation": separation,
        "gates": gates,
        "artifacts": {
            "checkpoint": portable(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "arrays": portable(arrays), "arrays_sha256": sha256(arrays),
            "history": portable(curve), "history_sha256": sha256(curve),
        },
        "runtime": {"seconds": time.perf_counter()-started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "quality_boundary": "single train-sample free-space memorization only; no validation or test conclusion",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": record["status"], "initial": initial, "final": final, "gates": gates}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
