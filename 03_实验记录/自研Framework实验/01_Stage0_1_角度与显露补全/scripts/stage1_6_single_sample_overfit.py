#!/usr/bin/env python3
"""Run the Stage 1.6 single-sample hidden-geometry overfit gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from adaptive3dgs.models import OcclusionHiddenUNet, compute_hidden_geometry_loss


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-record", type=Path, required=True)
    parser.add_argument("--base-record", type=Path, required=True)
    parser.add_argument("--scene", default="ai_001_001")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--support-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def resize(value: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    kwargs = {"size": size, "mode": mode}
    if mode == "bilinear":
        kwargs.update({"align_corners": False, "antialias": True})
    return F.interpolate(value, **kwargs)


def save_map(path: Path, value: np.ndarray, valid: np.ndarray | None = None) -> None:
    finite = np.isfinite(value)
    if valid is not None:
        finite &= valid
    shown = np.zeros(value.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = np.quantile(value[finite], [0.02, 0.98])
        normalized = np.clip((value - lo) / max(float(hi - lo), 1e-6), 0, 1)
        shown[finite] = np.uint8(np.rint(normalized[finite] * 255))
    Image.fromarray(shown).save(path)


def metrics(output, target_ratio: torch.Tensor, positive: torch.Tensor) -> dict[str, float]:
    predicted = output.delta_ratio[positive]
    target = target_ratio[positive]
    rel = (predicted - target).abs() / target.clamp_min(1e-4)
    return {
        "ratio_abs_rel": float(rel.mean().item()),
        "ratio_mae": float((predicted - target).abs().mean().item()),
        "positive_support_mean": float(output.support_probability[positive].mean().item()),
        "unknown_support_mean": float(output.support_probability[~positive].mean().item()),
        "positive_confidence_mean": float(output.geometry_confidence[positive].mean().item()),
        "positive_confidence_std": float(output.geometry_confidence[positive].std().item()),
    }


def main() -> int:
    args = parse_args()
    for name in ("evidence_record", "base_record", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with --physical-gpu-index")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    evidence_record = json.loads(args.evidence_record.read_text(encoding="utf-8"))
    base_record = json.loads(args.base_record.read_text(encoding="utf-8"))
    evidence_item = next(item for item in evidence_record["results"] if item["scene"] == args.scene)
    base_item = next(item for item in base_record["results"] if item["scene"] == args.scene)
    if evidence_item["split"] != "train" or base_item["split"] != "train":
        raise RuntimeError("single-sample overfit is restricted to the train split")
    evidence_path, base_path = Path(evidence_item["arrays"]), Path(base_item["arrays"])
    if sha256(evidence_path) != evidence_item["arrays_sha256"] or sha256(base_path) != base_item["arrays_sha256"]:
        raise RuntimeError("input artifact hash mismatch")
    with np.load(evidence_path, allow_pickle=False) as data:
        rgb_np = data["source_rgb_uint8"].copy()
        source_truth_np = data["source_depth_z_label_only_float32"].copy()
        hidden_truth_np = data["hidden_depth_z_float32"].copy()
        positive_np = data["positive_mask_uint8"].astype(bool)
    with np.load(base_path, allow_pickle=False) as data:
        base_np = data["base_depth_z_float32"].copy()

    device = torch.device("cuda:0")
    size = (args.height, args.width)
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)[None].float().div(255)
    base = torch.from_numpy(base_np)[None, None].float()
    source_truth = torch.from_numpy(source_truth_np)[None, None].float()
    hidden_truth = torch.from_numpy(hidden_truth_np)[None, None].float()
    positive = torch.from_numpy(positive_np)[None, None].float()
    rgb = resize(rgb, size, "bilinear").to(device)
    base = resize(base, size, "bilinear").to(device)
    source_truth = resize(source_truth, size, "nearest").to(device)
    hidden_truth = resize(hidden_truth, size, "nearest").to(device)
    positive = resize(positive, size, "nearest").bool().to(device)
    valid_positive = positive & torch.isfinite(source_truth) & torch.isfinite(hidden_truth)
    valid_positive &= (source_truth > 0) & (hidden_truth > source_truth)
    target_ratio = torch.zeros_like(source_truth)
    target_ratio[valid_positive] = hidden_truth[valid_positive] / source_truth[valid_positive] - 1
    class_prior = float(valid_positive.float().mean().item())
    if not 0 < class_prior < 1:
        raise RuntimeError("invalid positive class prior")

    model = OcclusionHiddenUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate / 100
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        initial_output = model(rgb, base)
        initial = metrics(initial_output, target_ratio, valid_positive)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(rgb, base)
        loss = compute_hidden_geometry_loss(
            output,
            source_truth,
            hidden_truth,
            valid_positive,
            class_prior=class_prior,
            support_weight=args.support_weight,
            uncertainty_weight=args.uncertainty_weight,
        )
        if not torch.isfinite(loss.total):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            with torch.no_grad():
                current = metrics(output, target_ratio, valid_positive)
            history.append(
                {
                    "step": step + 1,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "total": float(loss.total.item()),
                    "relative_depth": float(loss.relative_depth.item()),
                    "support_pu": float(loss.support_pu.item()),
                    "uncertainty_nll": float(loss.uncertainty_nll.item()),
                    **current,
                }
            )
    model.eval()
    with torch.no_grad():
        final_output = model(rgb, base)
        final = metrics(final_output, target_ratio, valid_positive)

    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    arrays_path = args.output_dir / "prediction.npz"
    history_path = args.output_dir / "history.json"
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": {"base_channels": args.base_channels, "minimum_delta_ratio": 1e-4},
            "input_size_hw": list(size),
            "scene": args.scene,
        },
        checkpoint_path,
    )
    pred_ratio = final_output.delta_ratio[0, 0].cpu().numpy().astype(np.float32)
    pred_support = final_output.support_probability[0, 0].cpu().numpy().astype(np.float32)
    pred_confidence = final_output.geometry_confidence[0, 0].cpu().numpy().astype(np.float32)
    mask_np = valid_positive[0, 0].cpu().numpy()
    target_ratio_np = target_ratio[0, 0].cpu().numpy().astype(np.float32)
    np.savez_compressed(
        arrays_path,
        predicted_delta_ratio_float32=pred_ratio,
        predicted_support_probability_float32=pred_support,
        predicted_geometry_confidence_float32=pred_confidence,
        target_delta_ratio_positive_only_float32=target_ratio_np,
        positive_mask_uint8=mask_np.astype(np.uint8),
    )
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    preview_paths = {}
    for name, value, valid in (
        ("target_ratio", target_ratio_np, mask_np),
        ("predicted_ratio", pred_ratio, None),
        ("support", pred_support, None),
        ("confidence", pred_confidence, None),
    ):
        path = args.output_dir / f"{name}.png"
        save_map(path, value, valid)
        preview_paths[name] = path

    improvement = 1.0 - final["ratio_abs_rel"] / initial["ratio_abs_rel"]
    gates = {
        "relative_ratio_abs_rel_reduced_by_90_percent": improvement >= 0.90,
        "final_ratio_abs_rel_below_0_10": final["ratio_abs_rel"] < 0.10,
        "positive_support_mean_above_0_70": final["positive_support_mean"] > 0.70,
        "all_predictions_finite": bool(
            np.isfinite(pred_ratio).all()
            and np.isfinite(pred_support).all()
            and np.isfinite(pred_confidence).all()
        ),
        "strictly_positive_relative_increment": bool((pred_ratio > 0).all()),
        "confidence_in_open_unit_interval": bool(((pred_confidence > 0) & (pred_confidence < 1)).all()),
    }
    passed = all(gates.values())
    record = {
        "schema_version": "stage1.6-single-sample-overfit-v1",
        "status": "pass" if passed else "failed",
        "producer_machine_id": "linux5080",
        "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": visible,
        "sample": {
            "split": "train",
            "scene": args.scene,
            "source_frame": evidence_item["source_frame"],
            "positive_pixels_resized": int(valid_positive.sum().item()),
            "positive_class_prior": class_prior,
        },
        "inputs": {
            "evidence_record": portable(args.evidence_record),
            "evidence_record_sha256": sha256(args.evidence_record),
            "evidence_arrays_sha256": sha256(evidence_path),
            "base_record": portable(args.base_record),
            "base_record_sha256": sha256(args.base_record),
            "base_arrays_sha256": sha256(base_path),
            "model_inputs": ["source_rgb_uint8", "frozen_base_depth_z_float32"],
            "source_truth_depth_model_input": False,
            "source_truth_depth_usage": "positive_ratio_label_only",
        },
        "optimization": {
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": "cosine_to_one_percent",
            "support_weight": args.support_weight,
            "uncertainty_weight": args.uncertainty_weight,
            "base_channels": args.base_channels,
            "input_size_hw": list(size),
            "support_loss": "non_negative_positive_unlabeled_logistic_risk",
            "deterministic_algorithms": True,
        },
        "initial_metrics": initial,
        "final_metrics": final,
        "ratio_abs_rel_relative_improvement": improvement,
        "gates": gates,
        "artifacts": {
            "checkpoint": portable(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "arrays": portable(arrays_path),
            "arrays_sha256": sha256(arrays_path),
            "history": portable(history_path),
            "history_sha256": sha256(history_path),
            "previews": {
                name: {"path": portable(path), "sha256": sha256(path)}
                for name, path in preview_paths.items()
            },
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "quality_boundary": (
            "single train-sample memorization gate on relative hidden geometry; not a validation, "
            "HLP-GEO test, appearance, or P01 quality result"
        ),
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "initial": initial, "final": final, "gates": gates}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
