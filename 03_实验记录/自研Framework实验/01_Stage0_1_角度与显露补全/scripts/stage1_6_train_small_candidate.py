#!/usr/bin/env python3
"""Train and validation-select the first 8/2 Stage 1.6 geometry candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from adaptive3dgs.datasets import load_hypersim_training_triplets
from adaptive3dgs.models import (
    OcclusionHiddenUNet,
    compute_hidden_geometry_loss,
    positive_verified_negative_logistic_loss,
)


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
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--evidence-record", type=Path, required=True)
    parser.add_argument("--base-record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--record-schema", default="stage1.6-hlp-train-01-small-candidate-v1")
    parser.add_argument("--triplet-config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--camera-parameters", type=Path)
    return parser.parse_args()


def resize(value: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    kwargs = {"size": size, "mode": mode}
    if mode == "bilinear":
        kwargs.update({"align_corners": False, "antialias": True})
    return F.interpolate(value, **kwargs)


class Sample:
    def __init__(self, evidence_item, base_item, size, device, geometry_triplet=None):
        self.split = evidence_item["split"]
        self.scene = evidence_item["scene"]
        self.source_frame = int(evidence_item["source_frame"])
        evidence_path, base_path = Path(evidence_item["arrays"]), Path(base_item["arrays"])
        if sha256(evidence_path) != evidence_item["arrays_sha256"]:
            raise RuntimeError(f"evidence hash mismatch: {self.scene}")
        if sha256(base_path) != base_item["arrays_sha256"]:
            raise RuntimeError(f"base hash mismatch: {self.scene}")
        self.input_hashes = {
            "evidence_arrays_sha256": evidence_item["arrays_sha256"],
            "base_arrays_sha256": base_item["arrays_sha256"],
        }
        with np.load(evidence_path, allow_pickle=False) as data:
            rgb = torch.from_numpy(data["source_rgb_uint8"].copy()).permute(2, 0, 1)[None].float() / 255
            source = torch.from_numpy(data["source_depth_z_label_only_float32"].copy())[None, None].float()
            hidden = torch.from_numpy(data["hidden_depth_z_float32"].copy())[None, None].float()
            positive = torch.from_numpy(data["positive_mask_uint8"].copy())[None, None].float()
        with np.load(base_path, allow_pickle=False) as data:
            base = torch.from_numpy(data["base_depth_z_float32"].copy())[None, None].float()
        self.rgb = resize(rgb, size, "bilinear").to(device)
        self.base = resize(base, size, "bilinear").to(device)
        self.source = resize(source, size, "nearest").to(device)
        self.hidden = resize(hidden, size, "nearest").to(device)
        positive = resize(positive, size, "nearest").bool().to(device)
        self.positive = positive & torch.isfinite(self.source) & torch.isfinite(self.hidden)
        self.positive &= (self.source > 0) & (self.hidden > self.source)
        self.target_ratio = torch.zeros_like(self.source)
        self.target_ratio[self.positive] = self.hidden[self.positive] / self.source[self.positive] - 1
        self.class_prior = float(self.positive.float().mean().item())
        if not 0 < self.class_prior < 1:
            raise RuntimeError(f"invalid positive prior: {self.scene}")
        self.free_space_targets = []
        if geometry_triplet is not None:
            native_h, native_w = source.shape[-2:]
            scale = torch.tensor(
                [[size[1] / native_w, 0, 0], [0, size[0] / native_h, 0], [0, 0, 1]],
                dtype=torch.float64,
            )
            source_k = scale @ torch.from_numpy(
                geometry_triplet.source_camera.intrinsics_3x3_float64.copy()
            )
            yy, xx = torch.meshgrid(
                torch.arange(size[0], dtype=torch.float64),
                torch.arange(size[1], dtype=torch.float64),
                indexing="ij",
            )
            pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).reshape(3, -1)
            self.source_rays = (torch.linalg.inv(source_k) @ pixels).to(device=device, dtype=torch.float32)
            self.camera_rays_xy = self.source_rays[:2].reshape(1, 2, size[0], size[1])
            source_c2w = torch.linalg.inv(
                torch.from_numpy(geometry_triplet.source_camera.world_to_camera_4x4_float64.copy())
            )
            for target in geometry_triplet.target_observations:
                target_depth = torch.from_numpy(target.depth_z_float32.copy())[None, None].float()
                target_depth = resize(target_depth, size, "nearest").to(device)
                target_k = scale @ torch.from_numpy(target.intrinsics_3x3_float64.copy())
                target_from_source = (
                    torch.from_numpy(target.world_to_camera_4x4_float64.copy()) @ source_c2w
                )
                self.free_space_targets.append(
                    (
                        target_depth,
                        target_k.to(device=device, dtype=torch.float32),
                        target_from_source.to(device=device, dtype=torch.float32),
                    )
                )

    def verified_free_space_negative(self, delta_ratio: torch.Tensor) -> torch.Tensor:
        if not self.free_space_targets:
            raise RuntimeError("target-view geometry was not loaded")
        with torch.no_grad():
            candidate_z = self.source * (1 + delta_ratio.detach())
            height, width = candidate_z.shape[-2:]
            z = candidate_z.reshape(-1)
            xyz = self.source_rays * z[None]
            homogeneous = torch.cat((xyz, torch.ones(1, xyz.shape[1], device=xyz.device)), dim=0)
            negative = torch.zeros_like(z, dtype=torch.bool)
            source_valid = torch.isfinite(z) & (z > 0)
            for target_depth, target_k, target_from_source in self.free_space_targets:
                target_xyz = (target_from_source @ homogeneous)[:3]
                target_z = target_xyz[2]
                projected = target_k @ target_xyz
                u = torch.round(projected[0] / projected[2]).long()
                v = torch.round(projected[1] / projected[2]).long()
                inside = (
                    source_valid
                    & torch.isfinite(target_z)
                    & (target_z > 0)
                    & (u >= 0)
                    & (u < width)
                    & (v >= 0)
                    & (v < height)
                )
                indices = torch.nonzero(inside, as_tuple=False).squeeze(1)
                if indices.numel() == 0:
                    continue
                observed = target_depth[0, 0, v[indices], u[indices]]
                tolerance = torch.maximum(
                    torch.full_like(observed, 0.02), observed.abs() * 0.005
                )
                free_space = torch.isfinite(observed) & (observed > 0)
                free_space &= target_z[indices] < observed - tolerance
                negative[indices[free_space]] = True
            negative = negative.reshape(1, 1, height, width)
            return negative & ~self.positive


def evaluate(
    model,
    samples,
    positive_selection_weight=0.25,
    unknown_selection_weight=0.05,
    support_supervision="positive_unlabeled",
):
    ratio_abs_sum = ratio_mae_sum = 0.0
    support_pos_sum = confidence_pos_sum = confidence_square_sum = 0.0
    support_unknown_sum = 0.0
    support_verified_negative_sum = 0.0
    positive_count = unknown_count = verified_negative_count = 0
    finite = strict = confidence_valid = True
    model.eval()
    with torch.no_grad():
        for sample in samples:
            output = model(sample.rgb, sample.base, getattr(sample, "camera_rays_xy", None))
            pred = output.delta_ratio[sample.positive]
            target = sample.target_ratio[sample.positive]
            count = int(target.numel())
            unknown = ~sample.positive
            unknown_n = int(unknown.sum().item())
            ratio_abs_sum += float(((pred - target).abs() / target.clamp_min(1e-4)).sum().item())
            ratio_mae_sum += float((pred - target).abs().sum().item())
            pos_support = output.support_probability[sample.positive]
            pos_conf = output.geometry_confidence[sample.positive]
            support_pos_sum += float(pos_support.sum().item())
            confidence_pos_sum += float(pos_conf.sum().item())
            confidence_square_sum += float(pos_conf.square().sum().item())
            support_unknown_sum += float(output.support_probability[unknown].sum().item())
            if support_supervision == "target_view_free_space":
                verified_negative = sample.verified_free_space_negative(output.delta_ratio)
                verified_n = int(verified_negative.sum().item())
                support_verified_negative_sum += float(
                    output.support_probability[verified_negative].sum().item()
                )
                verified_negative_count += verified_n
            positive_count += count
            unknown_count += unknown_n
            finite &= bool(
                torch.isfinite(output.delta_ratio).all()
                and torch.isfinite(output.support_probability).all()
                and torch.isfinite(output.geometry_confidence).all()
            )
            strict &= bool((output.delta_ratio > 0).all())
            confidence_valid &= bool(
                ((output.geometry_confidence > 0) & (output.geometry_confidence < 1)).all()
            )
    confidence_mean = confidence_pos_sum / positive_count
    confidence_variance = max(confidence_square_sum / positive_count - confidence_mean**2, 0.0)
    metrics = {
        "ratio_abs_rel": ratio_abs_sum / positive_count,
        "ratio_mae": ratio_mae_sum / positive_count,
        "positive_support_mean": support_pos_sum / positive_count,
        "unknown_support_mean": support_unknown_sum / unknown_count,
        "verified_negative_support_mean": (
            support_verified_negative_sum / verified_negative_count
            if verified_negative_count
            else None
        ),
        "positive_confidence_mean": confidence_mean,
        "positive_confidence_std": confidence_variance**0.5,
        "positive_pixels": positive_count,
        "unknown_pixels": unknown_count,
        "verified_negative_pixels": verified_negative_count,
        "all_finite": finite,
        "strictly_positive_relative_increment": strict,
        "confidence_in_open_unit_interval": confidence_valid,
    }
    reference_support = (
        metrics["verified_negative_support_mean"]
        if support_supervision == "target_view_free_space"
        else metrics["unknown_support_mean"]
    )
    metrics["support_reference_mean"] = reference_support
    metrics["selection_score"] = (
        metrics["ratio_abs_rel"]
        + positive_selection_weight * (1 - metrics["positive_support_mean"])
        + unknown_selection_weight * reference_support
    )
    return metrics


def save_gray(path: Path, value: np.ndarray) -> None:
    finite = np.isfinite(value)
    lo, hi = np.quantile(value[finite], [0.02, 0.98])
    normalized = np.clip((value - lo) / max(float(hi - lo), 1e-6), 0, 1)
    Image.fromarray(np.uint8(np.rint(normalized * 255))).save(path)


def main() -> int:
    args = parse_args()
    for name in ("training_config", "evidence_record", "base_record", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("triplet_config", "dataset_root", "camera_parameters"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with --physical-gpu-index")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = json.loads(args.training_config.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_run":
        raise RuntimeError("training config must be frozen_before_run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    evidence = json.loads(args.evidence_record.read_text(encoding="utf-8"))
    base = json.loads(args.base_record.read_text(encoding="utf-8"))
    support_supervision = str(config.get("support_supervision", "positive_unlabeled"))
    geometry_by_sample = {}
    if support_supervision == "target_view_free_space":
        if args.triplet_config is None or args.dataset_root is None or args.camera_parameters is None:
            raise RuntimeError("target_view_free_space requires triplet config, dataset root, and camera parameters")
        geometry_triplets = load_hypersim_training_triplets(
            args.triplet_config, args.dataset_root, args.camera_parameters
        )
        geometry_by_sample = {
            (item.scene, int(item.source_frame_index)): item for item in geometry_triplets
        }
    elif support_supervision != "positive_unlabeled":
        raise RuntimeError(f"unsupported support supervision: {support_supervision}")
    base_by_sample = {(item["scene"], int(item["source_frame"])): item for item in base["results"]}
    size = tuple(config["input_size_hw"])
    samples = [
        Sample(
            item,
            base_by_sample[(item["scene"], int(item["source_frame"]))],
            size,
            device,
            geometry_by_sample.get((item["scene"], int(item["source_frame"]))),
        )
        for item in evidence["results"]
    ]
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "val"]
    expected_train = int(config.get("train_samples", config["train_scenes"]))
    expected_validation = int(config.get("validation_samples", config["validation_scenes"]))
    if len(train) != expected_train or len(validation) != expected_validation:
        raise RuntimeError("train/validation count does not match frozen config")

    model = OcclusionHiddenUNet(base_channels=config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    batch_size = int(config.get("batch_size", 1))
    if batch_size < 1:
        raise RuntimeError("batch_size must be positive")
    total_steps = config["epochs"] * math.ceil(len(train) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config["learning_rate"] / 100
    )
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    positive_selection_weight = float(config.get("positive_support_selection_weight", 0.25))
    unknown_selection_weight = float(config.get("unknown_support_selection_weight", 0.05))
    initial_train = evaluate(
        model, train, positive_selection_weight, unknown_selection_weight, support_supervision
    )
    initial_validation = evaluate(
        model, validation, positive_selection_weight, unknown_selection_weight, support_supervision
    )
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    best_train = best_validation = None
    history = []
    rng = random.Random(args.seed)
    for epoch in range(1, config["epochs"] + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        model.train()
        epoch_losses = []
        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_total = 0.0
            for index in batch_indices:
                sample = train[index]
                output = model(sample.rgb, sample.base, getattr(sample, "camera_rays_xy", None))
                geometry_loss = compute_hidden_geometry_loss(
                    output,
                    sample.source,
                    sample.hidden,
                    sample.positive,
                    class_prior=sample.class_prior,
                    support_weight=(config["support_weight"] if support_supervision == "positive_unlabeled" else 0),
                    uncertainty_weight=config["uncertainty_weight"],
                )
                if support_supervision == "target_view_free_space":
                    verified_negative = sample.verified_free_space_negative(output.delta_ratio)
                    support_loss = positive_verified_negative_logistic_loss(
                        output.support_logits, sample.positive, verified_negative
                    )
                    total_loss = geometry_loss.total + config["support_weight"] * support_loss
                else:
                    support_loss = geometry_loss.support_pu
                    total_loss = geometry_loss.total
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}, scene {sample.scene}")
                (total_loss / len(batch_indices)).backward()
                batch_total += float(total_loss.detach().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(batch_total / len(batch_indices))
        if epoch % config["validation_interval_epochs"] == 0 or epoch == config["epochs"]:
            train_metrics = evaluate(
                model, train, positive_selection_weight, unknown_selection_weight, support_supervision
            )
            validation_metrics = evaluate(
                model, validation, positive_selection_weight, unknown_selection_weight, support_supervision
            )
            history.append(
                {
                    "epoch": epoch,
                    "mean_train_step_loss": float(np.mean(epoch_losses)),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            if validation_metrics["selection_score"] < best_score:
                best_score = validation_metrics["selection_score"]
                best_epoch = epoch
                best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                best_train, best_validation = train_metrics, validation_metrics
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()

    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.json"
    torch.save(
        {
            "model": best_state,
            "architecture": {"base_channels": config["base_channels"], "minimum_delta_ratio": 1e-4},
            "input_size_hw": list(size),
            "best_epoch": best_epoch,
            "selection_score": best_score,
        },
        checkpoint_path,
    )
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    predictions = []
    with torch.no_grad():
        for sample in samples:
            output = model(sample.rgb, sample.base, getattr(sample, "camera_rays_xy", None))
            sample_id = f"{sample.split}-{sample.scene}-{sample.source_frame:04d}"
            arrays_path = args.output_dir / f"{sample_id}-prediction.npz"
            preview_path = args.output_dir / f"{sample_id}-support.png"
            np.savez_compressed(
                arrays_path,
                predicted_delta_ratio_float32=output.delta_ratio[0, 0].cpu().numpy().astype(np.float32),
                predicted_support_probability_float32=output.support_probability[0, 0].cpu().numpy().astype(np.float32),
                predicted_geometry_confidence_float32=output.geometry_confidence[0, 0].cpu().numpy().astype(np.float32),
                target_delta_ratio_positive_only_float32=sample.target_ratio[0, 0].cpu().numpy().astype(np.float32),
                positive_mask_uint8=sample.positive[0, 0].cpu().numpy().astype(np.uint8),
            )
            save_gray(preview_path, output.support_probability[0, 0].cpu().numpy())
            predictions.append(
                {
                    "split": sample.split,
                    "scene": sample.scene,
                    "source_frame": sample.source_frame,
                    "positive_class_prior": sample.class_prior,
                    **sample.input_hashes,
                    "arrays": portable(arrays_path),
                    "arrays_sha256": sha256(arrays_path),
                    "support_preview": portable(preview_path),
                    "support_preview_sha256": sha256(preview_path),
                }
            )

    gates_config = config["gates"]
    score_improvement = 1 - best_validation["selection_score"] / initial_validation["selection_score"]
    train_improvement = 1 - best_train["ratio_abs_rel"] / initial_train["ratio_abs_rel"]
    gates = {
        "validation_selection_score_improved_by_frozen_minimum": score_improvement
        >= gates_config["best_validation_selection_score_relative_improvement_minimum"],
        "validation_ratio_abs_rel_below_initial": best_validation["ratio_abs_rel"]
        < initial_validation["ratio_abs_rel"],
        "validation_positive_minus_reference_support_above_frozen_minimum": (
            best_validation["positive_support_mean"] - best_validation["support_reference_mean"]
            >= gates_config["best_validation_positive_minus_unknown_support_minimum"]
        ),
        "train_ratio_abs_rel_improved_by_frozen_minimum": train_improvement
        >= gates_config["best_train_ratio_abs_rel_relative_improvement_minimum"],
        "all_best_outputs_finite": best_train["all_finite"] and best_validation["all_finite"],
        "all_best_relative_increments_strictly_positive": best_train["strictly_positive_relative_increment"]
        and best_validation["strictly_positive_relative_increment"],
    }
    passed = all(gates.values())
    record = {
        "schema_version": args.record_schema,
        "status": "pass_candidate_frozen" if passed else "failed",
        "producer_machine_id": "linux5080",
        "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": visible,
        "inputs": {
            "training_config": portable(args.training_config),
            "training_config_sha256": sha256(args.training_config),
            "evidence_record": portable(args.evidence_record),
            "evidence_record_sha256": sha256(args.evidence_record),
            "base_record": portable(args.base_record),
            "base_record_sha256": sha256(args.base_record),
            "model_inputs": ["source_rgb_uint8", "frozen_base_depth_z_float32"],
            "source_truth_depth_model_input": False,
            "held_out_hlp_geo_test_used": False,
            "support_supervision": support_supervision,
            "gradient_accumulation_batch_size": batch_size,
            "target_view_geometry_source_truth_usage": "label_side_free_space_test_only",
            "target_view_geometry_uses_source_truth_as_model_input": False,
        },
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "best_epoch": best_epoch,
        "best_train": best_train,
        "best_validation": best_validation,
        "validation_selection_score_relative_improvement": score_improvement,
        "train_ratio_abs_rel_relative_improvement": train_improvement,
        "gates": gates,
        "predictions": predictions,
        "artifacts": {
            "checkpoint": portable(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "history": portable(history_path),
            "history_sha256": sha256(history_path),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "quality_boundary": (
            f"{len(train)}-train/{len(validation)}-validation geometry candidate; "
            "selected without HLP-GEO test or P01; "
            "no hidden appearance or target-view render conclusion"
        ),
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": record["status"],
                "best_epoch": best_epoch,
                "initial_validation": initial_validation,
                "best_validation": best_validation,
                "gates": gates,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
