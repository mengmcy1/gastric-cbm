#!/usr/bin/env python3
"""Build HLP-TRAIN-01 source-anchored positive hidden evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from adaptive3dgs import build_source_hidden_evidence
from adaptive3dgs.datasets import load_hypersim_training_triplets


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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--relative-behind-margin", type=float, default=0.01)
    parser.add_argument("--absolute-behind-margin", type=float, default=0.01)
    parser.add_argument("--minimum-positive-fraction", type=float, default=0.001)
    parser.add_argument("--expected-triplets", type=int, default=10)
    parser.add_argument("--record-schema", default="stage1.6-hlp-train-01-hidden-evidence-smoke-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "dataset_root", "camera_parameters", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    if not 0 < args.minimum_positive_fraction < 1:
        raise ValueError("minimum positive fraction must be in (0,1)")
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    triplets = load_hypersim_training_triplets(args.config, args.dataset_root, args.camera_parameters)
    results = []
    gates: dict[str, bool] = {}
    for triplet in triplets:
        evidence = build_source_hidden_evidence(
            triplet.source_depth_z_label_only_float32,
            triplet.source_camera.intrinsics_3x3_float64,
            triplet.source_camera.world_to_camera_4x4_float64,
            triplet.target_observations,
            relative_behind_margin=args.relative_behind_margin,
            absolute_behind_margin=args.absolute_behind_margin,
        )
        positive = evidence.positive_mask_uint8.astype(bool)
        source_depth = triplet.source_depth_z_label_only_float32
        positive_count = int(positive.sum())
        positive_fraction = float(positive.mean())
        delta = evidence.depth_z_float32[positive] - source_depth[positive]
        strict_absolute = bool(np.all(delta > args.absolute_behind_margin)) if positive_count else False
        strict_relative = bool(
            np.all(evidence.depth_z_float32[positive] > source_depth[positive] * (1 + args.relative_behind_margin))
        ) if positive_count else False
        unknown_count = int((~positive).sum())
        multi_observed = int((evidence.observation_count_uint16 >= 2).sum())
        scene_id = f"{triplet.scene}-{triplet.camera_name}-{triplet.source_frame_index:04d}"
        arrays_path = args.output_dir / f"{scene_id}-hidden-evidence.npz"
        np.savez_compressed(
            arrays_path,
            source_rgb_uint8=triplet.source_rgb_uint8,
            source_depth_z_label_only_float32=source_depth,
            hidden_depth_z_float32=evidence.depth_z_float32,
            hidden_rgb_uint8=evidence.rgb_uint8,
            positive_mask_uint8=evidence.positive_mask_uint8,
            observation_count_uint16=evidence.observation_count_uint16,
            depth_spread_float32=evidence.depth_spread_float32,
            source_view_coverage_uint8=evidence.source_view_coverage_uint8,
        )
        scene_gates = {
            "has_positive_evidence": positive_count > 0,
            "positive_fraction_above_minimum": positive_fraction >= args.minimum_positive_fraction,
            "all_positive_depths_strictly_behind_absolute_margin": strict_absolute,
            "all_positive_depths_strictly_behind_relative_margin": strict_relative,
            "unknown_pixels_are_not_serialized_as_negative_labels": unknown_count == int((~positive).sum()),
        }
        for name, value in scene_gates.items():
            gates[f"{scene_id}:{name}"] = value
        results.append(
            {
                "split": triplet.split,
                "scene": triplet.scene,
                "camera": triplet.camera_name,
                "source_frame": triplet.source_frame_index,
                "positive_pixels": positive_count,
                "positive_fraction": positive_fraction,
                "source_view_coverage_pixels": int(evidence.source_view_coverage_uint8.sum()),
                "multi_observed_positive_pixels": multi_observed,
                "hidden_delta_z_m": {
                    "minimum": float(delta.min()) if positive_count else None,
                    "median": float(np.median(delta)) if positive_count else None,
                    "p99": float(np.quantile(delta, 0.99)) if positive_count else None,
                },
                "unknown_pixels": unknown_count,
                "arrays": portable(arrays_path),
                "arrays_sha256": sha256(arrays_path),
                "gates": scene_gates,
            }
        )
    all_gates_pass = all(gates.values())
    record = {
        "schema_version": args.record_schema,
        "status": "pass" if all_gates_pass and len(triplets) == args.expected_triplets else "failed",
        "producer_machine_id": "linux5080",
        "inputs": {
            "config": portable(args.config),
            "config_sha256": sha256(args.config),
            "dataset_root": portable(args.dataset_root),
            "camera_parameters": portable(args.camera_parameters),
            "camera_parameters_sha256": sha256(args.camera_parameters),
        },
        "label_policy": {
            "relative_behind_margin": args.relative_behind_margin,
            "absolute_behind_margin_m": args.absolute_behind_margin,
            "minimum_positive_fraction": args.minimum_positive_fraction,
            "unobserved_source_rays": "unknown_not_negative",
            "source_truth_depth_model_input": False,
            "source_truth_depth_usage": "offline_label_construction_only",
        },
        "results": results,
        "gates": gates,
        "summary": {
            "triplets": len(triplets),
            "expected_triplets": args.expected_triplets,
            "train_triplets": sum(value.split == "train" for value in triplets),
            "validation_triplets": sum(value.split == "val" for value in triplets),
            "positive_pixels": sum(int(value["positive_pixels"]) for value in results),
            "minimum_positive_fraction": min(float(value["positive_fraction"]) for value in results),
            "median_positive_fraction": float(np.median([value["positive_fraction"] for value in results])),
            "gate_count": len(gates),
            "all_gates_pass": all_gates_pass,
            "runtime_seconds": time.perf_counter() - started,
        },
        "quality_boundary": "positive hidden evidence only; no dense support negatives, model training, or test-set result",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "summary": record["summary"]}, indent=2))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
