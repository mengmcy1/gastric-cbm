#!/usr/bin/env python3
"""Validate the Adaptive3DGS HLP-GEO-01 dataset/camera adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


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
    parser.add_argument("--adaptive3dgs-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--visibility-root", type=Path, required=True)
    parser.add_argument("--visibility-record", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(world_to_camera)[:3, 3]


def main() -> int:
    args = parse_args()
    for name in (
        "adaptive3dgs_src",
        "config",
        "dataset_root",
        "camera_parameters",
        "visibility_root",
        "visibility_record",
        "record",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    sys.path.insert(0, str(args.adaptive3dgs_src))
    from adaptive3dgs.datasets import load_hlp_geo_samples

    samples = load_hlp_geo_samples(
        args.config,
        args.dataset_root,
        args.camera_parameters,
        args.visibility_root,
        include_source_depth=False,
    )
    target_results = []
    gates: dict[str, bool] = {
        "three_samples": len(samples) == 3,
        "six_targets": sum(len(value.targets) for value in samples) == 6,
        "rgb_only_context_has_no_source_depth_truth": all(value.context.depth_z_float32 is None for value in samples),
        "sample_ids_unique": len({value.context.sample_id for value in samples}) == len(samples),
    }
    total_hidden = 0
    total_outside = 0
    target_ids: set[str] = set()
    for sample in samples:
        source_center = camera_center(sample.context.world_to_camera_4x4_float64)
        gates[f"{sample.scene}_source_rgb_uint8_hwc"] = (
            sample.context.rgb_uint8.dtype == np.uint8
            and sample.context.rgb_uint8.shape == (768, 1024, 3)
        )
        gates[f"{sample.scene}_source_camera_finite"] = (
            np.isfinite(sample.context.intrinsics_3x3_float64).all()
            and np.isfinite(sample.context.world_to_camera_4x4_float64).all()
        )
        for target in sample.targets:
            geometry = target.geometry
            hidden = geometry.occlusion_hidden_mask
            outside = geometry.outside_source_fov_mask
            observed = geometry.observed_from_source_mask
            union = hidden | outside | observed
            hidden_count = int(hidden.sum())
            outside_count = int(outside.sum())
            total_hidden += hidden_count
            total_outside += outside_count
            measured_translation = float(np.linalg.norm(camera_center(target.camera.world_to_camera_4x4_float64) - source_center))
            translation_error = abs(measured_translation - target.translation_m)
            rotation = np.linalg.inv(target.camera.world_to_camera_4x4_float64)[:3, :3]
            gates[f"{target.target_id}_unique"] = target.target_id not in target_ids
            target_ids.add(target.target_id)
            gates[f"{target.target_id}_rgb_uint8_hwc"] = (
                target.rgb_uint8.dtype == np.uint8 and target.rgb_uint8.shape == (768, 1024, 3)
            )
            gates[f"{target.target_id}_masks_disjoint"] = not (
                np.any(hidden & outside) or np.any(hidden & observed) or np.any(outside & observed)
            )
            gates[f"{target.target_id}_hidden_nonempty"] = hidden_count > 0
            gates[f"{target.target_id}_outside_nonempty"] = outside_count > 0
            gates[f"{target.target_id}_truth_depth_positive"] = bool(
                np.isfinite(geometry.depth_z_float32[union]).all()
                and (geometry.depth_z_float32[union] > 0).all()
            )
            gates[f"{target.target_id}_translation_matches_config"] = translation_error <= 1e-6
            gates[f"{target.target_id}_proper_rotation"] = abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-8
            target_results.append(
                {
                    "target_id": target.target_id,
                    "yaw_deg": target.yaw_deg,
                    "configured_translation_m": target.translation_m,
                    "measured_translation_m": measured_translation,
                    "translation_abs_error_m": translation_error,
                    "occlusion_hidden_pixels": hidden_count,
                    "outside_source_fov_pixels": outside_count,
                }
            )
    visibility_record = json.loads(args.visibility_record.read_text(encoding="utf-8"))
    expected_hidden = sum(int(value["occlusion_hidden_pixels"]) for value in visibility_record["views"])
    expected_outside = sum(int(value["outside_source_fov_pixels"]) for value in visibility_record["views"])
    gates["hidden_total_matches_visibility_record"] = total_hidden == expected_hidden
    gates["outside_total_matches_visibility_record"] = total_outside == expected_outside
    gates = {key: bool(value) for key, value in gates.items()}
    status = "pass" if all(gates.values()) else "fail"
    record = {
        "schema_version": "stage1.5-hlp-geo-dataset-adapter-smoke-v1",
        "status": status,
        "producer_machine_id": "linux5080",
        "inputs": {
            "config": portable(args.config),
            "config_sha256": sha256(args.config),
            "camera_parameters": portable(args.camera_parameters),
            "camera_parameters_sha256": sha256(args.camera_parameters),
            "visibility_record": portable(args.visibility_record),
            "visibility_record_sha256": sha256(args.visibility_record),
        },
        "summary": {
            "samples": len(samples),
            "targets": len(target_results),
            "occlusion_hidden_pixels": total_hidden,
            "outside_source_fov_pixels": total_outside,
            "gate_count": len(gates),
            "all_gates_pass": all(gates.values()),
        },
        "targets": target_results,
        "gates": gates,
        "leakage_boundary": "include_source_depth=False; ProviderContext.depth_z_float32 is None for every sample",
        "quality_boundary": "dataset/camera/truth assembly only; no provider inference or quality claim",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"]}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
