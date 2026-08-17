#!/usr/bin/env python3
"""Calibrate Stage 1.8 target-view RGB-D region evaluation with an exact oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
    from adaptive3dgs import TargetViewPrediction, evaluate_target_view_prediction
    from adaptive3dgs.datasets import load_hlp_geo_samples

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError("Stage 1.8 oracle config must have status=frozen")
    samples = load_hlp_geo_samples(
        args.config,
        args.dataset_root,
        args.camera_parameters,
        args.visibility_root,
        include_source_depth=False,
    )
    results = []
    gates: dict[str, bool] = {
        "two_samples": len(samples) == 2,
        "four_targets": sum(len(sample.targets) for sample in samples) == 4,
        "provider_context_has_no_source_depth_truth": all(
            sample.context.depth_z_float32 is None for sample in samples
        ),
    }
    for sample in samples:
        split = next(value["split"] for value in config["triplets"] if value["scene"] == sample.scene)
        for target in sample.targets:
            height, width = target.geometry.depth_z_float32.shape
            oracle = TargetViewPrediction(
                rgb_uint8=target.rgb_uint8.copy(),
                depth_z_float32=target.geometry.depth_z_float32.copy(),
                support_probability_float32=np.ones((height, width), dtype=np.float32),
                geometry_confidence_float32=np.ones((height, width), dtype=np.float32),
                appearance_confidence_float32=np.ones((height, width), dtype=np.float32),
            )
            evaluation = evaluate_target_view_prediction(target.geometry, target.rgb_uint8, oracle)
            hidden = evaluation["regions"]["occlusion_hidden"]
            outside = evaluation["regions"]["outside_source_fov"]
            key = target.target_id
            gates[f"{key}_novel_regions_nonempty"] = hidden["truth_pixels"] > 0 and outside["truth_pixels"] > 0
            for region_name, region in (("hidden", hidden), ("outside", outside)):
                gates[f"{key}_{region_name}_coverage_exact"] = region["coverage"] == 1.0
                gates[f"{key}_{region_name}_depth_exact"] = region["geometry"]["abs_rel"] <= 1e-7
                gates[f"{key}_{region_name}_rgb_exact"] = region["appearance"]["mae_0_255"] == 0.0
            results.append(
                {
                    "split": split,
                    "scene": sample.scene,
                    "target_id": target.target_id,
                    "side": target.side,
                    "yaw_deg": target.yaw_deg,
                    "evaluation": evaluation,
                }
            )
    gates = {name: bool(value) for name, value in gates.items()}
    status = "pass" if all(gates.values()) else "fail"
    visibility_record = json.loads(args.visibility_record.read_text(encoding="utf-8"))
    record = {
        "schema_version": "stage1.8-target-view-oracle-smoke-v2",
        "status": status,
        "producer_machine_id": "linux5080",
        "inputs": {
            "runner": portable(Path(__file__)),
            "runner_sha256": sha256(Path(__file__)),
            "evaluator": portable(args.adaptive3dgs_src / "adaptive3dgs" / "evaluation.py"),
            "evaluator_sha256": sha256(args.adaptive3dgs_src / "adaptive3dgs" / "evaluation.py"),
            "config": portable(args.config),
            "config_sha256": sha256(args.config),
            "camera_parameters": portable(args.camera_parameters),
            "camera_parameters_sha256": sha256(args.camera_parameters),
            "visibility_record": portable(args.visibility_record),
            "visibility_record_sha256": sha256(args.visibility_record),
        },
        "summary": {
            "samples": len(samples),
            "targets": len(results),
            "gates": len(gates),
            "all_gates_pass": all(gates.values()),
            "occlusion_hidden_pixels": sum(
                item["evaluation"]["regions"]["occlusion_hidden"]["truth_pixels"] for item in results
            ),
            "outside_source_fov_pixels": sum(
                item["evaluation"]["regions"]["outside_source_fov"]["truth_pixels"] for item in results
            ),
            "visibility_status": visibility_record["status"],
        },
        "targets": results,
        "gates": gates,
        "leakage_boundary": (
            "Exact target RGB-D is used only to calibrate evaluator output. ProviderContext contains no source "
            "depth truth, and this run performs no model training."
        ),
        "quality_boundary": (
            "Oracle pass proves mask/data/evaluator alignment only; it is not a learned-provider quality result."
        ),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"]}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
