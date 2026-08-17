#!/usr/bin/env python3
"""Qualify Stage 1.8 train/val targets without weakening per-view geometry gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-record", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--maximum-depth-position-p99-relative", type=float, default=0.002)
    parser.add_argument("--minimum-train-targets", type=int, default=40)
    parser.add_argument("--minimum-val-targets", type=int, default=10)
    args = parser.parse_args()
    args.visibility_record = args.visibility_record.resolve(); args.record = args.record.resolve()
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite {args.record}")
    source = json.loads(args.visibility_record.read_text(encoding="utf-8"))
    eligible, rejected = [], []
    arrays_hashes_ok = True
    for view in source["views"]:
        reasons = []
        if not view["classification_complete"]:
            reasons.append("classification_incomplete")
        if view["target_depth_vs_position_p99_relative"] > args.maximum_depth_position_p99_relative:
            reasons.append("depth_position_p99_relative_exceeds_frozen_maximum")
        if view["occlusion_hidden_pixels"] <= 0:
            reasons.append("occlusion_hidden_empty")
        if view["outside_source_fov_pixels"] <= 0:
            reasons.append("outside_source_fov_empty")
        arrays_path = Path(view["arrays"])
        arrays_hash_ok = arrays_path.exists() and sha256(arrays_path) == view["arrays_sha256"]
        arrays_hashes_ok &= arrays_hash_ok
        item = {
            "split": view["split"], "scene": view["scene"], "camera": view["camera"],
            "source_frame": view["source_frame"], "target_frame": view["target_frame"], "side": view["side"],
            "yaw_deg": view["yaw_deg"], "arrays": view["arrays"], "arrays_sha256": view["arrays_sha256"],
            "arrays_hash_ok": arrays_hash_ok,
            "occlusion_hidden_pixels": view["occlusion_hidden_pixels"],
            "outside_source_fov_pixels": view["outside_source_fov_pixels"],
            "target_depth_vs_position_p99_relative": view["target_depth_vs_position_p99_relative"],
        }
        if reasons:
            item["reasons"] = reasons; rejected.append(item)
        else:
            eligible.append(item)
    train = sum(item["split"] == "train" for item in eligible)
    val = sum(item["split"] == "val" for item in eligible)
    gates = {
        "source_has_100_targets": len(source["views"]) == 100,
        "all_visibility_arrays_hashes_match": arrays_hashes_ok,
        "eligible_train_targets_meet_frozen_minimum": train >= args.minimum_train_targets,
        "eligible_val_targets_meet_frozen_minimum": val >= args.minimum_val_targets,
        "every_eligible_target_passes_strict_per_view_gates": all(
            item["target_depth_vs_position_p99_relative"] <= args.maximum_depth_position_p99_relative
            and item["occlusion_hidden_pixels"] > 0 and item["outside_source_fov_pixels"] > 0
            for item in eligible
        ),
    }
    status = "pass" if all(gates.values()) else "fail"
    record = {
        "schema_version": "stage1.8-target-training-pool-v1",
        "status": status,
        "producer_machine_id": "linux5080",
        "source_visibility_record": portable(args.visibility_record),
        "source_visibility_record_sha256": sha256(args.visibility_record),
        "frozen_qualification": {
            "maximum_depth_position_p99_relative": args.maximum_depth_position_p99_relative,
            "requires_classification_complete": True,
            "requires_occlusion_hidden_nonempty": True,
            "requires_outside_source_fov_nonempty": True,
            "minimum_train_targets": args.minimum_train_targets,
            "minimum_val_targets": args.minimum_val_targets,
        },
        "summary": {
            "source_targets": len(source["views"]), "eligible_targets": len(eligible),
            "eligible_train_targets": train, "eligible_val_targets": val,
            "rejected_targets": len(rejected),
        },
        "eligible_targets": eligible,
        "rejected_targets": rejected,
        "gates": gates,
        "quality_boundary": "geometry/data eligibility only; no model training or quality selection",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"], "gates": gates}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
