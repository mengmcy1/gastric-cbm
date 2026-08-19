#!/usr/bin/env python3
"""Freeze the Stage 1.9-3 8-train-scene/2-validation-scene directed-pair pool."""

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
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--qualified-pairs", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    args.training_config = args.training_config.resolve()
    args.qualified_pairs = args.qualified_pairs.resolve()
    args.record = args.record.resolve()
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite {args.record}")
    config = json.loads(args.training_config.read_text(encoding="utf-8"))
    pool = json.loads(args.qualified_pairs.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_run" or pool.get("status") != "pass":
        raise RuntimeError("training config must be frozen and qualified pool must pass")
    train_scenes = tuple(config["selection"]["train_scenes"])
    validation_scenes = tuple(config["selection"]["validation_scenes"])
    if len(train_scenes) != len(set(train_scenes)) or len(validation_scenes) != len(set(validation_scenes)):
        raise RuntimeError("selected scene lists must be unique")
    if set(train_scenes) & set(validation_scenes):
        raise RuntimeError("train and validation scenes overlap")
    selected = [
        pair for pair in pool["pairs"]
        if (pair["split"] == "train" and pair["scene"] in train_scenes)
        or (pair["split"] == "val" and pair["scene"] in validation_scenes)
    ]
    train = [pair for pair in selected if pair["split"] == "train"]
    validation = [pair for pair in selected if pair["split"] == "val"]
    gates = {
        "exact_train_scene_count": len({p["scene"] for p in train}) == config["expected_train_scenes"],
        "exact_validation_scene_count": len({p["scene"] for p in validation}) == config["expected_validation_scenes"],
        "exact_train_target_count": len(train) == config["expected_train_targets"],
        "exact_validation_target_count": len(validation) == config["expected_validation_targets"],
        "all_selected_pairs_qualified": all(bool(p["eligible"]) for p in selected),
        "both_directions_in_train": {p["source_role"] == "center" for p in train} == {True, False},
        "both_directions_in_validation": {p["source_role"] == "center" for p in validation} == {True, False},
        "all_novel_regions_nonempty": all(
            p["occlusion_hidden_pixels"] > 0 and p["outside_source_fov_pixels"] > 0 for p in selected
        ),
    }
    record = {
        "schema_version": "stage1.9-small-candidate-pool-v1",
        "status": "pass" if all(gates.values()) else "failed",
        "producer_machine_id": "linux5080",
        "inputs": {
            "training_config": portable(args.training_config),
            "training_config_sha256": sha256(args.training_config),
            "qualified_pairs": portable(args.qualified_pairs),
            "qualified_pairs_sha256": sha256(args.qualified_pairs),
        },
        "selection": {
            "train_scenes": list(train_scenes), "validation_scenes": list(validation_scenes),
            "policy": "all strictly qualified directed pairs from each explicitly frozen independent scene",
        },
        "summary": {
            "train_scenes": len({p["scene"] for p in train}),
            "validation_scenes": len({p["scene"] for p in validation}),
            "train_targets": len(train), "validation_targets": len(validation),
            "train_scene_types": sorted({p["scene_type"] for p in train}),
            "validation_scene_types": sorted({p["scene_type"] for p in validation}),
            "occlusion_hidden_pixels": sum(p["occlusion_hidden_pixels"] for p in selected),
            "outside_source_fov_pixels": sum(p["outside_source_fov_pixels"] for p in selected),
        },
        "gates": gates, "pairs": selected,
        "quality_boundary": "small-candidate train/validation pool only; held-out test and P01 excluded",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "summary": record["summary"], "gates": gates}, indent=2))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
