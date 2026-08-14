#!/usr/bin/env python3
"""Freeze a scene-diverse Hypersim train/validation pose-preflight pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--trajectories-csv", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--train-candidates", type=int, default=60)
    parser.add_argument("--validation-candidates", type=int, default=20)
    parser.add_argument("--minimum-public-frames", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def round_robin_scene_types(rows, count):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_type"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda value: (value["scene"], value["camera"]))
    selected = []
    types = sorted(grouped)
    index = 0
    while len(selected) < count:
        progress = False
        for scene_type in types:
            values = grouped[scene_type]
            if index < len(values):
                selected.append(values[index])
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            break
        index += 1
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} eligible candidates for requested {count}")
    return selected


def main():
    args = parse_args()
    for name in ("split_csv", "trajectories_csv", "camera_parameters", "output", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite output or record")
    frame_counts = Counter()
    camera_splits = defaultdict(set)
    with args.split_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["scene_name"], row["camera_name"])
            camera_splits[key].add(row["split_partition_name"])
            if row["included_in_public_release"] == "True":
                frame_counts[key] += 1
    trajectory_types = {}
    with args.trajectories_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parts = row["Animation"].split("_")
            if len(parts) < 5:
                continue
            scene, camera = "_".join(parts[:3]), "_".join(parts[3:])
            trajectory_types[(scene, camera)] = row["Scene type"] or "Unknown"
    held_out = {"ai_001_010", "ai_005_001", "ai_008_005"}
    by_split = {"train": [], "val": []}
    seen_scenes = set()
    for (scene, camera), public_frames in sorted(frame_counts.items()):
        splits = camera_splits[(scene, camera)]
        if len(splits) != 1:
            continue
        split = next(iter(splits))
        if split not in by_split or scene in held_out or public_frames < args.minimum_public_frames:
            continue
        # One camera trajectory per scene; cam_00 wins through lexical ordering.
        if scene in seen_scenes:
            continue
        seen_scenes.add(scene)
        by_split[split].append(
            {
                "scene": scene,
                "camera": camera,
                "scene_type": trajectory_types.get((scene, camera), "Unknown"),
                "public_frames": public_frames,
            }
        )
    train = round_robin_scene_types(by_split["train"], args.train_candidates)
    validation = round_robin_scene_types(by_split["val"], args.validation_candidates)
    output = {
        "schema_version": "stage1.6-hlp-train-03-candidates-v1",
        "status": "frozen_pose_preflight_pending",
        "date": "2026-08-14",
        "official_split_source": {"path": str(args.split_csv.relative_to(Path.cwd())), "sha256": sha256(args.split_csv)},
        "camera_trajectory_metadata": {"path": str(args.trajectories_csv.relative_to(Path.cwd())), "sha256": sha256(args.trajectories_csv)},
        "camera_parameters": {"path": str(args.camera_parameters.relative_to(Path.cwd())), "sha256": sha256(args.camera_parameters)},
        "held_out_test_scenes": sorted(held_out),
        "selection_scope": "scene-diverse pose preflight after HLP-TRAIN-02 same-scene multi-source generalization failure",
        "target_motion": {
            "total_yaw_deg": 30.0,
            "left_endpoint_deg": -15.0,
            "right_endpoint_deg": 15.0,
            "maximum_endpoint_error_deg": 3.0,
            "maximum_triplets_per_scene": 1
        },
        "train_candidates": train,
        "validation_candidates": validation,
        "leakage_rules": {
            "official_test_split_used_for_training_or_selection": False,
            "hlp_geo_01_test_scenes_used_for_training_or_selection": False,
            "p01_used_for_training_or_selection": False,
            "scene_level_train_validation_separation_preserved": True
        }
    }
    candidate_scenes = {x["scene"] for x in train + validation}
    gates = {
        "requested_counts_met": len(train) == args.train_candidates and len(validation) == args.validation_candidates,
        "all_scenes_unique": len(candidate_scenes) == len(train) + len(validation),
        "held_out_overlap_zero": not candidate_scenes.intersection(held_out),
        "minimum_public_frames_met": all(x["public_frames"] >= args.minimum_public_frames for x in train + validation),
    }
    record = {
        "schema_version": "stage1.6-hlp-train-03-candidate-build-v1",
        "status": "pass" if all(gates.values()) else "failed",
        "producer_machine_id": "linux5080",
        "summary": {
            "train_candidates": len(train),
            "validation_candidates": len(validation),
            "train_scene_types": dict(Counter(x["scene_type"] for x in train)),
            "validation_scene_types": dict(Counter(x["scene_type"] for x in validation)),
            "gates": gates,
        },
        "quality_boundary": "metadata-only candidate freeze; real camera pose angle qualification remains pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    record["output_config"] = str(args.output.relative_to(Path.cwd()))
    record["output_config_sha256"] = sha256(args.output)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": record["status"], "summary": record["summary"]}, indent=2))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
