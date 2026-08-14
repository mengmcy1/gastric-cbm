#!/usr/bin/env python3
"""Freeze a scene-type-stratified subset from successful pose preflight."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def round_robin(rows, count):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_type"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda x: (x["angle_error_sum_deg"], x["scene"]))
    result, index = [], 0
    types = sorted(grouped)
    while len(result) < count:
        progressed = False
        for scene_type in types:
            if index < len(grouped[scene_type]):
                result.append(grouped[scene_type][index]); progressed = True
                if len(result) == count:
                    break
        if not progressed:
            break
        index += 1
    if len(result) != count:
        raise RuntimeError(f"only {len(result)} qualifying rows for requested {count}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-config", type=Path, required=True)
    parser.add_argument("--preflight-record", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    for name in ("preflight_config", "preflight_record", "output", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite output or record")
    preflight = json.loads(args.preflight_config.read_text())
    audit = json.loads(args.preflight_record.read_text())
    successful = preflight["triplets"]
    train = round_robin([x for x in successful if x["split"] == "train"], args.train_count)
    validation = round_robin(
        [x for x in successful if x["split"] == "val"], args.validation_count
    )
    selected = train + validation
    held_out = set(preflight["held_out_test_scenes"])
    scenes = {x["scene"] for x in selected}
    gates = {
        "counts_met": len(train) == args.train_count and len(validation) == args.validation_count,
        "scenes_unique": len(scenes) == len(selected),
        "held_out_overlap_zero": not scenes.intersection(held_out),
        "all_pose_gates_pass": all(
            x["angle_error_sum_deg"] <= 2 * preflight["target_motion"]["maximum_endpoint_error_deg"]
            and min(x["left_translation_m"], x["right_translation_m"]) >= 0.1
            for x in selected
        ),
    }
    output = {
        "schema_version": "stage1.6-hlp-train-03-triplets-v1",
        "status": "pass" if all(gates.values()) else "failed",
        "source_pose_preflight": str(args.preflight_config.relative_to(Path.cwd())),
        "source_pose_preflight_sha256": hashlib.sha256(args.preflight_config.read_bytes()).hexdigest(),
        "target_motion": preflight["target_motion"],
        "selection_policy": {
            "objective": "round-robin scene type coverage, then minimum angle error",
            "train_count": args.train_count,
            "validation_count": args.validation_count,
            "one_triplet_per_scene": True,
        },
        "held_out_test_scenes": sorted(held_out),
        "triplets": selected,
    }
    record = {
        "schema_version": "stage1.6-hlp-train-03-triplet-finalize-v1",
        "status": output["status"],
        "producer_machine_id": "linux5080",
        "preflight_status": audit["status"],
        "preflight_available": {
            "train": sum(x["split"] == "train" for x in successful),
            "validation": sum(x["split"] == "val" for x in successful),
        },
        "selected": {
            "train": len(train),
            "validation": len(validation),
            "train_scene_types": dict(Counter(x["scene_type"] for x in train)),
            "validation_scene_types": dict(Counter(x["scene_type"] for x in validation)),
        },
        "gates": gates,
        "quality_boundary": "pose-qualified scene-diverse subset only; RGB-D and labels not fetched",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.record.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    record["output_config"] = str(args.output.relative_to(Path.cwd()))
    record["output_config_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": record["status"], "selected": record["selected"], "gates": gates}, indent=2))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
