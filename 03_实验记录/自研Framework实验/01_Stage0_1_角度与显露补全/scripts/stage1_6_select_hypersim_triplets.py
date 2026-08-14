#!/usr/bin/env python3
"""Select real-pose ±15° Hypersim triplets for HLP-TRAIN-01."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


HYPERSIM_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def load_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["dataset"][:]


def scene_scale(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = {row[0]: row[1] for row in csv.reader(stream) if len(row) >= 2}
    value = float(values["meters_per_asset_unit"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid scene scale: {path}")
    return value


def public_frames(path: Path, scene: str, camera: str, expected_split: str) -> set[int]:
    frames = set()
    observed_splits = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["scene_name"] != scene or row["camera_name"] != camera:
                continue
            if row["included_in_public_release"] == "True":
                frames.add(int(row["frame_id"]))
                observed_splits.add(row["split_partition_name"])
    if observed_splits != {expected_split}:
        raise ValueError(f"{scene}/{camera}: expected split {expected_split}, got {sorted(observed_splits)}")
    return frames


def relative_yaw_deg(source_rotation: np.ndarray, target_rotation: np.ndarray) -> float:
    source_opencv = source_rotation @ HYPERSIM_TO_OPENCV
    target_opencv = target_rotation @ HYPERSIM_TO_OPENCV
    relative = source_opencv.T @ target_opencv
    return float(np.degrees(np.arctan2(relative[0, 2], relative[2, 2])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("candidates", "pose_root", "output_config", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_config.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite triplet config or record")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    motion = candidates["target_motion"]
    split_path = Path.cwd() / candidates["official_split_source"]["path"]
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != candidates["official_split_source"]["sha256"]:
        raise ValueError("official split source SHA256 mismatch")
    target_left = float(motion["left_endpoint_deg"])
    target_right = float(motion["right_endpoint_deg"])
    max_error = float(motion["maximum_endpoint_error_deg"])
    maximum_triplets = int(motion.get("maximum_triplets_per_scene", 1))
    if maximum_triplets < 1:
        raise ValueError("maximum_triplets_per_scene must be positive")
    minimum_translation = 0.1
    selected: list[dict[str, object]] = []
    scene_diagnostics: list[dict[str, object]] = []

    grouped = [("train", value) for value in candidates["train_candidates"]] + [
        ("val", value) for value in candidates["validation_candidates"]
    ]
    for split, candidate in grouped:
        scene, camera = str(candidate["scene"]), str(candidate["camera"])
        detail = args.pose_root / scene / "_detail" / camera
        frames = load_hdf5(detail / "camera_keyframe_frame_indices.hdf5").astype(int)
        rotations = load_hdf5(detail / "camera_keyframe_orientations.hdf5").astype(np.float64)
        positions = load_hdf5(detail / "camera_keyframe_positions.hdf5").astype(np.float64)
        if not (len(frames) == len(rotations) == len(positions)):
            raise ValueError(f"{scene}: pose array lengths differ")
        allowed = public_frames(split_path, scene, camera, split)
        indices = [index for index, frame in enumerate(frames) if int(frame) in allowed]
        scale = scene_scale(args.pose_root / scene / "_detail" / "metadata_scene.csv")
        qualifying: list[tuple[float, int, dict[str, tuple[float, int, float, float]]]] = []
        for source_index in indices:
            choices: dict[str, tuple[float, int, float, float]] = {}
            for side, target_angle in (("left", target_left), ("right", target_right)):
                side_best = None
                for target_index in indices:
                    if target_index == source_index:
                        continue
                    translation = float(np.linalg.norm(positions[target_index] - positions[source_index]) * scale)
                    if translation < minimum_translation:
                        continue
                    yaw = relative_yaw_deg(rotations[source_index], rotations[target_index])
                    error = abs(yaw - target_angle)
                    candidate_value = (error, target_index, yaw, translation)
                    if side_best is None or candidate_value < side_best:
                        side_best = candidate_value
                if side_best is not None:
                    choices[side] = side_best
            if len(choices) != 2 or choices["left"][0] > max_error or choices["right"][0] > max_error:
                continue
            score = choices["left"][0] + choices["right"][0]
            qualifying.append((score, source_index, choices))
        qualifying.sort(key=lambda value: (value[0], value[1]))
        if len(qualifying) < maximum_triplets:
            scene_diagnostics.append(
                {
                    "split": split,
                    "scene": scene,
                    "camera": camera,
                    "status": "insufficient_qualifying_triplets",
                    "qualifying_source_count": len(qualifying),
                    "requested_triplets": maximum_triplets,
                }
            )
            continue
        scene_triplets = []
        for rank, (_, source_index, choices) in enumerate(qualifying[:maximum_triplets], start=1):
            item: dict[str, object] = {
                "split": split,
                "scene": scene,
                "camera": camera,
                "scene_type": candidate["scene_type"],
                "scene_triplet_rank": rank,
                "source_frame": int(frames[source_index]),
                "left_frame": int(frames[choices["left"][1]]),
                "right_frame": int(frames[choices["right"][1]]),
                "left_yaw_deg": choices["left"][2],
                "right_yaw_deg": choices["right"][2],
                "left_translation_m": choices["left"][3],
                "right_translation_m": choices["right"][3],
                "angle_error_sum_deg": choices["left"][0] + choices["right"][0],
            }
            selected.append(item)
            scene_triplets.append(item)
        scene_diagnostics.append(
            {
                "split": split,
                "scene": scene,
                "camera": camera,
                "status": "selected",
                "public_pose_frames": len(indices),
                "qualifying_source_count": len(qualifying),
                "selected_triplet_count": len(scene_triplets),
                "triplets": scene_triplets,
            }
        )

    expected_count = len(grouped) * maximum_triplets
    held_out = set(candidates["held_out_test_scenes"])
    selected_scenes = {str(value["scene"]) for value in selected}
    all_selected = len(selected) == expected_count
    no_leakage = not held_out.intersection(selected_scenes)
    status = "pass" if all_selected and no_leakage else "failed_selection"
    output = {
        "schema_version": "stage1.6-hlp-train-01-triplets-v1",
        "status": status,
        "source_candidates": str(args.candidates.relative_to(Path.cwd())),
        "source_candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "target_motion": motion,
        "selection_policy": {
            "minimum_endpoint_translation_m": minimum_translation,
            "objective": "minimum sum of absolute left/right yaw endpoint errors",
            "tie_break": "lowest source pose index; unique source frames by construction",
            "maximum_triplets_per_scene": maximum_triplets,
        },
        "held_out_test_scenes": sorted(held_out),
        "triplets": selected,
    }
    record = {
        "schema_version": "stage1.6-hlp-train-01-pose-selection-v1",
        "status": status,
        "producer_machine_id": "linux5080",
        "output_config": str(args.output_config.relative_to(Path.cwd())),
        "selection_policy": {
            "minimum_endpoint_translation_m": minimum_translation,
            "maximum_endpoint_error_deg": max_error,
        },
        "summary": {
            "candidate_scenes": expected_count,
            "selected_triplets": len(selected),
            "train_triplets": sum(value["split"] == "train" for value in selected),
            "validation_triplets": sum(value["split"] == "val" for value in selected),
            "held_out_overlap_count": len(held_out.intersection(selected_scenes)),
            "all_endpoint_errors_within_limit": all(
                float(value["angle_error_sum_deg"]) <= 2 * max_error for value in selected
            ),
            "all_endpoint_translations_above_minimum": all(
                min(float(value["left_translation_m"]), float(value["right_translation_m"]))
                >= minimum_translation
                for value in selected
            ),
        },
        "scenes": scene_diagnostics,
        "quality_boundary": "real-pose selection only; RGB-D files and hidden evidence are not fetched or generated yet",
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record["output_config_sha256"] = hashlib.sha256(args.output_config.read_bytes()).hexdigest()
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"]}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
