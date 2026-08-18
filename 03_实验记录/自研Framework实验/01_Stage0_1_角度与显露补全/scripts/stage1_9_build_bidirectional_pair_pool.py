#!/usr/bin/env python3
"""Build the no-download Stage 1.9 directed-pair pool from frozen HLP-TRAIN-03 triplets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


HYPERSIM_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--filter-angle-band", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def relative_yaw_deg(source_rotation: np.ndarray, target_rotation: np.ndarray) -> float:
    source_opencv = source_rotation @ HYPERSIM_TO_OPENCV
    target_opencv = target_rotation @ HYPERSIM_TO_OPENCV
    relative = source_opencv.T @ target_opencv
    return float(np.degrees(np.arctan2(relative[0, 2], relative[2, 2])))


def required_frame_files(root: Path, scene: str, camera: str, frame: int) -> list[Path]:
    return [
        root / scene / "images" / f"scene_{camera}_final_hdf5" / f"frame.{frame:04d}.color.hdf5",
        root / scene / "images" / f"scene_{camera}_geometry_hdf5" / f"frame.{frame:04d}.depth_meters.hdf5",
        root / scene / "images" / f"scene_{camera}_geometry_hdf5" / f"frame.{frame:04d}.position.hdf5",
    ]


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    for name in ("triplets", "dataset_root", "output_config", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_config.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite pair config or audit record")
    source = json.loads(args.triplets.read_text(encoding="utf-8"))
    if source.get("status") != "pass" or len(source.get("triplets", [])) != 50:
        raise ValueError("expected the frozen passing 50-scene HLP-TRAIN-03 triplet config")

    pairs: list[dict[str, object]] = []
    missing_files: list[str] = []
    scene_diagnostics: list[dict[str, object]] = []
    for triplet in source["triplets"]:
        scene, camera = str(triplet["scene"]), str(triplet["camera"])
        detail = args.dataset_root / scene / "_detail" / camera
        frame_path = detail / "camera_keyframe_frame_indices.hdf5"
        rotation_path = detail / "camera_keyframe_orientations.hdf5"
        position_path = detail / "camera_keyframe_positions.hdf5"
        scale_path = args.dataset_root / scene / "_detail" / "metadata_scene.csv"
        for path in (frame_path, rotation_path, position_path, scale_path):
            if not path.is_file():
                missing_files.append(portable(path))
        if missing_files:
            continue
        frames = load_hdf5(frame_path).astype(int)
        rotations = load_hdf5(rotation_path).astype(np.float64)
        positions = load_hdf5(position_path).astype(np.float64)
        index = {int(frame): offset for offset, frame in enumerate(frames)}
        scale = scene_scale(scale_path)
        roles = {
            "left": int(triplet["left_frame"]),
            "center": int(triplet["source_frame"]),
            "right": int(triplet["right_frame"]),
        }
        for frame in roles.values():
            for path in required_frame_files(args.dataset_root, scene, camera, frame):
                if not path.is_file():
                    missing_files.append(portable(path))
        directed_roles = (("center", "left"), ("center", "right"), ("left", "center"), ("right", "center"))
        scene_pairs = []
        for source_role, target_role in directed_roles:
            source_frame, target_frame = roles[source_role], roles[target_role]
            source_index, target_index = index[source_frame], index[target_frame]
            yaw = relative_yaw_deg(rotations[source_index], rotations[target_index])
            translation = float(np.linalg.norm(positions[target_index] - positions[source_index]) * scale)
            item = {
                "split": triplet["split"],
                "scene": scene,
                "camera": camera,
                "scene_type": triplet["scene_type"],
                "source_role": source_role,
                "target_role": target_role,
                "source_frame": source_frame,
                "target_frame": target_frame,
                "yaw_deg": yaw,
                "translation_m": translation,
            }
            pairs.append(item)
            scene_pairs.append(item)
        scene_diagnostics.append({"scene": scene, "split": triplet["split"], "pairs": scene_pairs})

    rejected_angle_pairs = [p for p in pairs if not 12.0 <= abs(float(p["yaw_deg"])) <= 18.0]
    if args.filter_angle_band:
        pairs = [p for p in pairs if 12.0 <= abs(float(p["yaw_deg"])) <= 18.0]
    unique_keys = {(p["scene"], p["camera"], p["source_frame"], p["target_frame"]) for p in pairs}
    angle_ok = all(12.0 <= abs(float(p["yaw_deg"])) <= 18.0 for p in pairs)
    translation_ok = all(float(p["translation_m"]) >= 0.1 for p in pairs)
    split_counts = {name: sum(p["split"] == name for p in pairs) for name in ("train", "val")}
    gates = {
        "expected_pairs": len(pairs) == (195 if args.filter_angle_band else 200),
        "expected_train_pairs": split_counts["train"] == (155 if args.filter_angle_band else 160),
        "expected_val_pairs": split_counts["val"] == 40,
        "unique_directed_pairs": len(unique_keys) == len(pairs),
        "all_required_files_local": not missing_files,
        "yaw_absolute_12_to_18_deg": angle_ok,
        "translation_at_least_0_1_m": translation_ok,
        "held_out_test_excluded": not set(source["held_out_test_scenes"]).intersection({p["scene"] for p in pairs}),
    }
    status = "pass" if all(gates.values()) else "failed"
    output = {
        "schema_version": f"stage1.9-bidirectional-pair-pool-{'v2' if args.filter_angle_band else 'v1'}",
        "status": status,
        "source_triplets": portable(args.triplets),
        "source_triplets_sha256": sha256(args.triplets),
        "held_out_test_scenes": source["held_out_test_scenes"],
        "policy": {
            "included": ["center_to_left", "center_to_right", "left_to_center", "right_to_center"],
            "excluded": ["left_to_right", "right_to_left"],
            "reason": "Retain approximately 15-degree one-sided requests; do not turn the 30-degree total-range protocol into a 30-degree one-sided request.",
            "inference_inputs": "one source RGB plus frozen RGB-only BaseDepth and one explicit target camera",
        },
        "rejected_angle_pairs": rejected_angle_pairs if args.filter_angle_band else [],
        "pairs": pairs,
    }
    record = {
        "schema_version": f"stage1.9-bidirectional-pair-pool-audit-{'v2' if args.filter_angle_band else 'v1'}",
        "status": status,
        "producer_machine_id": args.machine_id,
        "inputs": {"triplets": portable(args.triplets), "sha256": sha256(args.triplets), "dataset_root": portable(args.dataset_root)},
        "counts": {"scenes": len(scene_diagnostics), "pairs": len(pairs), **split_counts},
        "ranges": {
            "absolute_yaw_deg_minimum": min(abs(float(p["yaw_deg"])) for p in pairs),
            "absolute_yaw_deg_maximum": max(abs(float(p["yaw_deg"])) for p in pairs),
            "translation_m_minimum": min(float(p["translation_m"]) for p in pairs),
            "translation_m_maximum": max(float(p["translation_m"]) for p in pairs),
        },
        "gates": gates,
        "missing_files": sorted(set(missing_files)),
        "download_required": False,
        "next_gate": "Generate frozen RGB-only BaseDepth for the 100 endpoint frames, then create source-relative three-region visibility truth and qualify directed pairs before any model training.",
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "counts": record["counts"], "ranges": record["ranges"], "gates": gates}, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
