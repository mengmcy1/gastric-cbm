#!/usr/bin/env python3
"""Build source-relative three-region truth and a qualified Stage 1.9 directed-pair pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

import stage1_hypersim_visibility_masks as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualified-config", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--minimum-train-targets", type=int, default=100)
    parser.add_argument("--minimum-val-targets", type=int, default=20)
    parser.add_argument("--minimum-train-scenes", type=int, default=30)
    parser.add_argument("--minimum-val-scenes", type=int, default=8)
    return parser.parse_args()


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    for name in ("pair_config", "dataset_root", "camera_parameters", "output_dir", "qualified_config", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.qualified_config.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite Stage 1.9 visibility outputs")
    config = json.loads(args.pair_config.read_text(encoding="utf-8"))
    if config.get("status") != "pass" or len(config.get("pairs", [])) != 195:
        raise ValueError("expected the passing 195-pair Stage 1.9 v2 config")
    scenes = {p["scene"] for p in config["pairs"]}
    parameters = common.camera_parameters(args.camera_parameters, scenes)
    args.output_dir.mkdir(parents=True)
    results = []
    scene_cache = {}
    for pair in config["pairs"]:
        scene, camera = pair["scene"], pair["camera"]
        cache_key = (scene, camera)
        if cache_key not in scene_cache:
            scene_dir = args.dataset_root / scene
            frames = common.load_hdf5(scene_dir / "_detail" / camera / "camera_keyframe_frame_indices.hdf5").astype(int)
            rotations = common.load_hdf5(scene_dir / "_detail" / camera / "camera_keyframe_orientations.hdf5")
            positions = common.load_hdf5(scene_dir / "_detail" / camera / "camera_keyframe_positions.hdf5")
            scene_cache[cache_key] = (positions, rotations, {int(f): i for i, f in enumerate(frames)}, common.meters_per_asset_unit(scene_dir / "_detail" / "metadata_scene.csv"))
        positions, rotations, index_by_frame, scale = scene_cache[cache_key]
        source_frame, target_frame = int(pair["source_frame"]), int(pair["target_frame"])
        source_index, target_index = index_by_frame[source_frame], index_by_frame[target_frame]
        source_depth_path = common.frame_path(args.dataset_root, scene, camera, source_frame, "depth_meters")
        target_depth_path = common.frame_path(args.dataset_root, scene, camera, target_frame, "depth_meters")
        target_position_path = common.frame_path(args.dataset_root, scene, camera, target_frame, "position")
        source_depth = common.load_hdf5(source_depth_path)
        target_depth = common.load_hdf5(target_depth_path)
        target_positions = common.load_hdf5(target_position_path)
        height, width = source_depth.shape
        metadata = parameters[scene]
        if target_depth.shape != (height, width) or (width, height) != (metadata["width"], metadata["height"]):
            raise ValueError(f"resolution mismatch: {scene}/{source_frame}->{target_frame}")
        finite_target = np.isfinite(target_positions).all(axis=2) & np.isfinite(target_depth) & (target_depth > 0)
        target_distance = np.linalg.norm(target_positions - positions[target_index][None, None], axis=2) * scale
        depth_residual = np.abs(target_distance - target_depth)
        depth_relative = depth_residual / np.maximum(target_depth, 1e-12)
        x, y, ray_scale, _ = common.project_world_to_source(
            target_positions, positions[source_index], rotations[source_index], metadata["M_cam_from_uv"], width, height
        )
        x, y, ray_scale = x.reshape(height, width), y.reshape(height, width), ray_scale.reshape(height, width)
        inside = finite_target & np.isfinite(x) & np.isfinite(y) & (ray_scale > 0)
        inside &= (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        sampled = map_coordinates(source_depth, [np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)], order=1, mode="nearest")
        distance_from_source = np.linalg.norm(target_positions - positions[source_index][None, None], axis=2) * scale
        tolerance = np.maximum(0.02, 0.005 * sampled)
        source_valid = np.isfinite(sampled) & (sampled > 0)
        observed = inside & source_valid & (np.abs(distance_from_source - sampled) <= tolerance)
        occluded = inside & source_valid & (distance_from_source > sampled + tolerance)
        conflict = inside & source_valid & (distance_from_source < sampled - tolerance)
        outside = finite_target & ~inside
        unresolved = finite_target & ~(observed | occluded | conflict | outside)
        classified = observed | occluded | conflict | outside | unresolved
        depth_p99 = float(np.quantile(depth_relative[finite_target], 0.99))
        complete = bool(np.array_equal(classified, finite_target))
        eligible = depth_p99 <= 0.002 and complete and bool(occluded.any()) and bool(outside.any())
        stem = f"{scene}-{camera}-{source_frame:04d}-to-{target_frame:04d}"
        arrays = args.output_dir / f"{stem}-visibility.npz"
        preview = args.output_dir / f"{stem}-visibility.png"
        np.savez_compressed(
            arrays,
            valid_target_uint8=finite_target.astype(np.uint8),
            observed_from_source_uint8=observed.astype(np.uint8),
            occlusion_hidden_uint8=occluded.astype(np.uint8),
            outside_source_fov_uint8=outside.astype(np.uint8),
            geometry_conflict_uint8=conflict.astype(np.uint8),
            unresolved_uint8=unresolved.astype(np.uint8),
            target_depth_meters_float32=target_depth.astype(np.float32),
        )
        common.save_mask_preview(preview, observed, occluded, outside, conflict)
        results.append({
            **pair,
            "eligible": eligible,
            "target_depth_vs_position_p99_relative": depth_p99,
            "classification_complete": complete,
            "valid_target_pixels": int(finite_target.sum()),
            "observed_from_source_pixels": int(observed.sum()),
            "occlusion_hidden_pixels": int(occluded.sum()),
            "outside_source_fov_pixels": int(outside.sum()),
            "geometry_conflict_pixels": int(conflict.sum()),
            "unresolved_pixels": int(unresolved.sum()),
            "arrays": portable(arrays), "arrays_sha256": common.sha256(arrays),
            "preview": portable(preview), "preview_sha256": common.sha256(preview),
            "source_depth_sha256": common.sha256(source_depth_path),
            "target_depth_sha256": common.sha256(target_depth_path),
            "target_position_sha256": common.sha256(target_position_path),
        })
    eligible = [r for r in results if r["eligible"]]
    train = [r for r in eligible if r["split"] == "train"]
    val = [r for r in eligible if r["split"] == "val"]
    train_scenes, val_scenes = {r["scene"] for r in train}, {r["scene"] for r in val}
    gates = {
        "eligible_train_targets_at_least_100": len(train) >= args.minimum_train_targets,
        "eligible_val_targets_at_least_20": len(val) >= args.minimum_val_targets,
        "eligible_train_scenes_at_least_30": len(train_scenes) >= args.minimum_train_scenes,
        "eligible_val_scenes_at_least_8": len(val_scenes) >= args.minimum_val_scenes,
        "all_classifications_complete": all(r["classification_complete"] for r in results),
        "held_out_test_excluded": not set(config.get("held_out_test_scenes", [])).intersection({r["scene"] for r in results}),
    }
    status = "pass" if all(gates.values()) else "failed_insufficient_qualified_pool"
    qualified = {
        "schema_version": "stage1.9-qualified-directed-pair-pool-v1",
        "status": "pass" if status == "pass" else "failed",
        "source_pair_config": portable(args.pair_config),
        "source_pair_config_sha256": common.sha256(args.pair_config),
        "qualification": {
            "target_depth_vs_position_p99_relative_maximum": 0.002,
            "classification_complete": True,
            "occlusion_hidden_nonempty": True,
            "outside_source_fov_nonempty": True,
        },
        "pairs": eligible,
    }
    record = {
        "schema_version": "stage1.9-directed-visibility-pool-v1",
        "status": status,
        "producer_machine_id": args.machine_id,
        "inputs": {"pair_config": portable(args.pair_config), "pair_config_sha256": common.sha256(args.pair_config), "dataset_root": portable(args.dataset_root)},
        "classification": {"absolute_tolerance_m": 0.02, "relative_tolerance": 0.005, "target_depth_position_p99_relative_maximum": 0.002},
        "gates": gates,
        "summary": {
            "candidate_targets": len(results), "eligible_targets": len(eligible),
            "eligible_train_targets": len(train), "eligible_val_targets": len(val),
            "eligible_train_scenes": len(train_scenes), "eligible_val_scenes": len(val_scenes),
            "total_occlusion_hidden_pixels": sum(r["occlusion_hidden_pixels"] for r in eligible),
            "total_outside_source_fov_pixels": sum(r["outside_source_fov_pixels"] for r in eligible),
        },
        "views": results,
        "qualified_config": portable(args.qualified_config),
        "quality_boundary": "Target truth is label/evaluator only. Provider BaseDepth comes from the separately frozen RGB-only UniDepth records.",
    }
    args.qualified_config.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.qualified_config.write_text(json.dumps(qualified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"], "gates": gates}, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
