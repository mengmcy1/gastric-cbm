#!/usr/bin/env python3
"""Build dense Hypersim occlusion-hidden and outside-source-FOV truth masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["dataset"][:].astype(np.float64)


def meters_per_asset_unit(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = {row[0]: row[1] for row in csv.reader(stream) if len(row) >= 2}
    return float(values["meters_per_asset_unit"])


def camera_parameters(path: Path, scenes: set[str]) -> dict[str, dict[str, object]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scene = row["scene_name"]
            if scene not in scenes:
                continue
            matrix = np.asarray(
                [float(row[f"M_cam_from_uv_{i}{j}"]) for i in range(3) for j in range(3)],
                dtype=np.float64,
            ).reshape(3, 3)
            result[scene] = {
                "width": int(float(row["settings_output_img_width"])),
                "height": int(float(row["settings_output_img_height"])),
                "M_cam_from_uv": matrix,
            }
    if set(result) != scenes:
        raise KeyError(f"camera metadata missing scenes: {sorted(scenes.difference(result))}")
    return result


def frame_path(root: Path, scene: str, camera: str, frame: int, suffix: str) -> Path:
    group = "final_hdf5" if suffix == "color" else "geometry_hdf5"
    return root / scene / "images" / f"scene_{camera}_{group}" / f"frame.{frame:04d}.{suffix}.hdf5"


def project_world_to_source(
    positions_world: np.ndarray,
    camera_position: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    matrix_camera_from_uv: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = positions_world.reshape(-1, 3)
    camera = (points - camera_position) @ rotation_world_from_camera
    uv_h = camera @ np.linalg.inv(matrix_camera_from_uv).T
    ray_scale = uv_h[:, 2]
    u = uv_h[:, 0] / ray_scale
    v = uv_h[:, 1] / ray_scale
    x = (u + 1.0) * width / 2.0 - 0.5
    y = (1.0 - v) * height / 2.0 - 0.5
    return x, y, ray_scale, camera


def save_mask_preview(path: Path, observed: np.ndarray, occluded: np.ndarray, outside: np.ndarray, conflict: np.ndarray) -> None:
    preview = np.zeros((*observed.shape, 3), dtype=np.uint8)
    preview[observed] = [50, 200, 80]
    preview[occluded] = [230, 60, 60]
    preview[outside] = [60, 100, 230]
    preview[conflict] = [230, 180, 40]
    Image.fromarray(preview).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.02)
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "dataset_root", "camera_parameters", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scenes = {value["scene"] for value in config["triplets"]}
    parameters = camera_parameters(args.camera_parameters, scenes)
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for triplet in config["triplets"]:
        scene = triplet["scene"]
        camera = triplet["camera"]
        scene_dir = args.dataset_root / scene
        output_scene = args.output_dir / scene
        output_scene.mkdir()
        frame_indices = load_hdf5(scene_dir / "_detail" / camera / "camera_keyframe_frame_indices.hdf5").astype(int)
        rotations = load_hdf5(scene_dir / "_detail" / camera / "camera_keyframe_orientations.hdf5")
        positions = load_hdf5(scene_dir / "_detail" / camera / "camera_keyframe_positions.hdf5")
        index_by_frame = {int(frame): index for index, frame in enumerate(frame_indices)}
        scale = meters_per_asset_unit(scene_dir / "_detail" / "metadata_scene.csv")
        source_frame = int(triplet["source_frame"])
        source_index = index_by_frame[source_frame]
        source_depth_path = frame_path(args.dataset_root, scene, camera, source_frame, "depth_meters")
        source_depth = load_hdf5(source_depth_path)
        height, width = source_depth.shape
        metadata = parameters[scene]
        if (width, height) != (metadata["width"], metadata["height"]):
            raise ValueError(f"resolution mismatch for {scene}: {(width, height)}")

        for side in ("left", "right"):
            target_frame = int(triplet[f"{side}_frame"])
            target_index = index_by_frame[target_frame]
            target_position_path = frame_path(args.dataset_root, scene, camera, target_frame, "position")
            target_depth_path = frame_path(args.dataset_root, scene, camera, target_frame, "depth_meters")
            target_positions = load_hdf5(target_position_path)
            target_depth = load_hdf5(target_depth_path)
            finite_target = np.isfinite(target_positions).all(axis=2) & np.isfinite(target_depth) & (target_depth > 0)
            target_distance_from_position = np.linalg.norm(
                target_positions - positions[target_index][None, None], axis=2
            ) * scale
            depth_residual = np.abs(target_distance_from_position - target_depth)
            depth_relative_residual = depth_residual / np.maximum(target_depth, 1e-12)
            x, y, ray_scale, _ = project_world_to_source(
                target_positions,
                positions[source_index],
                rotations[source_index],
                metadata["M_cam_from_uv"],
                width,
                height,
            )
            x = x.reshape(height, width)
            y = y.reshape(height, width)
            ray_scale = ray_scale.reshape(height, width)
            inside = (
                finite_target
                & np.isfinite(x)
                & np.isfinite(y)
                & (ray_scale > 0)
                & (x >= 0)
                & (x <= width - 1)
                & (y >= 0)
                & (y <= height - 1)
            )
            sampled_source_depth = map_coordinates(
                source_depth,
                [np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)],
                order=1,
                mode="nearest",
            )
            distance_from_source = np.linalg.norm(
                target_positions - positions[source_index][None, None], axis=2
            ) * scale
            tolerance = np.maximum(args.absolute_tolerance_m, args.relative_tolerance * sampled_source_depth)
            source_depth_valid = np.isfinite(sampled_source_depth) & (sampled_source_depth > 0)
            observed = inside & source_depth_valid & (np.abs(distance_from_source - sampled_source_depth) <= tolerance)
            occluded = inside & source_depth_valid & (distance_from_source > sampled_source_depth + tolerance)
            conflict = inside & source_depth_valid & (distance_from_source < sampled_source_depth - tolerance)
            outside = finite_target & ~inside
            unresolved = finite_target & ~(observed | occluded | conflict | outside)
            arrays_path = output_scene / f"{side}_visibility_truth.npz"
            np.savez_compressed(
                arrays_path,
                valid_target_uint8=finite_target.astype(np.uint8),
                observed_from_source_uint8=observed.astype(np.uint8),
                occlusion_hidden_uint8=occluded.astype(np.uint8),
                outside_source_fov_uint8=outside.astype(np.uint8),
                geometry_conflict_uint8=conflict.astype(np.uint8),
                unresolved_uint8=unresolved.astype(np.uint8),
                target_depth_meters_float32=target_depth.astype(np.float32),
                target_point_distance_from_source_m_float32=distance_from_source.astype(np.float32),
                sampled_source_depth_m_float32=sampled_source_depth.astype(np.float32),
            )
            preview_path = output_scene / f"{side}_visibility_truth.png"
            save_mask_preview(preview_path, observed, occluded, outside, conflict)
            valid_count = int(finite_target.sum())
            classified = observed | occluded | outside | conflict | unresolved
            results.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "side": side,
                    "yaw_deg": triplet[f"{side}_yaw_deg"],
                    "translation_m": triplet[f"{side}_translation_m"],
                    "resolution_wh": [width, height],
                    "meters_per_asset_unit": scale,
                    "target_depth_vs_position_median_abs_m": float(np.median(depth_residual[finite_target])),
                    "target_depth_vs_position_p99_abs_m": float(np.quantile(depth_residual[finite_target], 0.99)),
                    "target_depth_vs_position_p99_relative": float(
                        np.quantile(depth_relative_residual[finite_target], 0.99)
                    ),
                    "valid_target_pixels": valid_count,
                    "observed_from_source_pixels": int(observed.sum()),
                    "occlusion_hidden_pixels": int(occluded.sum()),
                    "occlusion_hidden_fraction": float(occluded.sum() / valid_count),
                    "outside_source_fov_pixels": int(outside.sum()),
                    "outside_source_fov_fraction": float(outside.sum() / valid_count),
                    "geometry_conflict_pixels": int(conflict.sum()),
                    "geometry_conflict_fraction": float(conflict.sum() / valid_count),
                    "unresolved_pixels": int(unresolved.sum()),
                    "classification_complete": bool(np.array_equal(classified, finite_target)),
                    "arrays": str(arrays_path.relative_to(Path.cwd())),
                    "arrays_sha256": sha256(arrays_path),
                    "preview": str(preview_path.relative_to(Path.cwd())),
                    "preview_sha256": sha256(preview_path),
                    "source_depth_sha256": sha256(source_depth_path),
                    "target_depth_sha256": sha256(target_depth_path),
                    "target_position_sha256": sha256(target_position_path),
                }
            )

    gates = {
        "all_target_depth_matches_position_p99_relative_at_most_0_2_percent": all(
            item["target_depth_vs_position_p99_relative"] <= 0.002 for item in results
        ),
        "all_classifications_complete": all(item["classification_complete"] for item in results),
        "all_views_have_occlusion_hidden_truth": all(item["occlusion_hidden_pixels"] > 0 for item in results),
        "all_views_have_outside_source_fov_truth": all(item["outside_source_fov_pixels"] > 0 for item in results),
    }
    record = {
        "schema_version": "stage1.4-hlp-geo-01-visibility-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "producer_machine_id": "linux5080",
        "config": str(args.config.relative_to(Path.cwd())),
        "config_sha256": sha256(args.config),
        "camera_parameters": str(args.camera_parameters.relative_to(Path.cwd())),
        "camera_parameters_sha256": sha256(args.camera_parameters),
        "classification": {
            "projection": "Hypersim official M_cam_from_uv inverse with pixel-center mapping",
            "depth_comparison": "target world position distance from source vs bilinear source depth_meters",
            "absolute_tolerance_m": args.absolute_tolerance_m,
            "relative_tolerance": args.relative_tolerance,
            "depth_position_validation_note": (
                "official depth and position files are float16; use relative P99 to avoid rejecting "
                "far scenes solely from expected half-precision absolute quantization"
            ),
            "colors": {"observed": "green", "occlusion_hidden": "red", "outside_source_fov": "blue", "conflict": "yellow"},
        },
        "gates": gates,
        "views": results,
        "summary": {
            "triplets": len(config["triplets"]),
            "target_views": len(results),
            "total_occlusion_hidden_pixels": sum(item["occlusion_hidden_pixels"] for item in results),
            "total_outside_source_fov_pixels": sum(item["outside_source_fov_pixels"] for item in results),
            "maximum_geometry_conflict_fraction": max(item["geometry_conflict_fraction"] for item in results),
        },
        "quality_boundary": "dense truth masks are ready; Flash3D prediction has not yet been evaluated against them",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "gates": gates, "summary": record["summary"]}, indent=2))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
