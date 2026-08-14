#!/usr/bin/env python3
"""Enumerate auditable ETH3D source-target candidates for InfiniSplat.

The InfiniSplat paper selects directed pairs from ten-consecutive-view
windows, requires more than 60% camera-frustum overlap, and bins metric
camera baselines.  The released code does not include the authors' pair list
or overlap implementation.  This script therefore reports COLMAP sparse-point
co-visibility as an explicitly named proxy; it must not be presented as the
paper's undisclosed exact frustum-overlap value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    name: str
    camera_id: int
    qvec_wxyz: np.ndarray
    tvec: np.ndarray
    point_ids: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--min-baseline-m", type=float, default=2.0)
    parser.add_argument("--min-covisibility-proxy", type=float, default=0.60)
    parser.add_argument("--max-relative-rotation-deg", type=float, default=60.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in data_lines(path):
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
    return cameras


def load_images(path: Path) -> list[ImagePose]:
    lines = data_lines(path)
    if len(lines) % 2:
        raise ValueError(f"Expected two data lines per COLMAP image: {path}")
    images: list[ImagePose] = []
    for pose_line, points_line in zip(lines[0::2], lines[1::2]):
        fields = pose_line.split()
        observations = points_line.split()
        if len(observations) % 3:
            raise ValueError(f"Malformed POINTS2D line for {fields[-1]}")
        point_ids = frozenset(
            int(observations[index])
            for index in range(2, len(observations), 3)
            if int(observations[index]) >= 0
        )
        images.append(
            ImagePose(
                image_id=int(fields[0]),
                qvec_wxyz=np.asarray([float(value) for value in fields[1:5]]),
                tvec=np.asarray([float(value) for value in fields[5:8]]),
                camera_id=int(fields[8]),
                name=fields[9],
                point_ids=point_ids,
            )
        )
    return sorted(images, key=lambda image: image.name)


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def w2c_matrix(image: ImagePose) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation(image.qvec_wxyz)
    matrix[:3, 3] = image.tvec
    return matrix


def camera_center(image: ImagePose) -> np.ndarray:
    rotation = quaternion_to_rotation(image.qvec_wxyz)
    return -(rotation.T @ image.tvec)


def rotation_angle_deg(source: ImagePose, target: ImagePose) -> float:
    source_rotation = quaternion_to_rotation(source.qvec_wxyz)
    target_rotation = quaternion_to_rotation(target.qvec_wxyz)
    relative = target_rotation @ source_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def pinhole_intrinsics(camera: Camera) -> list[list[float]]:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"Unsupported camera model: {camera}")
    fx, fy, cx, cy = camera.params
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def baseline_bin(baseline: float) -> str:
    if baseline < 0.5:
        return "[0,0.5)"
    if baseline < 1.0:
        return "[0.5,1)"
    if baseline < 2.0:
        return "[1,2)"
    return "[2,+inf)"


def main() -> None:
    args = parse_args()
    calibration_dir = args.calibration_dir.resolve()
    images_root = args.images_root.resolve()
    cameras_path = calibration_dir / "cameras.txt"
    images_path = calibration_dir / "images.txt"
    cameras = load_cameras(cameras_path)
    images = load_images(images_path)

    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(images):
        for target_index, target in enumerate(images):
            if source_index == target_index:
                continue
            sequence_distance = abs(target_index - source_index)
            if sequence_distance >= args.window_size:
                continue

            shared = source.point_ids & target.point_ids
            source_count = len(source.point_ids)
            target_count = len(target.point_ids)
            shared_count = len(shared)
            denominator_min = min(source_count, target_count)
            union_count = len(source.point_ids | target.point_ids)
            covisibility_min = shared_count / denominator_min if denominator_min else 0.0
            covisibility_source = shared_count / source_count if source_count else 0.0
            covisibility_target = shared_count / target_count if target_count else 0.0
            covisibility_jaccard = shared_count / union_count if union_count else 0.0

            baseline = float(np.linalg.norm(camera_center(source) - camera_center(target)))
            rotation = rotation_angle_deg(source, target)
            source_camera = cameras[source.camera_id]
            target_camera = cameras[target.camera_id]
            source_to_target = w2c_matrix(target) @ np.linalg.inv(w2c_matrix(source))
            passes_proxy = (
                baseline >= args.min_baseline_m
                and rotation <= args.max_relative_rotation_deg
                and covisibility_min >= args.min_covisibility_proxy
            )
            rows.append(
                {
                    "source_index": source_index,
                    "target_index": target_index,
                    "sequence_distance": sequence_distance,
                    "source_name": source.name,
                    "target_name": target.name,
                    "source_image_path": str((images_root / Path(source.name).name).resolve()),
                    "target_image_path": str((images_root / Path(target.name).name).resolve()),
                    "baseline_m": baseline,
                    "baseline_bin": baseline_bin(baseline),
                    "relative_rotation_deg": rotation,
                    "source_sparse_points": source_count,
                    "target_sparse_points": target_count,
                    "shared_sparse_points": shared_count,
                    "covisibility_proxy_shared_over_min": covisibility_min,
                    "covisibility_proxy_shared_over_source": covisibility_source,
                    "covisibility_proxy_shared_over_target": covisibility_target,
                    "covisibility_proxy_jaccard": covisibility_jaccard,
                    "passes_large_baseline_proxy_filter": passes_proxy,
                    "source_camera": {
                        "camera_id": source.camera_id,
                        "image_shape_wh": [source_camera.width, source_camera.height],
                        "intrinsics_px": pinhole_intrinsics(source_camera),
                        "w2c_world": w2c_matrix(source).tolist(),
                    },
                    "target_camera": {
                        "camera_id": target.camera_id,
                        "image_shape_wh": [target_camera.width, target_camera.height],
                        "intrinsics_px": pinhole_intrinsics(target_camera),
                        "w2c_world": w2c_matrix(target).tolist(),
                    },
                    "source_camera_to_target_camera_w2c": source_to_target.tolist(),
                }
            )

    rows.sort(
        key=lambda row: (
            not bool(row["passes_large_baseline_proxy_filter"]),
            -float(row["covisibility_proxy_shared_over_min"]),
            -float(row["baseline_m"]),
            float(row["relative_rotation_deg"]),
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0-eth3d-large-baseline-pair-candidates",
        "status": "candidate_selection_not_official_pair_list",
        "methodology": {
            "paper_protocol_known": {
                "trajectory_window_size": args.window_size,
                "metric_baseline_bins_m": ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,+inf)"],
                "required_frustum_overlap": ">60%",
                "filters": "excessive rotations and backward-facing views",
            },
            "released_material_gap": "No official 512-pair list or frustum-overlap implementation was released.",
            "local_proxy": "shared registered COLMAP sparse points divided by the smaller visible-point count",
            "warning": "The proxy is auditable but is not the paper's exact undisclosed frustum-overlap metric.",
            "local_thresholds": {
                "min_baseline_m": args.min_baseline_m,
                "min_covisibility_proxy": args.min_covisibility_proxy,
                "max_relative_rotation_deg": args.max_relative_rotation_deg,
            },
        },
        "inputs": {
            "cameras_txt_sha256": sha256(cameras_path),
            "images_txt_sha256": sha256(images_path),
            "image_count": len(images),
            "camera_count": len(cameras),
        },
        "summary": {
            "directed_candidate_count": len(rows),
            "passing_proxy_count": sum(bool(row["passes_large_baseline_proxy_filter"]) for row in rows),
        },
        "candidates": rows,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_fields = [
        "source_index",
        "target_index",
        "sequence_distance",
        "source_name",
        "target_name",
        "baseline_m",
        "baseline_bin",
        "relative_rotation_deg",
        "source_sparse_points",
        "target_sparse_points",
        "shared_sparse_points",
        "covisibility_proxy_shared_over_min",
        "covisibility_proxy_shared_over_source",
        "covisibility_proxy_shared_over_target",
        "covisibility_proxy_jaccard",
        "passes_large_baseline_proxy_filter",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in csv_fields} for row in rows)

    print(json.dumps(payload["summary"], indent=2))
    for row in rows[:10]:
        print(
            f"{row['source_name']} -> {row['target_name']} "
            f"baseline={row['baseline_m']:.3f}m "
            f"rotation={row['relative_rotation_deg']:.2f}deg "
            f"covis_min={row['covisibility_proxy_shared_over_min']:.3f} "
            f"pass={row['passes_large_baseline_proxy_filter']}"
        )


if __name__ == "__main__":
    main()
