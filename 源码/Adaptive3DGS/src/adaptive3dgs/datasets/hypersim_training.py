"""Leakage-aware Hypersim RGB-D triplets for offline supervision building."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..clb import Camera
from ..supervision import TargetRGBDObservation
from ..validation import ValidationError
from .hypersim import (
    _camera_parameter_rows,
    _frame_path,
    _load_hdf5,
    _scene_scale,
    calibration_from_hypersim,
    camera_from_hypersim,
    intrinsics_from_hypersim,
    radial_depth_to_z,
    tonemap_hypersim,
)


@dataclass(frozen=True, slots=True)
class HypersimTrainingTriplet:
    split: str
    scene: str
    camera_name: str
    source_frame_index: int
    source_rgb_uint8: np.ndarray
    source_depth_z_label_only_float32: np.ndarray
    source_camera: Camera
    target_observations: tuple[TargetRGBDObservation, ...]


def load_hypersim_training_triplets(
    config_path: str | Path,
    dataset_root: str | Path,
    camera_parameters_path: str | Path,
) -> tuple[HypersimTrainingTriplet, ...]:
    config_path = Path(config_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    camera_parameters_path = Path(camera_parameters_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "pass":
        raise ValidationError("training triplet config must have status=pass")
    triplets = config.get("triplets")
    if not isinstance(triplets, list) or not triplets:
        raise ValidationError("training config requires non-empty triplets")
    held_out = {str(value) for value in config.get("held_out_test_scenes", [])}
    selected_scenes = {str(value["scene"]) for value in triplets}
    overlap = held_out.intersection(selected_scenes)
    if overlap:
        raise ValidationError(f"training config overlaps held-out test scenes: {sorted(overlap)}")
    parameters = _camera_parameter_rows(camera_parameters_path, selected_scenes)
    result: list[HypersimTrainingTriplet] = []
    for triplet in triplets:
        split = str(triplet["split"])
        if split not in {"train", "val"}:
            raise ValidationError(f"unsupported training split: {split}")
        scene, camera_name = str(triplet["scene"]), str(triplet["camera"])
        detail = dataset_root / scene / "_detail" / camera_name
        frames = _load_hdf5(detail / "camera_keyframe_frame_indices.hdf5").astype(int)
        rotations = _load_hdf5(detail / "camera_keyframe_orientations.hdf5").astype(np.float64)
        positions = _load_hdf5(detail / "camera_keyframe_positions.hdf5").astype(np.float64)
        index_by_frame = {int(frame): index for index, frame in enumerate(frames)}
        scale = _scene_scale(dataset_root / scene / "_detail" / "metadata_scene.csv")
        parameter = parameters[scene]
        width, height = int(parameter["width"]), int(parameter["height"])
        intrinsics, rotation_adjustment = calibration_from_hypersim(
            parameter["matrix_camera_from_uv"], width, height
        )

        def load_rgb_depth(frame: int) -> tuple[np.ndarray, np.ndarray]:
            radial = _load_hdf5(_frame_path(dataset_root, scene, camera_name, frame, "depth_meters")).astype(
                np.float32
            )
            if radial.shape != (height, width):
                raise ValidationError(f"{scene}/{frame}: depth resolution mismatch")
            rgb, _ = tonemap_hypersim(
                _load_hdf5(_frame_path(dataset_root, scene, camera_name, frame, "color")),
                np.isfinite(radial) & (radial > 0),
            )
            return rgb, radial_depth_to_z(radial, intrinsics)

        source_frame = int(triplet["source_frame"])
        source_index = index_by_frame[source_frame]
        source_rgb, source_depth = load_rgb_depth(source_frame)
        source_camera = camera_from_hypersim(
            "center",
            rotations[source_index],
            positions[source_index],
            scale,
            intrinsics,
            angle_deg=0.0,
            opencv_camera_rotation_adjustment=rotation_adjustment,
        )
        observations = []
        for side in ("left", "right"):
            frame = int(triplet[f"{side}_frame"])
            index = index_by_frame[frame]
            rgb, depth = load_rgb_depth(frame)
            camera = camera_from_hypersim(
                side,
                rotations[index],
                positions[index],
                scale,
                intrinsics,
                angle_deg=float(triplet[f"{side}_yaw_deg"]),
                opencv_camera_rotation_adjustment=rotation_adjustment,
            )
            observations.append(
                TargetRGBDObservation(
                    observation_id=f"{scene}-{camera_name}-{side}-{frame:04d}",
                    rgb_uint8=rgb,
                    depth_z_float32=depth,
                    intrinsics_3x3_float64=camera.intrinsics_3x3_float64,
                    world_to_camera_4x4_float64=camera.world_to_camera_4x4_float64,
                )
            )
        result.append(
            HypersimTrainingTriplet(
                split=split,
                scene=scene,
                camera_name=camera_name,
                source_frame_index=source_frame,
                source_rgb_uint8=source_rgb,
                source_depth_z_label_only_float32=source_depth,
                source_camera=source_camera,
                target_observations=tuple(observations),
            )
        )
    return tuple(result)
