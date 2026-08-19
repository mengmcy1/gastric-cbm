"""Leakage-safe directed-pair preparation for the Stage 1.9 source-feature model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from adaptive3dgs.datasets.hypersim import (
    _camera_parameter_rows,
    _frame_path,
    _load_hdf5,
    _scene_scale,
    calibration_from_hypersim,
    camera_from_hypersim,
    radial_depth_to_z,
    tonemap_hypersim,
)
from adaptive3dgs.target_view import (
    forward_splat_source_to_target_with_grid,
    scale_intrinsics,
    target_camera_conditioning_in_source,
)
from stage1_8_target_view_common import _resize, portable, sha256, target_metrics


@dataclass
class PreparedDirectedSample:
    split: str
    scene: str
    camera: str
    source_role: str
    target_role: str
    source_frame: int
    target_frame: int
    yaw_deg: float
    source_rgb: torch.Tensor
    target_to_source_grid: torch.Tensor
    warped_rgb: torch.Tensor
    warped_depth: torch.Tensor
    warp_valid: torch.Tensor
    rays_in_source: torch.Tensor
    origin_in_source: torch.Tensor
    source_depth_scale: torch.Tensor
    target_rgb: torch.Tensor
    target_depth: torch.Tensor
    observed: torch.Tensor
    occluded: torch.Tensor
    outside: torch.Tensor
    input_hashes: dict[str, str]

    @property
    def side(self) -> str:
        return self.target_role

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.source_rgb,
            self.target_to_source_grid,
            self.warped_rgb,
            self.warped_depth,
            self.warp_valid,
            self.rays_in_source,
            self.origin_in_source,
            self.source_depth_scale,
        )


def _load_records(paths: tuple[Path, ...], required_key: str) -> dict[tuple[str, str, int], dict]:
    result: dict[tuple[str, str, int], dict] = {}
    for record_path in paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "pass":
            raise RuntimeError(f"input record is not pass: {record_path}")
        for item in record["results"]:
            key = (str(item["scene"]), str(item["camera"]), int(item["source_frame"]))
            if key in result:
                raise RuntimeError(f"duplicate frozen input {key}")
            if required_key not in item:
                raise RuntimeError(f"{required_key} missing for {key}")
            result[key] = item
    return result


def _image_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).permute(2, 0, 1)[None].float().to(device)


def _map_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value)[None, None].float().to(device)


def prepare_directed_samples(
    pair_config_path: Path,
    dataset_root: Path,
    camera_parameters_path: Path,
    rgb_record_paths: tuple[Path, ...],
    base_record_paths: tuple[Path, ...],
    size: tuple[int, int],
    device: torch.device,
    *,
    selected: set[tuple[str, int, int]],
) -> list[PreparedDirectedSample]:
    config = json.loads(pair_config_path.read_text(encoding="utf-8"))
    if config.get("status") != "pass":
        raise RuntimeError("directed-pair config must have status=pass")
    pairs = [
        pair for pair in config["pairs"]
        if (str(pair["scene"]), int(pair["source_frame"]), int(pair["target_frame"])) in selected
    ]
    if len(pairs) != len(selected):
        found = {(str(p["scene"]), int(p["source_frame"]), int(p["target_frame"])) for p in pairs}
        raise RuntimeError(f"selected directed pairs missing: {sorted(selected.difference(found))}")
    rgb_by_key = _load_records(rgb_record_paths, "arrays")
    base_by_key = _load_records(base_record_paths, "arrays")
    parameters = _camera_parameter_rows(camera_parameters_path, {str(pair["scene"]) for pair in pairs})
    target_height, target_width = size
    result: list[PreparedDirectedSample] = []
    scene_cache: dict[tuple[str, str], tuple[dict[int, int], np.ndarray, np.ndarray, float]] = {}
    for pair in pairs:
        scene, camera = str(pair["scene"]), str(pair["camera"])
        source_frame, target_frame = int(pair["source_frame"]), int(pair["target_frame"])
        source_key, target_key = (scene, camera, source_frame), (scene, camera, target_frame)
        if source_key not in rgb_by_key or source_key not in base_by_key:
            raise RuntimeError(f"frozen RGB/BaseDepth coverage missing for {source_key} -> {target_key}")

        def load_frozen_rgb(key: tuple[str, str, int]) -> tuple[np.ndarray, Path]:
            item = rgb_by_key[key]
            path = Path(item["arrays"])
            if sha256(path) != item["arrays_sha256"]:
                raise RuntimeError(f"RGB evidence hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as values:
                if "source_rgb_uint8" not in values.files:
                    raise RuntimeError(f"source_rgb_uint8 missing: {path}")
                rgb = values["source_rgb_uint8"].astype(np.float32) / 255.0
            return rgb, path

        source_rgb_full, source_rgb_path = load_frozen_rgb(source_key)
        target_color_path = _frame_path(dataset_root, scene, camera, target_frame, "color")
        target_depth_path = _frame_path(dataset_root, scene, camera, target_frame, "depth_meters")
        target_depth_radial_for_tonemap = _load_hdf5(target_depth_path).astype(np.float32)
        target_rgb_full, _ = tonemap_hypersim(
            _load_hdf5(target_color_path),
            np.isfinite(target_depth_radial_for_tonemap) & (target_depth_radial_for_tonemap > 0),
        )
        target_rgb_full = target_rgb_full.astype(np.float32) / 255.0
        base_item = base_by_key[source_key]
        base_path = Path(base_item["arrays"])
        if sha256(base_path) != base_item["arrays_sha256"]:
            raise RuntimeError(f"BaseDepth hash mismatch: {base_path}")
        with np.load(base_path, allow_pickle=False) as values:
            if "base_depth_z_float32" not in values.files:
                raise RuntimeError(f"base_depth_z_float32 missing: {base_path}")
            base_full = values["base_depth_z_float32"].astype(np.float32)
        source_height, source_width = base_full.shape
        if source_rgb_full.shape[:2] != (source_height, source_width):
            raise RuntimeError(f"RGB/BaseDepth shape mismatch for {source_key}")

        parameter = parameters[scene]
        original_width, original_height = int(parameter["width"]), int(parameter["height"])
        if (original_height, original_width) != (source_height, source_width):
            raise RuntimeError(f"camera/RGB shape mismatch for {source_key}")
        intrinsics, rotation_adjustment = calibration_from_hypersim(
            parameter["matrix_camera_from_uv"], original_width, original_height
        )
        cache_key = (scene, camera)
        if cache_key not in scene_cache:
            detail = dataset_root / scene / "_detail" / camera
            frames = _load_hdf5(detail / "camera_keyframe_frame_indices.hdf5").astype(int)
            rotations = _load_hdf5(detail / "camera_keyframe_orientations.hdf5").astype(np.float64)
            positions = _load_hdf5(detail / "camera_keyframe_positions.hdf5").astype(np.float64)
            scene_cache[cache_key] = (
                {int(frame): index for index, frame in enumerate(frames)}, rotations, positions,
                _scene_scale(dataset_root / scene / "_detail" / "metadata_scene.csv"),
            )
        index_by_frame, rotations, positions, scene_scale = scene_cache[cache_key]
        source_camera = camera_from_hypersim(
            "source", rotations[index_by_frame[source_frame]], positions[index_by_frame[source_frame]],
            scene_scale, intrinsics, angle_deg=0.0,
            opencv_camera_rotation_adjustment=rotation_adjustment,
        )
        target_camera = camera_from_hypersim(
            "target", rotations[index_by_frame[target_frame]], positions[index_by_frame[target_frame]],
            scene_scale, intrinsics, angle_deg=float(pair["yaw_deg"]),
            opencv_camera_rotation_adjustment=rotation_adjustment,
        )
        resized_source_rgb = _resize(source_rgb_full, size, "bilinear").astype(np.float32)
        resized_target_rgb = _resize(target_rgb_full, size, "bilinear").astype(np.float32)
        resized_base = _resize(base_full, size, "bilinear").astype(np.float32)
        resized_k = scale_intrinsics(intrinsics, (source_width, source_height), (target_width, target_height))
        warped_rgb, warped_depth, warp_valid, grid = forward_splat_source_to_target_with_grid(
            resized_source_rgb, resized_base, resized_k, source_camera.world_to_camera_4x4_float64,
            resized_k, target_camera.world_to_camera_4x4_float64,
        )
        rays, origin = target_camera_conditioning_in_source(
            target_height, target_width, resized_k, source_camera.world_to_camera_4x4_float64,
            target_camera.world_to_camera_4x4_float64,
        )
        visibility_path = Path(pair["arrays"])
        if sha256(visibility_path) != pair["arrays_sha256"]:
            raise RuntimeError(f"visibility hash mismatch: {visibility_path}")
        with np.load(visibility_path, allow_pickle=False) as values:
            target_depth_full = radial_depth_to_z(values["target_depth_meters_float32"], intrinsics)
            observed = _resize(values["observed_from_source_uint8"].astype(np.float32), size, "nearest") > 0.5
            occluded = _resize(values["occlusion_hidden_uint8"].astype(np.float32), size, "nearest") > 0.5
            outside = _resize(values["outside_source_fov_uint8"].astype(np.float32), size, "nearest") > 0.5
        target_depth = _resize(target_depth_full, size, "nearest").astype(np.float32)
        if np.any(observed & occluded) or np.any(observed & outside) or np.any(occluded & outside):
            raise RuntimeError(f"visibility masks overlap: {source_key} -> {target_frame}")
        valid_base = np.isfinite(resized_base) & (resized_base > 0)
        if not valid_base.any():
            raise RuntimeError(f"no valid frozen BaseDepth: {source_key}")
        depth_scale = float(np.median(resized_base[valid_base]))
        result.append(PreparedDirectedSample(
            split=str(pair["split"]), scene=scene, camera=camera,
            source_role=str(pair["source_role"]), target_role=str(pair["target_role"]),
            source_frame=source_frame, target_frame=target_frame, yaw_deg=float(pair["yaw_deg"]),
            source_rgb=_image_tensor(resized_source_rgb, device),
            target_to_source_grid=torch.from_numpy(grid)[None].float().to(device),
            warped_rgb=_image_tensor(warped_rgb, device), warped_depth=_map_tensor(warped_depth, device),
            warp_valid=_map_tensor(warp_valid.astype(np.float32), device),
            rays_in_source=_image_tensor(rays, device), origin_in_source=_image_tensor(origin, device),
            source_depth_scale=torch.tensor([[[[depth_scale]]]], dtype=torch.float32, device=device),
            target_rgb=_image_tensor(resized_target_rgb, device), target_depth=_map_tensor(target_depth, device),
            observed=_map_tensor(observed.astype(np.float32), device).bool(),
            occluded=_map_tensor(occluded.astype(np.float32), device).bool(),
            outside=_map_tensor(outside.astype(np.float32), device).bool(),
            input_hashes={
                "source_rgb_arrays": portable(source_rgb_path), "source_rgb_arrays_sha256": sha256(source_rgb_path),
                "target_color_label": portable(target_color_path), "target_color_label_sha256": sha256(target_color_path),
                "target_depth_label": portable(target_depth_path), "target_depth_label_sha256": sha256(target_depth_path),
                "base_arrays": portable(base_path), "base_arrays_sha256": sha256(base_path),
                "visibility_arrays": portable(visibility_path), "visibility_arrays_sha256": sha256(visibility_path),
            },
        ))
    return result


__all__ = ["PreparedDirectedSample", "prepare_directed_samples", "portable", "sha256", "target_metrics"]
