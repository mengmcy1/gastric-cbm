"""HLP-GEO-01 adapter with explicit Hypersim-to-OpenCV conversion."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..clb import Camera
from ..evaluation import GeometryTarget
from ..providers import ProviderContext
from ..validation import ValidationError


HYPERSIM_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True, slots=True)
class HypersimTargetView:
    target_id: str
    scene: str
    camera_name: str
    side: str
    frame_index: int
    camera: Camera
    rgb_uint8: np.ndarray
    geometry: GeometryTarget
    yaw_deg: float
    translation_m: float


@dataclass(frozen=True, slots=True)
class HypersimSample:
    scene: str
    camera_name: str
    source_frame_index: int
    context: ProviderContext
    targets: tuple[HypersimTargetView, ...]


def _load_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "dataset" not in handle:
            raise ValidationError(f"HDF5 dataset key missing: {path}")
        return handle["dataset"][:]


def _frame_path(root: Path, scene: str, camera: str, frame: int, suffix: str) -> Path:
    group = "final_hdf5" if suffix == "color" else "geometry_hdf5"
    return root / scene / "images" / f"scene_{camera}_{group}" / f"frame.{frame:04d}.{suffix}.hdf5"


def _scene_scale(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = {row[0]: row[1] for row in csv.reader(stream) if len(row) >= 2}
    try:
        scale = float(values["meters_per_asset_unit"])
    except KeyError as error:
        raise ValidationError(f"meters_per_asset_unit missing: {path}") from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValidationError(f"invalid meters_per_asset_unit: {scale}")
    return scale


def _camera_parameter_rows(path: Path, scenes: set[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
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
                "matrix_camera_from_uv": matrix,
            }
    missing = scenes.difference(result)
    if missing:
        raise ValidationError(f"camera parameters missing scenes: {sorted(missing)}")
    return result


def _rq_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transposed_q, transposed_r = np.linalg.qr(np.flipud(matrix).T)
    upper = np.flipud(transposed_r.T)
    upper = np.fliplr(upper)
    rotation = transposed_q.T
    rotation = np.flipud(rotation)
    signs = np.sign(np.diag(upper))
    signs[signs == 0] = 1
    correction = np.diag(signs)
    upper = upper @ correction
    rotation = correction @ rotation
    if np.linalg.det(rotation) < 0:
        upper = -upper
        rotation = -rotation
    return upper, rotation


def calibration_from_hypersim(
    matrix_camera_from_uv: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return standard K and a rotation that absorbs a tilted image plane."""

    matrix = np.asarray(matrix_camera_from_uv, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValidationError("M_cam_from_uv must be a finite invertible 3x3 matrix")
    pixel_to_uv = np.asarray(
        [[2.0 / width, 0.0, 1.0 / width - 1.0], [0.0, -2.0 / height, 1.0 - 1.0 / height], [0, 0, 1]],
        dtype=np.float64,
    )
    pixel_to_original_opencv_ray = HYPERSIM_TO_OPENCV @ matrix @ pixel_to_uv
    projection = np.linalg.inv(pixel_to_original_opencv_ray)
    intrinsics, rotation_adjustment = _rq_decomposition(projection)
    intrinsics /= intrinsics[2, 2]
    if not (
        np.isfinite(intrinsics).all()
        and np.isfinite(rotation_adjustment).all()
        and intrinsics[0, 0] > 0
        and intrinsics[1, 1] > 0
        and np.allclose(intrinsics[2], [0, 0, 1], atol=1e-9)
        and np.allclose(rotation_adjustment @ rotation_adjustment.T, np.eye(3), atol=1e-9)
        and np.linalg.det(rotation_adjustment) > 0
    ):
        raise ValidationError("could not decompose Hypersim projection into K and rotation")
    return intrinsics, rotation_adjustment


def intrinsics_from_hypersim(matrix_camera_from_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    intrinsics, _ = calibration_from_hypersim(matrix_camera_from_uv, width, height)
    return intrinsics


def camera_from_hypersim(
    camera_id: str,
    rotation_world_from_hypersim: np.ndarray,
    position_asset: np.ndarray,
    meters_per_asset_unit: float,
    intrinsics: np.ndarray,
    *,
    angle_deg: float | None,
    opencv_camera_rotation_adjustment: np.ndarray | None = None,
) -> Camera:
    adjustment = (
        np.eye(3, dtype=np.float64)
        if opencv_camera_rotation_adjustment is None
        else np.asarray(opencv_camera_rotation_adjustment, dtype=np.float64)
    )
    if adjustment.shape != (3, 3) or not np.allclose(adjustment @ adjustment.T, np.eye(3), atol=1e-9):
        raise ValidationError("OpenCV camera rotation adjustment must be a 3x3 rotation")
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = (
        np.asarray(rotation_world_from_hypersim, dtype=np.float64)
        @ HYPERSIM_TO_OPENCV
        @ adjustment.T
    )
    camera_to_world[:3, 3] = np.asarray(position_asset, dtype=np.float64) * meters_per_asset_unit
    world_to_camera = np.linalg.inv(camera_to_world)
    return Camera(
        camera_id=camera_id,
        intrinsics_3x3_float64=np.asarray(intrinsics, dtype=np.float64).copy(),
        world_to_camera_4x4_float64=world_to_camera,
        angle_deg=angle_deg,
        metadata={"source_convention": "Hypersim right/up/back", "target_convention": "OpenCV right/down/forward"},
    )


def radial_depth_to_z(radial_depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    radial = np.asarray(radial_depth, dtype=np.float32)
    if radial.ndim != 2:
        raise ValidationError("radial depth must be HxW")
    height, width = radial.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float64)
    rays = pixels @ np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)).T
    ray_norm = np.linalg.norm(rays, axis=2)
    return (radial * rays[:, :, 2] / ray_norm).astype(np.float32)


def tonemap_hypersim(color_linear: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    color = np.asarray(color_linear, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(color).all(axis=2)
    if color.ndim != 3 or color.shape[2] != 3 or valid.shape != color.shape[:2]:
        raise ValidationError("Hypersim color/valid-mask shape mismatch")
    brightness = 0.3 * color[:, :, 0] + 0.59 * color[:, :, 1] + 0.11 * color[:, :, 2]
    percentile = float(np.percentile(brightness[valid], 90)) if np.any(valid) else 0.0
    if 0 < percentile < 0.0001:
        scale = 0.0
    elif percentile == 0:
        scale = 1.0
    else:
        scale = 0.8 ** 2.2 / percentile
    mapped = np.power(np.maximum(scale * color, 0), 1.0 / 2.2)
    return np.round(np.clip(mapped, 0, 1) * 255).astype(np.uint8), float(scale)


def _load_visibility(path: Path, intrinsics: np.ndarray) -> GeometryTarget:
    with np.load(path, allow_pickle=False) as values:
        required = {
            "observed_from_source_uint8",
            "occlusion_hidden_uint8",
            "outside_source_fov_uint8",
            "target_depth_meters_float32",
        }
        missing = sorted(required.difference(values.files))
        if missing:
            raise ValidationError(f"visibility truth missing arrays {missing}: {path}")
        radial = values["target_depth_meters_float32"].astype(np.float32)
        return GeometryTarget(
            depth_z_float32=radial_depth_to_z(radial, intrinsics),
            observed_from_source_mask=values["observed_from_source_uint8"].astype(bool),
            occlusion_hidden_mask=values["occlusion_hidden_uint8"].astype(bool),
            outside_source_fov_mask=values["outside_source_fov_uint8"].astype(bool),
        )


def load_hlp_geo_samples(
    config_path: str | Path,
    dataset_root: str | Path,
    camera_parameters_path: str | Path,
    visibility_root: str | Path,
    *,
    include_source_depth: bool = False,
) -> tuple[HypersimSample, ...]:
    config_path = Path(config_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    camera_parameters_path = Path(camera_parameters_path).resolve()
    visibility_root = Path(visibility_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    triplets = config.get("triplets")
    if not isinstance(triplets, list) or not triplets:
        raise ValidationError("HLP-GEO config requires non-empty triplets")
    parameters = _camera_parameter_rows(camera_parameters_path, {str(value["scene"]) for value in triplets})
    samples: list[HypersimSample] = []
    for triplet in triplets:
        scene, camera_name = str(triplet["scene"]), str(triplet["camera"])
        detail = dataset_root / scene / "_detail" / camera_name
        frame_indices = _load_hdf5(detail / "camera_keyframe_frame_indices.hdf5").astype(int)
        rotations = _load_hdf5(detail / "camera_keyframe_orientations.hdf5").astype(np.float64)
        positions = _load_hdf5(detail / "camera_keyframe_positions.hdf5").astype(np.float64)
        index_by_frame = {int(frame): index for index, frame in enumerate(frame_indices)}
        scale = _scene_scale(dataset_root / scene / "_detail" / "metadata_scene.csv")
        parameter = parameters[scene]
        width, height = int(parameter["width"]), int(parameter["height"])
        intrinsics = intrinsics_from_hypersim(parameter["matrix_camera_from_uv"], width, height)
        source_frame = int(triplet["source_frame"])
        source_index = index_by_frame[source_frame]
        source_depth_radial = _load_hdf5(
            _frame_path(dataset_root, scene, camera_name, source_frame, "depth_meters")
        ).astype(np.float32)
        if source_depth_radial.shape != (height, width):
            raise ValidationError(f"{scene}: source resolution does not match camera parameters")
        source_rgb, tonemap_scale = tonemap_hypersim(
            _load_hdf5(_frame_path(dataset_root, scene, camera_name, source_frame, "color")),
            np.isfinite(source_depth_radial) & (source_depth_radial > 0),
        )
        source_camera = camera_from_hypersim(
            "center", rotations[source_index], positions[source_index], scale, intrinsics, angle_deg=0.0
        )
        context = ProviderContext(
            sample_id=f"{scene}-{camera_name}-{source_frame:04d}",
            rgb_uint8=source_rgb,
            depth_z_float32=radial_depth_to_z(source_depth_radial, intrinsics) if include_source_depth else None,
            intrinsics_3x3_float64=source_camera.intrinsics_3x3_float64,
            world_to_camera_4x4_float64=source_camera.world_to_camera_4x4_float64,
            metadata={
                "dataset": "HLP-GEO-01",
                "scene": scene,
                "camera": camera_name,
                "frame_index": source_frame,
                "source_depth_truth_exposed": include_source_depth,
                "tonemap_scale": tonemap_scale,
            },
        )
        targets: list[HypersimTargetView] = []
        for side in ("left", "right"):
            frame = int(triplet[f"{side}_frame"])
            index = index_by_frame[frame]
            target_camera = camera_from_hypersim(
                side,
                rotations[index],
                positions[index],
                scale,
                intrinsics,
                angle_deg=float(triplet[f"{side}_yaw_deg"]),
            )
            target_depth_radial = _load_hdf5(
                _frame_path(dataset_root, scene, camera_name, frame, "depth_meters")
            ).astype(np.float32)
            target_rgb, _ = tonemap_hypersim(
                _load_hdf5(_frame_path(dataset_root, scene, camera_name, frame, "color")),
                np.isfinite(target_depth_radial) & (target_depth_radial > 0),
            )
            targets.append(
                HypersimTargetView(
                    target_id=f"{scene}-{camera_name}-{side}-{frame:04d}",
                    scene=scene,
                    camera_name=camera_name,
                    side=side,
                    frame_index=frame,
                    camera=target_camera,
                    rgb_uint8=target_rgb,
                    geometry=_load_visibility(visibility_root / scene / f"{side}_visibility_truth.npz", intrinsics),
                    yaw_deg=float(triplet[f"{side}_yaw_deg"]),
                    translation_m=float(triplet[f"{side}_translation_m"]),
                )
            )
        samples.append(
            HypersimSample(
                scene=scene,
                camera_name=camera_name,
                source_frame_index=source_frame,
                context=context,
                targets=tuple(targets),
            )
        )
    return tuple(samples)
