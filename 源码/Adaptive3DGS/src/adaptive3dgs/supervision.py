"""Positive-only multiview evidence for source-anchored hidden surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from .validation import ValidationError


@dataclass(frozen=True, slots=True)
class TargetRGBDObservation:
    observation_id: str
    rgb_uint8: npt.NDArray[np.uint8]
    depth_z_float32: npt.NDArray[np.float32]
    intrinsics_3x3_float64: npt.NDArray[np.float64]
    world_to_camera_4x4_float64: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HiddenLayerEvidence:
    """Nearest observed surface behind the source first hit.

    Pixels outside ``positive_mask_uint8`` are unknown, not negative support
    labels.  Sparse target RGB-D views cannot prove that a source ray has no
    additional surface behind its first hit.
    """

    depth_z_float32: npt.NDArray[np.float32]
    rgb_uint8: npt.NDArray[np.uint8]
    positive_mask_uint8: npt.NDArray[np.uint8]
    observation_count_uint16: npt.NDArray[np.uint16]
    depth_spread_float32: npt.NDArray[np.float32]
    source_view_coverage_uint8: npt.NDArray[np.uint8]


def _camera_matrix(value: npt.ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValidationError(f"{name} must be a finite {shape} matrix")
    return matrix


def _validate_observation(observation: TargetRGBDObservation) -> tuple[int, int]:
    depth = observation.depth_z_float32
    if depth.ndim != 2 or depth.dtype != np.float32:
        raise ValidationError(f"{observation.observation_id}: depth must be float32 HxW")
    height, width = depth.shape
    if observation.rgb_uint8.shape != (height, width, 3) or observation.rgb_uint8.dtype != np.uint8:
        raise ValidationError(f"{observation.observation_id}: RGB must be uint8 HxWx3")
    _camera_matrix(observation.intrinsics_3x3_float64, (3, 3), "target intrinsics")
    _camera_matrix(observation.world_to_camera_4x4_float64, (4, 4), "target extrinsics")
    return height, width


def build_source_hidden_evidence(
    source_depth_z_float32: npt.NDArray[np.float32],
    source_intrinsics_3x3_float64: npt.NDArray[np.float64],
    source_world_to_camera_4x4_float64: npt.NDArray[np.float64],
    observations: Sequence[TargetRGBDObservation],
    *,
    relative_behind_margin: float = 0.01,
    absolute_behind_margin: float = 0.01,
) -> HiddenLayerEvidence:
    """Fuse target first-hit RGB-D into positive hidden-surface evidence.

    Target points are transformed into the source camera and nearest-splatted
    onto source pixels.  A point is accepted only when it lies behind the
    source first hit by both the relative and absolute margins.  If several
    targets support one ray, the nearest hidden point is retained while count
    and depth spread preserve agreement diagnostics.
    """

    source_depth = np.asarray(source_depth_z_float32)
    if source_depth.ndim != 2 or source_depth.dtype != np.float32:
        raise ValidationError("source depth must be float32 HxW")
    if not observations:
        raise ValidationError("at least one target RGB-D observation is required")
    if relative_behind_margin < 0 or absolute_behind_margin < 0:
        raise ValueError("behind margins must be non-negative")
    source_k = _camera_matrix(source_intrinsics_3x3_float64, (3, 3), "source intrinsics")
    source_w2c = _camera_matrix(source_world_to_camera_4x4_float64, (4, 4), "source extrinsics")
    source_h, source_w = source_depth.shape
    pixel_count = source_h * source_w

    selected_depth = np.full(pixel_count, np.inf, dtype=np.float64)
    maximum_depth = np.full(pixel_count, -np.inf, dtype=np.float64)
    selected_rgb = np.zeros((pixel_count, 3), dtype=np.uint8)
    observation_count = np.zeros(pixel_count, dtype=np.uint32)
    source_view_coverage = np.zeros(pixel_count, dtype=bool)

    for observation in observations:
        target_h, target_w = _validate_observation(observation)
        target_k = observation.intrinsics_3x3_float64
        target_c2w = np.linalg.inv(observation.world_to_camera_4x4_float64)
        yy, xx = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing="ij")
        target_z = observation.depth_z_float32.astype(np.float64)
        target_valid = np.isfinite(target_z) & (target_z > 0)
        target_pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float64)
        target_rays = target_pixels @ np.linalg.inv(target_k).T
        target_points_xyz = target_rays * (target_z / target_rays[:, :, 2])[:, :, None]
        target_points = np.concatenate((target_points_xyz, np.ones((*target_z.shape, 1))), axis=2)
        world_points = target_points @ target_c2w.T
        source_points = world_points @ source_w2c.T
        source_z = source_points[:, :, 2]
        projected_homogeneous = source_points[:, :, :3] @ source_k.T
        with np.errstate(divide="ignore", invalid="ignore"):
            projected_x = projected_homogeneous[:, :, 0] / projected_homogeneous[:, :, 2]
            projected_y = projected_homogeneous[:, :, 1] / projected_homogeneous[:, :, 2]
        finite_projection = np.isfinite(projected_x) & np.isfinite(projected_y)
        source_x = np.full(projected_x.shape, -1, dtype=np.int64)
        source_y = np.full(projected_y.shape, -1, dtype=np.int64)
        source_x[finite_projection] = np.rint(projected_x[finite_projection]).astype(np.int64)
        source_y[finite_projection] = np.rint(projected_y[finite_projection]).astype(np.int64)
        projects_inside = (
            target_valid
            & np.isfinite(source_z)
            & (source_z > 0)
            & finite_projection
            & (source_x >= 0)
            & (source_x < source_w)
            & (source_y >= 0)
            & (source_y < source_h)
        )
        if not np.any(projects_inside):
            continue

        flat_source = source_y[projects_inside] * source_w + source_x[projects_inside]
        source_view_coverage[flat_source] = True
        front = source_depth.reshape(-1)[flat_source].astype(np.float64)
        candidate_z = source_z[projects_inside]
        hidden = (
            np.isfinite(front)
            & (front > 0)
            & (candidate_z > front + absolute_behind_margin)
            & (candidate_z > front * (1.0 + relative_behind_margin))
        )
        if not np.any(hidden):
            continue
        indices = flat_source[hidden]
        depths = candidate_z[hidden]
        colors = observation.rgb_uint8[projects_inside][hidden]
        np.add.at(observation_count, indices, 1)
        np.maximum.at(maximum_depth, indices, depths)
        order = np.lexsort((depths, indices))
        sorted_indices = indices[order]
        first_per_pixel = np.empty(len(order), dtype=bool)
        first_per_pixel[0] = True
        first_per_pixel[1:] = sorted_indices[1:] != sorted_indices[:-1]
        nearest_candidates = order[first_per_pixel]
        nearest_indices = indices[nearest_candidates]
        nearer_than_previous = depths[nearest_candidates] < selected_depth[nearest_indices]
        update_candidates = nearest_candidates[nearer_than_previous]
        update_indices = indices[update_candidates]
        selected_depth[update_indices] = depths[update_candidates]
        selected_rgb[update_indices] = colors[update_candidates]

    positive = observation_count > 0
    # Re-apply the protocol after float32 serialization. A float64 candidate
    # infinitesimally above the margin can round exactly onto the forbidden
    # boundary and must become unknown rather than a non-strict positive.
    quantized_depth = selected_depth.astype(np.float32)
    source_flat = source_depth.reshape(-1)
    positive &= np.isfinite(quantized_depth)
    absolute_threshold_float32 = source_flat + np.float32(absolute_behind_margin)
    relative_threshold_float32 = source_flat * np.float32(1.0 + relative_behind_margin)
    positive &= quantized_depth > absolute_threshold_float32
    positive &= quantized_depth > relative_threshold_float32
    observation_count[~positive] = 0
    selected_rgb[~positive] = 0
    depth = np.full(pixel_count, np.nan, dtype=np.float32)
    depth[positive] = quantized_depth[positive]
    spread = np.full(pixel_count, np.nan, dtype=np.float32)
    spread[positive] = (maximum_depth[positive] - selected_depth[positive]).astype(np.float32)
    return HiddenLayerEvidence(
        depth_z_float32=depth.reshape(source_h, source_w),
        rgb_uint8=selected_rgb.reshape(source_h, source_w, 3),
        positive_mask_uint8=positive.reshape(source_h, source_w).astype(np.uint8),
        observation_count_uint16=np.minimum(observation_count, np.iinfo(np.uint16).max)
        .astype(np.uint16)
        .reshape(source_h, source_w),
        depth_spread_float32=spread.reshape(source_h, source_w),
        source_view_coverage_uint8=source_view_coverage.reshape(source_h, source_w).astype(np.uint8),
    )
