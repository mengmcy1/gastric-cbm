"""Target-camera geometry inputs and deterministic source-to-target splatting."""

from __future__ import annotations

import numpy as np

from .validation import ValidationError


def scale_intrinsics(intrinsics: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    source_width, source_height = source_wh
    target_width, target_height = target_wh
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    result = np.asarray(intrinsics, dtype=np.float64).copy()
    if result.shape != (3, 3):
        raise ValidationError("intrinsics must be 3x3")
    result[0] *= target_width / source_width
    result[1] *= target_height / source_height
    return result


def forward_splat_source_to_target(
    source_rgb_float32: np.ndarray,
    source_depth_z_float32: np.ndarray,
    source_intrinsics_3x3_float64: np.ndarray,
    source_world_to_camera_4x4_float64: np.ndarray,
    target_intrinsics_3x3_float64: np.ndarray,
    target_world_to_camera_4x4_float64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-pixel z-buffer splat of source RGB-D into the requested target camera."""

    warped_rgb, warped_depth, warped_valid, _ = forward_splat_source_to_target_with_grid(
        source_rgb_float32,
        source_depth_z_float32,
        source_intrinsics_3x3_float64,
        source_world_to_camera_4x4_float64,
        target_intrinsics_3x3_float64,
        target_world_to_camera_4x4_float64,
    )
    return warped_rgb, warped_depth, warped_valid


def forward_splat_source_to_target_with_grid(
    source_rgb_float32: np.ndarray,
    source_depth_z_float32: np.ndarray,
    source_intrinsics_3x3_float64: np.ndarray,
    source_world_to_camera_4x4_float64: np.ndarray,
    target_intrinsics_3x3_float64: np.ndarray,
    target_world_to_camera_4x4_float64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward splat RGB-D and return the z-winner target-to-source grid."""

    rgb = np.asarray(source_rgb_float32, dtype=np.float32)
    depth = np.asarray(source_depth_z_float32, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValidationError("source RGB-D must be HxWx3 and HxW")
    if not np.isfinite(rgb).all() or not ((rgb >= 0) & (rgb <= 1)).all():
        raise ValidationError("source RGB must be finite in [0,1]")
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3).astype(np.float64)
    rays = pixels @ np.linalg.inv(np.asarray(source_intrinsics_3x3_float64, dtype=np.float64)).T
    source_xyz = rays * depth.reshape(-1, 1)
    source_to_target = np.asarray(target_world_to_camera_4x4_float64, dtype=np.float64) @ np.linalg.inv(
        np.asarray(source_world_to_camera_4x4_float64, dtype=np.float64)
    )
    target_xyz = source_xyz @ source_to_target[:3, :3].T + source_to_target[:3, 3]
    target_z = target_xyz[:, 2]
    projected = target_xyz @ np.asarray(target_intrinsics_3x3_float64, dtype=np.float64).T
    target_x = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
    target_y = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
    valid = np.isfinite(source_xyz).all(axis=1) & np.isfinite(target_xyz).all(axis=1)
    valid &= np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0) & (target_z > 0)
    valid &= (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
    source_indices = np.nonzero(valid)[0]
    target_indices = target_y[valid] * width + target_x[valid]
    order = np.lexsort((target_z[valid], target_indices))
    ordered_targets = target_indices[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = ordered_targets[1:] != ordered_targets[:-1]
    chosen_source = source_indices[order[keep]]
    chosen_target = ordered_targets[keep]
    warped_rgb = np.zeros((height * width, 3), dtype=np.float32)
    warped_depth = np.zeros(height * width, dtype=np.float32)
    warped_valid = np.zeros(height * width, dtype=np.uint8)
    source_grid = np.full((height * width, 2), -2.0, dtype=np.float32)
    warped_rgb[chosen_target] = rgb.reshape(-1, 3)[chosen_source]
    warped_depth[chosen_target] = target_z[chosen_source].astype(np.float32)
    warped_valid[chosen_target] = 1
    source_x = chosen_source % width
    source_y = chosen_source // width
    source_grid[chosen_target, 0] = 2.0 * source_x / max(width - 1, 1) - 1.0
    source_grid[chosen_target, 1] = 2.0 * source_y / max(height - 1, 1) - 1.0
    return (
        warped_rgb.reshape(height, width, 3),
        warped_depth.reshape(height, width),
        warped_valid.reshape(height, width),
        source_grid.reshape(height, width, 2),
    )


def target_camera_conditioning_in_source(
    height: int,
    width: int,
    target_intrinsics_3x3_float64: np.ndarray,
    source_world_to_camera_4x4_float64: np.ndarray,
    target_world_to_camera_4x4_float64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-target-pixel ray directions and camera origin in source coordinates."""

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float64)
    target_rays = pixels @ np.linalg.inv(np.asarray(target_intrinsics_3x3_float64, dtype=np.float64)).T
    source_from_target = np.asarray(source_world_to_camera_4x4_float64, dtype=np.float64) @ np.linalg.inv(
        np.asarray(target_world_to_camera_4x4_float64, dtype=np.float64)
    )
    directions = target_rays @ source_from_target[:3, :3].T
    directions /= np.linalg.norm(directions, axis=2, keepdims=True).clip(1e-12)
    origin = np.broadcast_to(source_from_target[:3, 3], (height, width, 3)).copy()
    return directions.astype(np.float32), origin.astype(np.float32)
