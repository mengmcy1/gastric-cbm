from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image
from scipy import ndimage

from stage01_common import (
    REPO_ROOT,
    camera_info_for_angle,
    create_camera_context,
    file_hash,
    prepare_output_dir,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将中心原图 RGB + 中心渲染深度前向重投影到 true_arc 端点并量化真实观测覆盖。"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_entry(entry: dict, label: str) -> Path:
    path = resolve_repo_path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    observed = file_hash(path)
    if observed != entry["sha256"].upper():
        raise ValueError(f"{label} SHA256 不匹配：{observed} != {entry['sha256']}")
    return path


def load_alpha(path: Path) -> np.ndarray:
    alpha = np.asarray(Image.open(path), dtype=np.float32)
    if alpha.max() > 1.0:
        alpha /= 65535.0
    return np.clip(alpha, 0.0, 1.0)


def unproject_to_world(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    camera_points = np.stack([x, y, depth], axis=1)
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    return (camera_points - translation) @ rotation


def project_world(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = points_world @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    z = camera_points[:, 2]
    safe = np.maximum(z, 1e-8)
    u = intrinsics[0, 0] * camera_points[:, 0] / safe + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera_points[:, 1] / safe + intrinsics[1, 2]
    return u, v, z


def neighbor_candidates(
    projected_u: np.ndarray,
    projected_v: np.ndarray,
    projected_z: np.ndarray,
    source_indices: np.ndarray,
    width: int,
    height: int,
):
    floor_u = np.floor(projected_u).astype(np.int32)
    floor_v = np.floor(projected_v).astype(np.int32)
    fraction_u = projected_u - floor_u
    fraction_v = projected_v - floor_v
    for offset_u, offset_v, weight in (
        (0, 0, (1.0 - fraction_u) * (1.0 - fraction_v)),
        (1, 0, fraction_u * (1.0 - fraction_v)),
        (0, 1, (1.0 - fraction_u) * fraction_v),
        (1, 1, fraction_u * fraction_v),
    ):
        target_u = floor_u + offset_u
        target_v = floor_v + offset_v
        valid = (
            (projected_z > 0)
            & np.isfinite(projected_u)
            & np.isfinite(projected_v)
            & np.isfinite(projected_z)
            & (target_u >= 0)
            & (target_u < width)
            & (target_v >= 0)
            & (target_v < height)
            & (weight > 1e-6)
        )
        yield (
            target_v[valid].astype(np.int64) * width + target_u[valid].astype(np.int64),
            projected_z[valid].astype(np.float32),
            weight[valid].astype(np.float32),
            source_indices[valid],
        )


def forward_splat(
    source_rgb: np.ndarray,
    source_alpha: np.ndarray,
    source_valid: np.ndarray,
    source_depth: np.ndarray,
    center_intrinsics: np.ndarray,
    center_extrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    target_extrinsics: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    height, width = source_depth.shape
    grid_v, grid_u = np.indices((height, width), dtype=np.float32)
    flat_valid = source_valid.ravel()
    source_indices = np.flatnonzero(flat_valid).astype(np.int64)
    u = grid_u.ravel()[flat_valid]
    v = grid_v.ravel()[flat_valid]
    depth = source_depth.ravel()[flat_valid]
    points_world = unproject_to_world(u, v, depth, center_intrinsics, center_extrinsics)

    identity_u, identity_v, identity_z = project_world(
        points_world, center_intrinsics, center_extrinsics
    )
    identity_error = np.hypot(identity_u - u, identity_v - v)
    identity_depth_error = np.abs(identity_z - depth)

    target_u, target_v, target_z = project_world(
        points_world, target_intrinsics, target_extrinsics
    )
    flat_size = height * width
    z_buffer = np.full(flat_size, np.inf, dtype=np.float32)
    for pixel, z, _, _ in neighbor_candidates(
        target_u, target_v, target_z, source_indices, width, height
    ):
        np.minimum.at(z_buffer, pixel, z)

    best_score = np.full(flat_size, -np.inf, dtype=np.float32)
    source_alpha_flat = source_alpha.ravel()
    for pixel, z, weight, source_index in neighbor_candidates(
        target_u, target_v, target_z, source_indices, width, height
    ):
        tolerance = np.maximum(0.01, 0.005 * z_buffer[pixel])
        near_front = z <= z_buffer[pixel] + tolerance
        score = weight * source_alpha_flat[source_index]
        np.maximum.at(best_score, pixel[near_front], score[near_front])

    winner = np.full(flat_size, -1, dtype=np.int64)
    winner_z = np.full(flat_size, np.inf, dtype=np.float32)
    winner_weight = np.zeros(flat_size, dtype=np.float32)
    for pixel, z, weight, source_index in neighbor_candidates(
        target_u, target_v, target_z, source_indices, width, height
    ):
        tolerance = np.maximum(0.01, 0.005 * z_buffer[pixel])
        score = weight * source_alpha_flat[source_index]
        choose = (z <= z_buffer[pixel] + tolerance) & (score >= best_score[pixel] - 1e-7)
        selected_pixel = pixel[choose]
        winner[selected_pixel] = source_index[choose]
        winner_z[selected_pixel] = z[choose]
        winner_weight[selected_pixel] = weight[choose]

    raw_coverage = winner >= 0
    reprojected_rgb = np.zeros((flat_size, 3), dtype=np.uint8)
    reprojected_rgb[raw_coverage] = source_rgb.reshape(-1, 3)[winner[raw_coverage]]
    source_u = np.full(flat_size, -1, dtype=np.int16)
    source_v = np.full(flat_size, -1, dtype=np.int16)
    source_u[raw_coverage] = (winner[raw_coverage] % width).astype(np.int16)
    source_v[raw_coverage] = (winner[raw_coverage] // width).astype(np.int16)
    winning_alpha = np.zeros(flat_size, dtype=np.float32)
    winning_alpha[raw_coverage] = source_alpha_flat[winner[raw_coverage]]

    return {
        "reprojected_rgb": reprojected_rgb.reshape(height, width, 3),
        "raw_coverage": raw_coverage.reshape(height, width),
        "reprojected_depth": winner_z.reshape(height, width),
        "bilinear_weight": winner_weight.reshape(height, width),
        "winning_source_alpha": winning_alpha.reshape(height, width),
        "source_u": source_u.reshape(height, width),
        "source_v": source_v.reshape(height, width),
        "source_point_count": int(source_indices.size),
        "identity_reprojection_max_px": float(identity_error.max(initial=0.0)),
        "identity_reprojection_p99_px": float(np.quantile(identity_error, 0.99)),
        "identity_depth_max_abs_m": float(identity_depth_error.max(initial=0.0)),
    }


def component_stats(mask: np.ndarray) -> dict:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=np.int64)
    return {
        "component_count": int(len(areas)),
        "largest_component_px": int(areas.max(initial=0)),
        "components_ge_64px": int((areas >= 64).sum()),
    }


def fraction(mask: np.ndarray, region: np.ndarray) -> float | None:
    denominator = int(region.sum())
    return float(mask[region].mean()) if denominator else None


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("experiment_id") != "P01_ang30_observed_reprojection_v1":
        raise ValueError("拒绝使用未冻结的重投影输入清单。")
    protocol = manifest["protocol"]
    output_dir = prepare_output_dir(args.output_dir)

    ply_path = verify_entry(manifest["base_ply"], "base PLY")
    source_rgb_path = verify_entry(manifest["source"]["rgb_original"], "source RGB")
    source_depth_path = verify_entry(manifest["source"]["depth_center"], "center depth")
    source_alpha_path = verify_entry(manifest["source"]["alpha_center"], "center alpha")
    render_config_path = verify_entry(manifest["render_config"], "render config")
    endpoint = manifest["endpoints"][args.side]
    target_frame_path = verify_entry(endpoint["frame"], f"{args.side} frame")
    target_depth_path = verify_entry(endpoint["depth"], f"{args.side} depth")
    target_alpha_path = verify_entry(endpoint["alpha"], f"{args.side} alpha")
    accepted_mask_path = verify_entry(endpoint["accepted_mask"], f"{args.side} accepted mask")

    source_depth = np.load(source_depth_path).astype(np.float32)
    source_alpha = load_alpha(source_alpha_path)
    height, width = source_depth.shape
    source_rgb_original = Image.open(source_rgb_path).convert("RGB")
    source_rgb = np.asarray(
        source_rgb_original.resize((width, height), Image.Resampling.LANCZOS)
    )
    target_frame = np.asarray(Image.open(target_frame_path).convert("RGB"))
    target_depth = np.load(target_depth_path).astype(np.float32)
    target_alpha = load_alpha(target_alpha_path)
    accepted = np.asarray(Image.open(accepted_mask_path).convert("L")) > 0
    for label, value in {
        "source alpha": source_alpha,
        "target frame": target_frame,
        "target depth": target_depth,
        "target alpha": target_alpha,
        "accepted mask": accepted,
    }.items():
        if value.shape[:2] != (height, width):
            raise ValueError(f"{label} 分辨率与中心深度不一致：{value.shape}")

    render_config = read_json(render_config_path)
    if render_config["trajectory_mode"] != "true_arc" or float(render_config["angle_total_deg"]) != 30.0:
        raise ValueError("渲染配置不是冻结的 true_arc 30°协议。")

    _, metadata, camera_model = create_camera_context(ply_path)
    _, center_camera = camera_info_for_angle(camera_model, 0.0, "true_arc")
    if (int(center_camera.width), int(center_camera.height)) != (width, height):
        raise ValueError(
            "相机模型输出分辨率与冻结渲染不一致："
            f"camera={(center_camera.width, center_camera.height)}, frozen={(width, height)}"
        )
    endpoint_angle = float(protocol["endpoint_angles_deg"][args.side])
    endpoint_eye, target_camera = camera_info_for_angle(camera_model, endpoint_angle, "true_arc")
    center_intrinsics = center_camera.intrinsics.cpu().numpy().astype(np.float64)
    center_extrinsics = center_camera.extrinsics.cpu().numpy().astype(np.float64)
    target_intrinsics = target_camera.intrinsics.cpu().numpy().astype(np.float64)
    target_extrinsics = target_camera.extrinsics.cpu().numpy().astype(np.float64)
    intrinsics_difference = float(np.abs(center_intrinsics - target_intrinsics).max())
    if intrinsics_difference > 1e-6:
        raise ValueError(f"端点内参发生变化：max diff={intrinsics_difference}")

    source_valid = (
        np.isfinite(source_depth)
        & (source_depth > 0)
        & (source_alpha >= float(protocol["source_alpha_threshold"]))
    )
    start = time.perf_counter()
    splat = forward_splat(
        source_rgb,
        source_alpha,
        source_valid,
        source_depth,
        center_intrinsics,
        center_extrinsics,
        target_intrinsics,
        target_extrinsics,
    )
    seconds = time.perf_counter() - start

    raw_coverage = splat["raw_coverage"]
    reprojected_depth = splat["reprojected_depth"]
    target_depth_valid = np.isfinite(target_depth) & (target_depth > 0)
    target_depth_reliable = target_depth_valid & (
        target_alpha >= float(protocol["target_occlusion_alpha_threshold"])
    )
    depth_tolerance = np.maximum(
        float(protocol["target_depth_absolute_tolerance_m"]),
        float(protocol["target_depth_relative_tolerance"]) * np.maximum(target_depth, 0.0),
    )
    target_occlusion_pass = (~target_depth_reliable) | (
        reprojected_depth <= target_depth + depth_tolerance
    )
    observable = raw_coverage & target_occlusion_pass
    depth_error = np.full((height, width), np.nan, dtype=np.float32)
    comparable = raw_coverage & target_depth_reliable
    depth_error[comparable] = np.abs(reprojected_depth[comparable] - target_depth[comparable])
    relative_depth_error = np.full((height, width), np.nan, dtype=np.float32)
    relative_depth_error[comparable] = depth_error[comparable] / np.maximum(
        target_depth[comparable], 1e-6
    )
    depth_consistent = comparable & (depth_error <= depth_tolerance)

    confidence = (
        splat["bilinear_weight"] * splat["winning_source_alpha"]
    ).astype(np.float32)
    reliable_factor = np.ones_like(confidence)
    reliable_factor[target_depth_reliable & raw_coverage] = np.exp(
        -np.nan_to_num(relative_depth_error[target_depth_reliable & raw_coverage], nan=10.0)
        / max(float(protocol["target_depth_relative_tolerance"]), 1e-6)
    )
    reliable_factor[~target_depth_reliable] = 0.5
    confidence *= reliable_factor
    confidence[~observable] = 0.0
    high_confidence = observable & (
        confidence >= float(protocol["high_confidence_threshold"])
    )
    residual = accepted & ~observable
    residual_high_confidence = accepted & ~high_confidence

    reprojected_rgb = splat["reprojected_rgb"]
    observed_only = np.zeros_like(reprojected_rgb)
    observed_only[observable] = reprojected_rgb[observable]
    accepted_composite = target_frame.copy()
    accepted_observable = accepted & observable
    accepted_composite[accepted_observable] = reprojected_rgb[accepted_observable]

    coverage_classes = np.zeros((height, width, 3), dtype=np.uint8)
    coverage_classes[accepted & high_confidence] = [0, 220, 0]
    coverage_classes[accepted & observable & ~high_confidence] = [255, 190, 0]
    coverage_classes[residual] = [230, 0, 0]
    overlay = np.round(target_frame.astype(np.float32) * 0.45 + coverage_classes * 0.55).astype(np.uint8)
    overlay[~accepted] = target_frame[~accepted]

    np.save(output_dir / "reprojected_depth_float32.npy", reprojected_depth.astype(np.float32))
    np.save(output_dir / "confidence_float32.npy", confidence)
    np.savez_compressed(
        output_dir / "source_coordinates_int16.npz",
        source_u=splat["source_u"],
        source_v=splat["source_v"],
    )
    iio.imwrite(output_dir / "reprojected_observed_rgb.png", observed_only)
    iio.imwrite(output_dir / "accepted_reprojection_composite.png", accepted_composite)
    iio.imwrite(output_dir / "coverage_raw.png", raw_coverage.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "coverage_observable.png", observable.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "coverage_high_confidence.png", high_confidence.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "residual_unobserved_mask.png", residual.astype(np.uint8) * 255)
    iio.imwrite(
        output_dir / "residual_not_high_confidence_mask.png",
        residual_high_confidence.astype(np.uint8) * 255,
    )
    iio.imwrite(output_dir / "accepted_coverage_classes.png", coverage_classes)
    iio.imwrite(output_dir / "accepted_coverage_overlay.png", overlay)

    known_reliable = (~accepted) & (target_alpha >= 0.95) & target_depth_valid
    accepted_count = int(accepted.sum())
    comparable_relative = relative_depth_error[comparable]
    result = {
        "schema_version": "1.0-observed-reprojection-result",
        "status": "success",
        "experiment_id": manifest["experiment_id"],
        "side": args.side,
        "endpoint_angle_deg": endpoint_angle,
        "endpoint_eye_xyz": [float(value) for value in endpoint_eye],
        "method": {
            "color_source": "original input.jpg resized from 4096x3072 to frozen 2048x1536 with Lanczos",
            "geometry_source": "center GSplat normalized Z-depth and center alpha",
            "projection": "center RGB-D unprojection to world, target-camera projection, 2x2 bilinear forward splat, nearest-depth z-buffer",
            "occlusion": "reject a projected point only when reliable target depth shows it lies behind the target surface beyond tolerance",
            "coverage_is_not_completion": True,
        },
        "camera_checks": {
            "intrinsics_max_abs_diff_center_vs_endpoint": intrinsics_difference,
            "identity_reprojection_max_px": splat["identity_reprojection_max_px"],
            "identity_reprojection_p99_px": splat["identity_reprojection_p99_px"],
            "identity_depth_max_abs_m": splat["identity_depth_max_abs_m"],
        },
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "base_ply_sha256": manifest["base_ply"]["sha256"],
            "source_rgb_sha256": manifest["source"]["rgb_original"]["sha256"],
            "source_depth_sha256": manifest["source"]["depth_center"]["sha256"],
            "source_alpha_sha256": manifest["source"]["alpha_center"]["sha256"],
            "target_depth_sha256": endpoint["depth"]["sha256"],
            "target_alpha_sha256": endpoint["alpha"]["sha256"],
            "accepted_mask_sha256": endpoint["accepted_mask"]["sha256"],
        },
        "counts": {
            "image_pixels": height * width,
            "source_valid_pixels": splat["source_point_count"],
            "accepted_pixels": accepted_count,
            "accepted_observable_pixels": int(accepted_observable.sum()),
            "accepted_high_confidence_pixels": int((accepted & high_confidence).sum()),
            "accepted_residual_unobserved_pixels": int(residual.sum()),
            "accepted_residual_not_high_confidence_pixels": int(residual_high_confidence.sum()),
        },
        "fractions": {
            "source_valid_fraction": float(source_valid.mean()),
            "full_frame_raw_coverage": float(raw_coverage.mean()),
            "full_frame_observable_coverage": float(observable.mean()),
            "known_reliable_observable_coverage": fraction(observable, known_reliable),
            "accepted_observable_coverage": fraction(observable, accepted),
            "accepted_high_confidence_coverage": fraction(high_confidence, accepted),
            "accepted_residual_unobserved": fraction(residual, accepted),
            "accepted_residual_not_high_confidence": fraction(residual_high_confidence, accepted),
            "target_depth_consistency_where_comparable": fraction(depth_consistent, comparable),
        },
        "depth_diagnostics": {
            "comparable_pixels": int(comparable.sum()),
            "relative_error_median": float(np.median(comparable_relative)) if comparable_relative.size else None,
            "relative_error_p95": float(np.quantile(comparable_relative, 0.95)) if comparable_relative.size else None,
        },
        "residual_components": component_stats(residual),
        "runtime": {"seconds": seconds, "device": "cpu"},
        "outputs": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.name == "reprojection_result.json":
            continue
        result["outputs"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
    write_json(output_dir / "reprojection_result.json", result)


if __name__ == "__main__":
    main()
