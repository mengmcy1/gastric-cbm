#!/usr/bin/env python3
"""Analyze center-view Gaussian density, footprint, and detail retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


VERTEX_FIELDS = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_infinisplat_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header ended before end_header")
            header.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only InfiniSplat binary_little_endian PLY is supported")
        vertex_line = next(line for line in header if line.startswith("element vertex "))
        count = int(vertex_line.split()[-1])
        dtype = np.dtype([(name, "<f4") for name in VERTEX_FIELDS])
        vertices = np.fromfile(stream, dtype=dtype, count=count)
        if vertices.shape[0] != count:
            raise ValueError(f"Expected {count} vertices, read {vertices.shape[0]}")
        extrinsic = np.fromfile(stream, dtype="<f4", count=16).reshape(4, 4)
        intrinsic = np.fromfile(stream, dtype="<f4", count=9).reshape(3, 3)
        width, height = np.fromfile(stream, dtype="<u4", count=2).tolist()
    return vertices, extrinsic, intrinsic, (int(width), int(height))


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64, copy=False)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    w, x, y, z = (q[:, index] for index in range(4))
    rotation = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def projected_radii(
    xyz: np.ndarray,
    log_scales: np.ndarray,
    quaternions: np.ndarray,
    intrinsic: np.ndarray,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    major = np.empty(xyz.shape[0], dtype=np.float32)
    minor = np.empty(xyz.shape[0], dtype=np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    for start in range(0, xyz.shape[0], chunk_size):
        end = min(start + chunk_size, xyz.shape[0])
        points = xyz[start:end].astype(np.float64, copy=False)
        scales = np.exp(log_scales[start:end].astype(np.float64, copy=False))
        rotation = quaternion_to_rotation(quaternions[start:end])
        scaled_rotation = rotation * scales[:, None, :]
        covariance = scaled_rotation @ np.transpose(scaled_rotation, (0, 2, 1))
        x, y, z = points[:, 0], points[:, 1], points[:, 2].clip(min=1e-6)
        jacobian = np.zeros((points.shape[0], 2, 3), dtype=np.float64)
        jacobian[:, 0, 0] = fx / z
        jacobian[:, 0, 2] = -fx * x / (z * z)
        jacobian[:, 1, 1] = fy / z
        jacobian[:, 1, 2] = -fy * y / (z * z)
        projected = jacobian @ covariance @ np.transpose(jacobian, (0, 2, 1))
        trace = projected[:, 0, 0] + projected[:, 1, 1]
        determinant = projected[:, 0, 0] * projected[:, 1, 1] - projected[:, 0, 1] ** 2
        discriminant = np.maximum(trace * trace - 4.0 * determinant, 0.0)
        eig_major = np.maximum((trace + np.sqrt(discriminant)) / 2.0, 0.0)
        eig_minor = np.maximum((trace - np.sqrt(discriminant)) / 2.0, 0.0)
        major[start:end] = np.sqrt(eig_major).astype(np.float32)
        minor[start:end] = np.sqrt(eig_minor).astype(np.float32)
    return major, minor


def grayscale_gradient(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    dy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    return dx + dy


def tile_mean(values: np.ndarray, tile: int) -> np.ndarray:
    height, width = values.shape
    tile_h, tile_w = height // tile, width // tile
    cropped = values[: tile_h * tile, : tile_w * tile]
    return cropped.reshape(tile_h, tile, tile_w, tile).mean(axis=(1, 3))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    rx, ry = rankdata(x[valid]), rankdata(y[valid])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {name: float(value) for name, value in zip(("p01", "p10", "p25", "p50", "p75", "p90", "p99"), q)}


def heatmap(values: np.ndarray, *, invert: bool = False) -> Image.Image:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        lo, hi = np.quantile(finite, [0.05, 0.95])
        normalized = np.clip((values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    normalized = np.nan_to_num(normalized, nan=0.0)
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["inputs"].items() if not key.endswith("sha256")}
    if sha256(paths["ply"]) != config["inputs"]["ply_sha256"]:
        raise ValueError("PLY SHA256 mismatch")
    if sha256(paths["source_rgb"]) != config["inputs"]["source_rgb_sha256"]:
        raise ValueError("source RGB SHA256 mismatch")

    vertices, extrinsic, intrinsic, (width, height) = load_infinisplat_ply(paths["ply"])
    if not np.allclose(extrinsic, np.eye(4), atol=1e-6):
        raise ValueError("This diagnostic requires the embedded center camera to be identity")
    xyz = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1)
    log_scales = np.stack([vertices[name] for name in ("scale_0", "scale_1", "scale_2")], axis=1)
    quaternions = np.stack([vertices[name] for name in ("rot_0", "rot_1", "rot_2", "rot_3")], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))

    z = xyz[:, 2]
    u = intrinsic[0, 0] * xyz[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * xyz[:, 1] / z + intrinsic[1, 2]
    in_frame = np.isfinite(u) & np.isfinite(v) & (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    xyz_f = xyz[in_frame]
    u_f, v_f, z_f = u[in_frame], v[in_frame], z[in_frame]
    opacity_f = opacity[in_frame]
    major, minor = projected_radii(xyz_f, log_scales[in_frame], quaternions[in_frame], intrinsic)

    tile = int(config["analysis"]["tile_size_px"])
    tile_w, tile_h = width // tile, height // tile
    tx = np.clip((u_f // tile).astype(np.int64), 0, tile_w - 1)
    ty = np.clip((v_f // tile).astype(np.int64), 0, tile_h - 1)
    flat_tile = ty * tile_w + tx
    tile_count = np.bincount(flat_tile, minlength=tile_h * tile_w).reshape(tile_h, tile_w)
    tile_depth_sum = np.bincount(flat_tile, weights=z_f, minlength=tile_h * tile_w)
    tile_radius_sum = np.bincount(flat_tile, weights=major, minlength=tile_h * tile_w)
    count_flat = tile_count.reshape(-1)
    tile_depth = np.divide(tile_depth_sum, count_flat, out=np.full_like(tile_depth_sum, np.nan), where=count_flat > 0).reshape(tile_h, tile_w)
    tile_radius = np.divide(tile_radius_sum, count_flat, out=np.full_like(tile_radius_sum, np.nan), where=count_flat > 0).reshape(tile_h, tile_w)

    source = ImageOps.exif_transpose(Image.open(paths["source_rgb"])).convert("RGB")
    source_inference = source.resize((width, height), Image.Resampling.BILINEAR)
    center = Image.open(paths["center_render"]).convert("RGB")
    if center.size != (width, height):
        raise ValueError(f"center render size {center.size} != PLY image size {(width, height)}")
    source_np, center_np = np.asarray(source_inference), np.asarray(center)
    source_detail = tile_mean(grayscale_gradient(source_np), tile)
    render_detail = tile_mean(grayscale_gradient(center_np), tile)
    detail_retention = render_detail / np.maximum(source_detail, 1e-4)
    rgb_mae = tile_mean(np.abs(source_np.astype(np.float32) - center_np.astype(np.float32)).mean(axis=2) / 255.0, tile)

    valid_tiles = (tile_count > 0) & (source_detail > np.quantile(source_detail, 0.20))
    correlations = {
        "detail_retention_vs_mean_depth": spearman(detail_retention[valid_tiles], tile_depth[valid_tiles]),
        "detail_retention_vs_center_density": spearman(detail_retention[valid_tiles], tile_count[valid_tiles].astype(float)),
        "detail_retention_vs_projected_major_radius": spearman(detail_retention[valid_tiles], tile_radius[valid_tiles]),
        "rgb_mae_vs_mean_depth": spearman(rgb_mae[valid_tiles], tile_depth[valid_tiles]),
    }

    depth_edges = np.quantile(z_f, config["analysis"]["depth_quantile_bins"])
    depth_bins = []
    for index in range(len(depth_edges) - 1):
        lower, upper = float(depth_edges[index]), float(depth_edges[index + 1])
        mask = (z_f >= lower) & (z_f <= upper if index == len(depth_edges) - 2 else z_f < upper)
        bin_tiles = np.unique(flat_tile[mask])
        depth_bins.append({
            "quantile_range": [config["analysis"]["depth_quantile_bins"][index], config["analysis"]["depth_quantile_bins"][index + 1]],
            "depth_range": [lower, upper],
            "gaussian_centers": int(mask.sum()),
            "occupied_tiles": int(bin_tiles.size),
            "centers_per_occupied_tile": float(mask.sum() / max(bin_tiles.size, 1)),
            "projected_major_radius_px": quantiles(major[mask]),
            "projected_minor_radius_px": quantiles(minor[mask]),
            "opacity": quantiles(opacity_f[mask]),
        })

    near_tiles = valid_tiles & (tile_depth <= np.nanquantile(tile_depth[valid_tiles], 0.25))
    far_tiles = valid_tiles & (tile_depth >= np.nanquantile(tile_depth[valid_tiles], 0.75))
    comparison = {}
    for label, mask in (("near_tiles", near_tiles), ("far_tiles", far_tiles)):
        comparison[label] = {
            "count": int(mask.sum()),
            "mean_depth": float(np.nanmedian(tile_depth[mask])),
            "center_density_per_tile": float(np.median(tile_count[mask])),
            "projected_major_radius_px": float(np.nanmedian(tile_radius[mask])),
            "source_detail": float(np.median(source_detail[mask])),
            "render_detail": float(np.median(render_detail[mask])),
            "detail_retention": float(np.median(detail_retention[mask])),
            "rgb_mae": float(np.median(rgb_mae[mask])),
        }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source_inference.save(output / "source_inference_bilinear.png")
    center.save(output / "center_render.png")
    panels = [
        source_inference,
        center,
        heatmap(np.log1p(tile_count)).resize((width, height), Image.Resampling.NEAREST),
        heatmap(tile_depth).resize((width, height), Image.Resampling.NEAREST),
        heatmap(tile_radius).resize((width, height), Image.Resampling.NEAREST),
        heatmap(detail_retention).resize((width, height), Image.Resampling.NEAREST),
    ]
    review = Image.new("RGB", (width * 3, height * 2))
    for index, panel in enumerate(panels):
        review.paste(panel, ((index % 3) * width, (index // 3) * height))
    review.save(output / "review_source_render_density_depth_radius_retention.png")

    far_density_ratio = comparison["far_tiles"]["center_density_per_tile"] / max(comparison["near_tiles"]["center_density_per_tile"], 1e-8)
    far_radius_ratio = comparison["far_tiles"]["projected_major_radius_px"] / max(comparison["near_tiles"]["projected_major_radius_px"], 1e-8)
    far_retention_ratio = comparison["far_tiles"]["detail_retention"] / max(comparison["near_tiles"]["detail_retention"], 1e-8)
    if far_density_ratio < 0.75 and (correlations["detail_retention_vs_center_density"] or 0.0) > 0.25:
        interpretation = "screen_space_center_density_supported"
    elif far_density_ratio >= 0.75 and (far_radius_ratio > 1.20 or far_retention_ratio < 0.80):
        interpretation = "footprint_or_representation_bandwidth_more_supported"
    else:
        interpretation = "mixed_or_inconclusive"

    result = {
        "schema_version": "1.0-infinisplat-center-density-diagnostic",
        "status": "success",
        "experiment_id": config["experiment_id"],
        "producer_machine_id": config["producer_machine_id"],
        "scope": config["interpretation_rules"]["boundary"],
        "inputs": {
            "ply": config["inputs"]["ply"],
            "ply_sha256": config["inputs"]["ply_sha256"],
            "source_rgb_sha256": config["inputs"]["source_rgb_sha256"],
            "center_render": config["inputs"]["center_render"],
        },
        "camera": {"image_size_wh": [width, height], "intrinsic": intrinsic.tolist(), "extrinsic_identity_max_error": float(np.max(np.abs(extrinsic - np.eye(4))))},
        "gaussians": {
            "total": int(vertices.shape[0]),
            "centers_in_frame": int(in_frame.sum()),
            "centers_per_pixel": float(in_frame.sum() / (width * height)),
            "depth": quantiles(z_f),
            "projected_major_radius_px": quantiles(major),
            "projected_minor_radius_px": quantiles(minor),
            "opacity": quantiles(opacity_f),
        },
        "depth_quantile_bins": depth_bins,
        "tile_comparison": comparison,
        "ratios_far_over_near": {
            "center_density": far_density_ratio,
            "projected_major_radius": far_radius_ratio,
            "detail_retention": far_retention_ratio,
        },
        "spearman_correlations": correlations,
        "automatic_interpretation": interpretation,
        "review_layout": [
            "top-left source inference RGB", "top-middle center render", "top-right log Gaussian-center density",
            "bottom-left mean Gaussian depth", "bottom-middle projected major radius", "bottom-right rendered/source detail retention"
        ],
        "output_review": str(output / "review_source_render_density_depth_radius_retention.png"),
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "interpretation": interpretation, "ratios_far_over_near": result["ratios_far_over_near"], "correlations": correlations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
