from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    name: str
    angle_deg: float
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLB-SYN-01 controlled representation test.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def look_at_camera(name: str, angle_deg: float, width: int, height: int, focal: float) -> Camera:
    radius = 6.0
    target = np.array([0.0, 0.0, 6.0], dtype=np.float64)
    theta = math.radians(angle_deg)
    eye = np.array([radius * math.sin(theta), 0.0, 6.0 - radius * math.cos(theta)], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    down_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(down_hint, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation
    intrinsics = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Camera(name, angle_deg, intrinsics, extrinsics, width, height)


def rays_world(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0 : camera.height, 0 : camera.width]
    pixel = np.stack([xx + 0.5, yy + 0.5, np.ones_like(xx)], axis=-1).astype(np.float64)
    rays_camera = pixel @ np.linalg.inv(camera.intrinsics).T
    rotation = camera.extrinsics[:3, :3]
    rays = rays_camera @ rotation
    origin = -rotation.T @ camera.extrinsics[:3, 3]
    return np.broadcast_to(origin, rays.shape), rays


def intersect_plane(camera: Camera, world_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins, rays = rays_world(camera)
    distance = (world_z - origins[..., 2]) / rays[..., 2]
    points = origins + distance[..., None] * rays
    points_camera = points @ camera.extrinsics[:3, :3].T + camera.extrinsics[:3, 3]
    return points, points_camera[..., 2], distance > 0.0


def foreground_footprint(points: np.ndarray) -> np.ndarray:
    return (np.abs(points[..., 0]) <= 0.85) & (np.abs(points[..., 1]) <= 1.15)


def background_color(points: np.ndarray) -> np.ndarray:
    x = points[..., 0]
    y = points[..., 1]
    checker = ((np.floor((x + 8.0) * 1.4) + np.floor((y + 4.0) * 1.4)) % 2) != 0
    color = np.empty((*x.shape, 3), dtype=np.uint8)
    color[..., 0] = np.where(checker, 65, 32)
    color[..., 1] = np.where(checker, 170, 105)
    color[..., 2] = np.where(checker, 230, 175)
    return color


def foreground_color(shape: tuple[int, int]) -> np.ndarray:
    color = np.zeros((*shape, 3), dtype=np.uint8)
    color[..., 0] = 235
    color[..., 1] = 92
    color[..., 2] = 55
    return color


def project(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = points @ camera.extrinsics[:3, :3].T + camera.extrinsics[:3, 3]
    z = camera_points[..., 2]
    uvw = camera_points @ camera.intrinsics.T
    u = uvw[..., 0] / z
    v = uvw[..., 1] / z
    return u, v, z


def center_fov_mask(points: np.ndarray, center: Camera) -> np.ndarray:
    u, v, z = project(points, center)
    return (z > 0.0) & (u >= 0.0) & (u < center.width) & (v >= 0.0) & (v < center.height)


def shared_plane_roundtrip_error(center: Camera, endpoint: Camera, world_z: float) -> np.ndarray:
    points, _, forward = intersect_plane(center, world_z)
    source_u, source_v, _ = project(points, center)
    endpoint_u, endpoint_v, endpoint_z = project(points, endpoint)
    inside = (
        forward
        & (endpoint_z > 0.0)
        & (endpoint_u >= 0.0)
        & (endpoint_u < endpoint.width)
        & (endpoint_v >= 0.0)
        & (endpoint_v < endpoint.height)
    )
    endpoint_pixels = np.stack(
        [endpoint_u[inside], endpoint_v[inside], np.ones(np.count_nonzero(inside))], axis=-1
    )
    rays_camera = endpoint_pixels @ np.linalg.inv(endpoint.intrinsics).T
    rays_world = rays_camera @ endpoint.extrinsics[:3, :3]
    origin = -endpoint.extrinsics[:3, :3].T @ endpoint.extrinsics[:3, 3]
    distance = (world_z - origin[2]) / rays_world[:, 2]
    recovered = origin[None] + distance[:, None] * rays_world
    recovered_u, recovered_v, _ = project(recovered, center)
    return np.hypot(recovered_u - source_u[inside], recovered_v - source_v[inside])


def save_layer(
    output_dir: Path,
    layer_id: str,
    support_type: str,
    anchor: Camera,
    world_z: float,
    valid: np.ndarray,
    rgb: np.ndarray,
    provenance_value: int,
    confidence: float,
) -> dict:
    _, depth, _ = intersect_plane(anchor, world_z)
    depth_out = np.where(valid, depth, np.nan).astype(np.float32)
    alpha = valid.astype(np.float32)
    provenance = np.where(valid, provenance_value, 0).astype(np.uint8)
    confidence_map = np.where(valid, confidence, 0.0).astype(np.float32)
    arrays_name = f"{layer_id}.npz"
    np.savez_compressed(
        output_dir / arrays_name,
        rgb_uint8=rgb.astype(np.uint8),
        depth_z_float32=depth_out,
        alpha_float32=alpha,
        valid_mask_uint8=valid.astype(np.uint8),
        provenance_uint8=provenance,
        geometry_confidence_float32=confidence_map,
        appearance_confidence_float32=confidence_map,
    )
    return {
        "layer_id": layer_id,
        "surface_id": layer_id,
        "support_type": support_type,
        "anchor_camera": anchor.name,
        "arrays_npz": arrays_name,
        "synthetic_plane_world_z": world_z,
    }


def sample_layer(layer: dict, arrays: dict[str, np.ndarray], camera: Camera, cameras: dict[str, Camera]) -> dict:
    points, target_depth, forward = intersect_plane(camera, float(layer["synthetic_plane_world_z"]))
    anchor = cameras[layer["anchor_camera"]]
    u, v, source_depth = project(points, anchor)
    ix = np.floor(u).astype(np.int64)
    iy = np.floor(v).astype(np.int64)
    inside = forward & (source_depth > 0.0) & (ix >= 0) & (ix < anchor.width) & (iy >= 0) & (iy < anchor.height)
    valid = np.zeros((camera.height, camera.width), dtype=bool)
    color = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    if np.any(inside):
        ty, tx = np.where(inside)
        sy, sx = iy[inside], ix[inside]
        source_valid = arrays["valid_mask_uint8"][sy, sx] != 0
        ty, tx, sy, sx = ty[source_valid], tx[source_valid], sy[source_valid], sx[source_valid]
        valid[ty, tx] = True
        color[ty, tx] = arrays["rgb_uint8"][sy, sx]
    return {"valid": valid, "depth": target_depth, "color": color}


def render(layers: list[dict], layer_arrays: dict[str, dict[str, np.ndarray]], camera: Camera, cameras: dict[str, Camera]) -> dict:
    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    color = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    owner = np.full((camera.height, camera.width), -1, dtype=np.int16)
    for index, layer in enumerate(layers):
        sample = sample_layer(layer, layer_arrays[layer["layer_id"]], camera, cameras)
        update = sample["valid"] & (sample["depth"] < depth)
        depth[update] = sample["depth"][update]
        color[update] = sample["color"][update]
        owner[update] = index
    return {"depth": depth, "color": color, "owner": owner}


def oracle_render(camera: Camera) -> dict:
    bg_points, bg_depth, bg_forward = intersect_plane(camera, 8.0)
    fg_points, fg_depth, fg_forward = intersect_plane(camera, 4.0)
    fg = foreground_footprint(fg_points) & fg_forward
    depth = np.where(fg, fg_depth, bg_depth)
    color = background_color(bg_points)
    color[fg] = foreground_color(fg.shape)[fg]
    valid = bg_forward | fg
    return {"depth": depth, "color": color, "valid": valid, "foreground": fg}


def camera_json(camera: Camera) -> dict:
    return {
        "angle_deg": camera.angle_deg,
        "intrinsics_3x3": camera.intrinsics.tolist(),
        "extrinsics_world_to_camera_4x4": camera.extrinsics.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    width, height, focal = 320, 240, 300.0
    cameras = {
        "center": look_at_camera("center", 0.0, width, height, focal),
        "left": look_at_camera("left", -15.0, width, height, focal),
        "right": look_at_camera("right", 15.0, width, height, focal),
    }
    center_bg_points, _, center_bg_forward = intersect_plane(cameras["center"], 8.0)
    center_fg_points, _, center_fg_forward = intersect_plane(cameras["center"], 4.0)
    center_fg = foreground_footprint(center_fg_points) & center_fg_forward
    observed_bg = center_bg_forward & ~foreground_footprint(center_bg_points)
    hidden_bg = center_bg_forward & foreground_footprint(center_bg_points)

    layers = []
    layers.append(save_layer(args.output_dir, "observed_background", "observed_surface", cameras["center"], 8.0, observed_bg, background_color(center_bg_points), 1, 1.0))
    layers.append(save_layer(args.output_dir, "observed_foreground", "observed_surface", cameras["center"], 4.0, center_fg, foreground_color(center_fg.shape), 1, 1.0))
    layers.append(save_layer(args.output_dir, "hidden_behind_foreground", "occlusion_hidden", cameras["center"], 8.0, hidden_bg, background_color(center_bg_points), 6, 1.0))

    for side in ("left", "right"):
        points, _, forward = intersect_plane(cameras[side], 8.0)
        outside = forward & ~center_fov_mask(points, cameras["center"])
        layers.append(save_layer(args.output_dir, f"{side}_outside_fov", "outside_source_fov", cameras[side], 8.0, outside, background_color(points), 6, 1.0))

    manifest = {
        "schema_version": "1.0-canonical-layer-bundle",
        "experiment_id": "CLB-SYN-01",
        "deployment": False,
        "coordinate_convention": {
            "world_frame": "shared_right_handed_world_frame",
            "camera_extrinsics": "world_to_camera_4x4",
            "camera_forward_axis": "+Z",
            "pixel_origin": "top_left",
            "depth_type": "positive_camera_z",
            "depth_unit": "meter",
        },
        "cameras": {name: camera_json(camera) for name, camera in cameras.items()},
        "layers": layers,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    layer_arrays = {}
    for layer in layers:
        with np.load(args.output_dir / layer["arrays_npz"], allow_pickle=False) as data:
            layer_arrays[layer["layer_id"]] = {key: data[key] for key in data.files}

    render_results = {}
    depth_errors = []
    wrong_order_pixels = 0
    compared_pixels = 0
    for name, camera in cameras.items():
        result = render(layers, layer_arrays, camera, cameras)
        oracle = oracle_render(camera)
        comparable = np.isfinite(result["depth"]) & oracle["valid"]
        relative_error = np.abs(result["depth"][comparable] - oracle["depth"][comparable]) / oracle["depth"][comparable]
        depth_errors.extend(relative_error.tolist())
        wrong_order_pixels += int(np.count_nonzero((result["depth"] > oracle["depth"] + 1e-6) & comparable))
        compared_pixels += int(comparable.sum())
        Image.fromarray(result["color"]).save(args.output_dir / f"render_{name}.png")
        Image.fromarray(oracle["color"]).save(args.output_dir / f"oracle_{name}.png")
        owner_counts = {layers[index]["layer_id"]: int(np.count_nonzero(result["owner"] == index)) for index in range(len(layers))}
        render_results[name] = {
            "covered_pixels": int(np.count_nonzero(result["owner"] >= 0)),
            "owner_visible_pixels": owner_counts,
        }

    hidden_ids = {"hidden_behind_foreground", "left_outside_fov", "right_outside_fov"}
    center_hidden_visible = sum(render_results["center"]["owner_visible_pixels"][item] for item in hidden_ids)
    endpoint_presence = {}
    for side in ("left", "right"):
        endpoint_presence[side] = {
            "occlusion_hidden_pixels": render_results[side]["owner_visible_pixels"]["hidden_behind_foreground"],
            "matching_outside_fov_pixels": render_results[side]["owner_visible_pixels"][f"{side}_outside_fov"],
        }
    errors = np.asarray(depth_errors, dtype=np.float64)
    roundtrip_errors = np.concatenate(
        [
            shared_plane_roundtrip_error(cameras["center"], cameras["left"], 8.0),
            shared_plane_roundtrip_error(cameras["center"], cameras["right"], 8.0),
        ]
    )
    metrics = {
        "center_hidden_visible_pixels": center_hidden_visible,
        "endpoint_presence": endpoint_presence,
        "endpoint_depth_relative_error_p95": float(np.quantile(errors, 0.95)) if errors.size else None,
        "wrong_occlusion_order_rate": wrong_order_pixels / max(compared_pixels, 1),
        "shared_surface_cross_view_median_reprojection_error_px": float(np.median(roundtrip_errors)),
    }
    gates = {
        "center_no_hidden_leakage": center_hidden_visible == 0,
        "both_hidden_types_visible_at_left": all(value > 0 for value in endpoint_presence["left"].values()),
        "both_hidden_types_visible_at_right": all(value > 0 for value in endpoint_presence["right"].values()),
        "depth_p95_at_most_1_percent": metrics["endpoint_depth_relative_error_p95"] <= 0.01,
        "wrong_order_rate_at_most_0_1_percent": metrics["wrong_occlusion_order_rate"] <= 0.001,
        "cross_view_reprojection_at_most_0_25_px": metrics["shared_surface_cross_view_median_reprojection_error_px"] <= 0.25,
    }
    record = {
        "schema_version": "1.0-clb-synthetic-representation-test",
        "experiment_id": "CLB-SYN-01",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "Representation and renderer capacity only; no learned prediction and no P01 claim.",
        "manifest": str(manifest_path),
        "metrics": metrics,
        "gates": gates,
        "renders": render_results,
    }
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if record["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
