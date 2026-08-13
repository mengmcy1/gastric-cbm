from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from stage01_common import REPO_ROOT, configure_sharp_cuda_toolkit, file_hash, prepare_output_dir
from stage1_clb_synthetic_smoke import Camera, oracle_render, render as render_clb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CLB-SYN-01 layers to Gaussians and validate rendering.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, default=0.42)
    parser.add_argument("--thickness-factor", type=float, default=0.08)
    parser.add_argument("--opacity", type=float, default=0.995)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_camera(name: str, value: dict, width: int, height: int) -> Camera:
    return Camera(
        name=name,
        angle_deg=float(value["angle_deg"]),
        intrinsics=np.asarray(value["intrinsics_3x3"], dtype=np.float64),
        extrinsics=np.asarray(value["extrinsics_world_to_camera_4x4"], dtype=np.float64),
        width=width,
        height=height,
    )


def load_arrays(manifest_path: Path, layer: dict) -> dict[str, np.ndarray]:
    path = Path(layer["arrays_npz"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def layer_to_gaussians(layer: dict, arrays: dict[str, np.ndarray], camera: Camera, args: argparse.Namespace):
    from sharp.utils import color_space as cs_utils
    from sharp.utils.gaussians import Gaussians3D

    valid = arrays["valid_mask_uint8"] != 0
    ys, xs = np.where(valid)
    depth = torch.from_numpy(arrays["depth_z_float32"][ys, xs]).float()
    intrinsics = torch.from_numpy(camera.intrinsics).float()
    x_camera = (torch.from_numpy(xs).float() + 0.5 - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y_camera = (torch.from_numpy(ys).float() + 0.5 - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    points_camera = torch.stack([x_camera, y_camera, depth], dim=1)
    extrinsics = torch.from_numpy(camera.extrinsics).float()
    rotation_world_to_camera = extrinsics[:3, :3]
    points_world = (points_camera - extrinsics[:3, 3]) @ rotation_world_to_camera

    pixel_world_x = depth / float(camera.intrinsics[0, 0])
    pixel_world_y = depth / float(camera.intrinsics[1, 1])
    scale_x = pixel_world_x * args.scale_factor
    scale_y = pixel_world_y * args.scale_factor
    scale_z = torch.minimum(scale_x, scale_y) * args.thickness_factor
    scales_camera = torch.stack([scale_x, scale_y, scale_z], dim=1)

    # Anchor-camera image plane axes transformed to the shared world frame.
    rotation_local_to_world = rotation_world_to_camera.T
    from sharp.utils import linalg

    rotations = rotation_local_to_world[None].expand(len(xs), -1, -1).contiguous()
    quaternions = linalg.quaternions_from_rotation_matrices(rotations)
    colors_srgb = torch.from_numpy(arrays["rgb_uint8"][ys, xs].copy()).float() / 255.0
    colors_linear = cs_utils.sRGB2linearRGB(colors_srgb)
    opacities = torch.full((len(xs),), float(args.opacity), dtype=torch.float32)
    gaussians = Gaussians3D(
        mean_vectors=points_world[None],
        singular_values=scales_camera[None],
        quaternions=quaternions[None],
        colors=colors_linear[None],
        opacities=opacities[None],
    )
    provenance = arrays["provenance_uint8"][ys, xs]
    return gaussians, provenance


def concatenate(items):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(*[torch.cat([getattr(item, field) for item in items], dim=1) for field in Gaussians3D._fields])


def render_gaussians(gaussians, camera: Camera, renderer, device: torch.device) -> dict[str, np.ndarray]:
    item = gaussians.to(device)
    extrinsics = torch.from_numpy(camera.extrinsics).float()[None].to(device)
    intrinsics = torch.eye(4, dtype=torch.float32)
    intrinsics[:3, :3] = torch.from_numpy(camera.intrinsics).float()
    with torch.inference_mode():
        result = renderer(item, extrinsics, intrinsics[None].to(device), camera.width, camera.height)
    torch.cuda.synchronize(device)
    return {
        "color": result.color[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
        "depth": result.depth[0, 0].cpu().numpy(),
        "alpha": result.alpha[0, 0].cpu().numpy(),
    }


def save_render(output_dir: Path, prefix: str, render: dict[str, np.ndarray]) -> None:
    Image.fromarray(np.round(render["color"] * 255.0).astype(np.uint8)).save(output_dir / f"{prefix}.png")
    Image.fromarray(np.round(np.clip(render["alpha"], 0, 1) * 65535).astype(np.uint16)).save(output_dir / f"{prefix}_alpha_u16.png")
    np.save(output_dir / f"{prefix}_depth_float32.npy", render["depth"].astype(np.float32))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian smoke requires CUDA")
    output_dir = prepare_output_dir(args.output_dir)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    configure_sharp_cuda_toolkit()
    sharp_source = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
    if str(sharp_source) not in sys.path:
        sys.path.insert(0, str(sharp_source))
    from sharp.utils import gsplat
    from sharp.utils.gaussians import save_ply

    manifest = load_manifest(args.manifest)
    first_arrays = load_arrays(args.manifest, manifest["layers"][0])
    height, width = first_arrays["valid_mask_uint8"].shape
    cameras = {name: make_camera(name, value, width, height) for name, value in manifest["cameras"].items()}

    layer_arrays = {}
    layer_gaussians = []
    observed_gaussians = []
    ranges = {}
    provenance_counts = {}
    next_index = 0
    for layer in manifest["layers"]:
        arrays = load_arrays(args.manifest, layer)
        layer_arrays[layer["layer_id"]] = arrays
        gaussians, provenance = layer_to_gaussians(layer, arrays, cameras[layer["anchor_camera"]], args)
        count = int(gaussians.mean_vectors.shape[1])
        ranges[layer["layer_id"]] = [next_index, next_index + count]
        next_index += count
        layer_gaussians.append(gaussians)
        if layer["support_type"] == "observed_surface":
            observed_gaussians.append(gaussians)
        unique, counts = np.unique(provenance, return_counts=True)
        provenance_counts[layer["layer_id"]] = {str(int(key)): int(value) for key, value in zip(unique, counts)}

    all_gaussians = concatenate(layer_gaussians)
    observed_only = concatenate(observed_gaussians)
    save_ply(all_gaussians, float(cameras["center"].intrinsics[0, 0]), (height, width), output_dir / "clb_all_gaussians.ply")
    save_ply(observed_only, float(cameras["center"].intrinsics[0, 0]), (height, width), output_dir / "clb_observed_gaussians.ply")
    (output_dir / "gaussian_sidecar.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0-clb-gaussian-sidecar",
                "source_manifest": str(args.manifest),
                "index_ranges_start_inclusive_end_exclusive": ranges,
                "provenance_counts": provenance_counts,
                "parameters": {
                    "scale_factor": args.scale_factor,
                    "thickness_factor": args.thickness_factor,
                    "opacity": args.opacity,
                    "visibility": "physical_z_buffer_only_no_endpoint_fade",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    device = torch.device("cuda")
    renderer = gsplat.GSplatRenderer(color_space="linearRGB", background_color="black", low_pass_filter_eps=0.0)
    metrics = {}
    review_images = []
    for name in ("center", "left", "right"):
        full = render_gaussians(all_gaussians, cameras[name], renderer, device)
        observed = render_gaussians(observed_only, cameras[name], renderer, device)
        save_render(output_dir, f"full_{name}", full)
        save_render(output_dir, f"observed_{name}", observed)
        oracle = oracle_render(cameras[name])
        oracle_rgb = oracle["color"].astype(np.float32) / 255.0
        Image.fromarray(oracle["color"]).save(output_dir / f"oracle_{name}.png")

        direct = render_clb(manifest["layers"], layer_arrays, cameras[name], cameras)
        hidden_indices = [
            index for index, layer in enumerate(manifest["layers"]) if layer["support_type"] != "observed_surface"
        ]
        hidden_target = np.isin(direct["owner"], hidden_indices)
        owner_interior = np.zeros_like(oracle["valid"], dtype=bool)
        for owner_index in range(len(manifest["layers"])):
            owner_interior |= ndi.binary_erosion(direct["owner"] == owner_index, iterations=3)
        reliable = ndi.binary_erosion(oracle["valid"], iterations=2) & owner_interior
        hidden_interior = ndi.binary_erosion(hidden_target, iterations=3)
        center_diff = np.abs(full["color"] - observed["color"])
        full_error = np.mean(np.abs(full["color"] - oracle_rgb), axis=2)
        observed_error = np.mean(np.abs(observed["color"] - oracle_rgb), axis=2)
        depth_valid = reliable & (full["alpha"] >= 0.95) & np.isfinite(full["depth"]) & (oracle["depth"] > 0)
        depth_relative = np.abs(full["depth"][depth_valid] - oracle["depth"][depth_valid]) / oracle["depth"][depth_valid]
        metrics[name] = {
            "full_alpha_ge_095_rate_reliable_surface_interior": float(np.mean(full["alpha"][reliable] >= 0.95)),
            "center_full_vs_observed_max_rgb_abs_surface_interior": float(np.max(center_diff[reliable])) if name == "center" else None,
            "center_full_vs_observed_max_rgb_abs_all_pixels_diagnostic": float(np.max(center_diff)) if name == "center" else None,
            "hidden_target_pixels": int(hidden_target.sum()),
            "hidden_interior_pixels": int(hidden_interior.sum()),
            "hidden_full_rgb_mae": float(np.mean(full_error[hidden_interior])) if np.any(hidden_interior) else None,
            "hidden_observed_only_rgb_mae": float(np.mean(observed_error[hidden_interior])) if np.any(hidden_interior) else None,
            "hidden_mae_relative_reduction": float(
                1.0 - np.mean(full_error[hidden_interior]) / max(np.mean(observed_error[hidden_interior]), 1e-12)
            ) if np.any(hidden_interior) else None,
            "depth_relative_error_p95_reliable": float(np.quantile(depth_relative, 0.95)) if depth_relative.size else None,
        }
        review_images.append((name, full, observed, oracle["color"]))

    gates = {
        "center_hidden_leakage_rgb_max_at_most_1_over_255_outside_3px_boundary_band": metrics["center"]["center_full_vs_observed_max_rgb_abs_surface_interior"] <= 1.0 / 255.0,
        "left_hidden_mae_reduction_at_least_80_percent": metrics["left"]["hidden_mae_relative_reduction"] >= 0.80,
        "right_hidden_mae_reduction_at_least_80_percent": metrics["right"]["hidden_mae_relative_reduction"] >= 0.80,
        "all_views_alpha_ge_095_at_least_99_percent_on_reliable_surface_interiors": all(value["full_alpha_ge_095_rate_reliable_surface_interior"] >= 0.99 for value in metrics.values()),
        "all_views_depth_p95_at_most_2_percent": all(value["depth_relative_error_p95_reliable"] <= 0.02 for value in metrics.values()),
    }

    canvas = Image.new("RGB", (width * 3, height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("observed GS", "full CLB GS", "oracle")):
        draw.text((column * width + 8, 5), label, fill="white", stroke_width=2, stroke_fill="black")
    for row, (name, full, observed, oracle_rgb) in enumerate(review_images):
        images = [
            np.round(observed["color"] * 255.0).astype(np.uint8),
            np.round(full["color"] * 255.0).astype(np.uint8),
            oracle_rgb,
        ]
        for column, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (column * width, row * height))
        draw.text((8, row * height + 24), name, fill="white", stroke_width=2, stroke_fill="black")
    canvas.save(output_dir / "review.png")

    record = {
        "schema_version": "1.0-clb-to-gaussian-synthetic-test",
        "experiment_id": "CLB-GS-SYN-01",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CLB synthetic-oracle to Gaussian conversion only; no learned prediction and no P01 claim.",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": file_hash(args.manifest),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "gaussian_count": int(all_gaussians.mean_vectors.shape[1]),
        "observed_gaussian_count": int(observed_only.mean_vectors.shape[1]),
        "parameters": {"scale_factor": args.scale_factor, "thickness_factor": args.thickness_factor, "opacity": args.opacity},
        "metrics": metrics,
        "gates": gates,
        "limitations": [
            "Synthetic oracle validates representation conversion, not hidden-layer prediction.",
            "Per-layer provenance is stored in the JSON sidecar because SHARP PLY has no provenance field.",
            "No endpoint angle fade is used; visibility comes from geometry, frustum, and z-buffer.",
        ],
    }
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "result.json").write_text(serialized, encoding="utf-8")
    args.record.write_text(serialized, encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if record["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
