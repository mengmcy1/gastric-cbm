#!/usr/bin/env python3
"""Evaluate Flash3D's second layer on HLP-GEO-01 dense geometry truth."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw

from adaptive3dgs import (
    Camera,
    CanonicalLayerBundle,
    GeometryTarget,
    ProviderContext,
    SupportType,
    evaluate_occlusion_geometry,
    save_bundle,
    validate_result,
)
from adaptive3dgs.adapters.flash3d import (
    Flash3DReadOnlyAdapter,
    Flash3DSecondLayerOutput,
    flash3d_render_to_geometry_prediction,
)
from flash3d_xformers_compat import force_dino_reference_attention, install_unidepth_compatibility


SH_C0 = 0.28209479177387814
HYPERSIM_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--visibility-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--oracle-source-scale-align", action="store_true")
    parser.add_argument("--frozen-record", type=Path)
    parser.add_argument("--coverage-regression-tolerance", type=float, default=5e-5)
    parser.add_argument("--abs-rel-regression-tolerance", type=float, default=5e-5)
    return parser.parse_args()


class FrozenFlash3DArrayBackend:
    """One-shot NumPy boundary between the official model and the core adapter."""

    backend_id = "flash3d-re10k-v2-frozen-repository-backend"

    def __init__(self, output: Flash3DSecondLayerOutput) -> None:
        self.output = output

    def predict_second_layer(self, context: ProviderContext) -> Flash3DSecondLayerOutput:
        if context.depth_z_float32 is not None:
            raise ValueError("Flash3D RGB-only reference backend refuses source depth truth")
        return self.output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def load_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["dataset"][:]


def frame_path(root: Path, scene: str, camera: str, frame: int, suffix: str) -> Path:
    group = "final_hdf5" if suffix == "color" else "geometry_hdf5"
    return root / scene / "images" / f"scene_{camera}_{group}" / f"frame.{frame:04d}.{suffix}.hdf5"


def camera_parameters(path: Path, scenes: set[str]) -> dict[str, dict[str, object]]:
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
                "M_cam_from_uv": matrix,
            }
    if set(result) != scenes:
        raise KeyError(f"camera metadata missing scenes: {sorted(scenes.difference(result))}")
    return result


def scene_scale(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = {row[0]: row[1] for row in csv.reader(stream) if len(row) >= 2}
    return float(values["meters_per_asset_unit"])


def intrinsics_from_hypersim(matrix: np.ndarray, native_w: int, native_h: int, width: int, height: int) -> np.ndarray:
    # HLP-GEO-01 uses the standard diagonal Hypersim projection. Keep the
    # pixel-center convention used by the dense truth generator.
    if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-10):
        raise ValueError("non-diagonal M_cam_from_uv is not supported by this frozen evaluator")
    ray = HYPERSIM_TO_OPENCV @ matrix
    if not (ray[0, 0] > 0 and ray[1, 1] < 0 and ray[2, 2] > 0):
        raise ValueError(f"unexpected Hypersim camera convention: {matrix}")
    native_fx = native_w / (2.0 * ray[0, 0])
    native_fy = native_h / (2.0 * -ray[1, 1])
    k = np.eye(3, dtype=np.float64)
    k[0, 0] = native_fx * width / native_w
    k[1, 1] = native_fy * height / native_h
    k[0, 2] = ((native_w - 1.0) / 2.0) * width / native_w
    k[1, 2] = ((native_h - 1.0) / 2.0) * height / native_h
    return k


def c2w_opencv(rotation_world_from_hypersim: np.ndarray, position_asset: np.ndarray, meters_per_unit: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_world_from_hypersim @ HYPERSIM_TO_OPENCV
    matrix[:3, 3] = position_asset * meters_per_unit
    return matrix


def tonemap_official_proxy(color: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, float]:
    # Exact official Hypersim CGIntrinsics formula. The official script uses
    # render_entity_id != -1; finite positive metric depth is the available
    # geometry-equivalent valid mask in this targeted subset.
    color = color.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(color).all(axis=2)
    brightness = 0.3 * color[:, :, 0] + 0.59 * color[:, :, 1] + 0.11 * color[:, :, 2]
    current = float(np.percentile(brightness[valid], 90)) if np.any(valid) else 0.0
    scale = 0.0 if 0 < current < 0.0001 else (1.0 if current == 0 else 0.8 ** 2.2 / current)
    mapped = np.power(np.maximum(scale * color, 0), 1.0 / 2.2)
    return np.clip(mapped, 0, 1), scale


def resize_rgb(value: np.ndarray, height: int, width: int) -> torch.Tensor:
    tensor = torch.from_numpy(value).permute(2, 0, 1).float()[None]
    return F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)


def resize_array(value: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(value.astype(np.float32))[None, None]
    kwargs = {"size": (height, width), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0, 0].numpy()


def save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(path)


def save_depth(path: Path, depth: np.ndarray, valid: np.ndarray) -> None:
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.quantile(depth[valid], [0.02, 0.98])
        normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        image[valid, 0] = np.round(255 * normalized[valid]).astype(np.uint8)
        image[valid, 1] = np.round(255 * (1 - normalized[valid])).astype(np.uint8)
        image[valid, 2] = 160
    Image.fromarray(image).save(path)


def geometry_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    count = int(mask.sum())
    if count == 0:
        return {"pixels": 0, "abs_rel": None, "mae_m": None, "rmse_m": None, "delta_1_05": None, "delta_1_10": None}
    p, t = pred[mask], truth[mask]
    ratio = np.maximum(p / t, t / p)
    error = p - t
    return {
        "pixels": count,
        "abs_rel": float(np.mean(np.abs(error) / t)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "delta_1_05": float(np.mean(ratio < 1.05)),
        "delta_1_10": float(np.mean(ratio < 1.10)),
    }


def calibration_bins(alpha: np.ndarray, correct: np.ndarray, mask: np.ndarray) -> list[dict[str, float | int | None]]:
    result = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = mask & (alpha >= low) & (alpha <= high if high == 1 else alpha < high)
        result.append(
            {
                "alpha_low": float(low),
                "alpha_high": float(high),
                "pixels": int(selected.sum()),
                "mean_rendered_alpha": float(alpha[selected].mean()) if np.any(selected) else None,
                "fraction_depth_within_10_percent": float(correct[selected].mean()) if np.any(selected) else None,
            }
        )
    return result


def review_image(path: Path, target: np.ndarray, layer2: np.ndarray, alpha: np.ndarray, truth_mask: np.ndarray) -> None:
    height, width = target.shape[:2]
    panels = [
        target,
        layer2,
        np.repeat(alpha[:, :, None], 3, axis=2),
        np.stack([truth_mask.astype(float), np.zeros_like(alpha), np.zeros_like(alpha)], axis=2),
    ]
    labels = ["target", "layer 2 only", "layer 2 alpha", "occlusion truth"]
    canvas = Image.new("RGB", (width * 4, height), "black")
    for index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(np.round(np.clip(panel, 0, 1) * 255).astype(np.uint8)), (index * width, 0))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        draw.text((index * width + 7, 6), label, fill="white", stroke_width=2, stroke_fill="black")
    canvas.save(path)


def main() -> int:
    args = parse_args()
    for name in ("repo", "checkpoint", "config", "dataset_root", "camera_parameters", "visibility_root", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.frozen_record is not None:
        args.frozen_record = args.frozen_record.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    if not (0 < args.alpha_threshold < 1):
        raise ValueError("alpha threshold must be in (0,1)")
    if args.coverage_regression_tolerance < 0 or args.abs_rel_regression_tolerance < 0:
        raise ValueError("regression tolerances must be non-negative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parameters = camera_parameters(args.camera_parameters, {item["scene"] for item in config["triplets"]})
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    original_hub_load = torch.hub.load

    @functools.wraps(original_hub_load)
    def local_unidepth_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs):
        if repo_or_dir == "lpiccinelli-eth/UniDepth":
            hub_kwargs.pop("trust_repo", None)
            hub_kwargs.pop("force_reload", None)
            hub_kwargs["source"] = "local"
            return original_hub_load(str(args.repo.parent / "UniDepth"), model, *hub_args, **hub_kwargs)
        return original_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs)

    torch.hub.load = local_unidepth_hub_load
    install_unidepth_compatibility()
    with initialize_config_dir(version_base=None, config_dir=str(args.repo / "configs")):
        cfg = compose(config_name="config", overrides=["+experiment=layered_re10k"])
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 0
    cfg.model.gaussian_rendering = True
    cfg.model.backbone.weights_init = "scratch"
    sys.path.insert(0, str(args.repo))
    from models.model import GaussianPredictor

    model = GaussianPredictor(cfg)
    force_dino_reference_attention()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    current = model.state_dict()
    adapted = {key: current[key].clone() if "backproject_depth" in key else value for key, value in checkpoint["model"].items()}
    loaded = model.load_state_dict(adapted, strict=False)
    external_prefix = "models.unidepth_extended.unidepth."
    bad_missing = [key for key in loaded.missing_keys if not key.startswith(external_prefix)]
    if bad_missing or loaded.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {bad_missing}, {loaded.unexpected_keys}")
    device = torch.device("cuda:0")
    model.to(device)
    model.set_eval()
    torch.cuda.reset_peak_memory_stats(device)
    target_h, target_w = int(cfg.dataset.height), int(cfg.dataset.width)
    pad = int(cfg.dataset.pad_border_aug)
    results: list[dict[str, object]] = []
    source_diagnostics: list[dict[str, object]] = []
    clb_diagnostics: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for triplet in config["triplets"]:
        scene, camera = triplet["scene"], triplet["camera"]
        output_scene = args.output_dir / scene
        output_scene.mkdir()
        detail = args.dataset_root / scene / "_detail" / camera
        frame_indices = load_hdf5(detail / "camera_keyframe_frame_indices.hdf5").astype(int)
        rotations = load_hdf5(detail / "camera_keyframe_orientations.hdf5").astype(np.float64)
        positions = load_hdf5(detail / "camera_keyframe_positions.hdf5").astype(np.float64)
        index_by_frame = {int(frame): index for index, frame in enumerate(frame_indices)}
        meters_per_unit = scene_scale(args.dataset_root / scene / "_detail" / "metadata_scene.csv")
        source_frame = int(triplet["source_frame"])
        source_index = index_by_frame[source_frame]
        source_color_path = frame_path(args.dataset_root, scene, camera, source_frame, "color")
        source_depth_path = frame_path(args.dataset_root, scene, camera, source_frame, "depth_meters")
        source_color_raw = load_hdf5(source_color_path)
        source_depth = load_hdf5(source_depth_path)
        native_h, native_w = source_depth.shape
        mapped_source, source_tonemap_scale = tonemap_official_proxy(source_color_raw, source_depth)
        source = resize_rgb(mapped_source, target_h, target_w)
        save_rgb(output_scene / "source_tonemap.png", source.permute(1, 2, 0).numpy())
        k_target = intrinsics_from_hypersim(parameters[scene]["M_cam_from_uv"], native_w, native_h, target_w, target_h)
        k_source = k_target.copy()
        k_source[0, 2] += pad
        k_source[1, 2] += pad
        source_c2w = c2w_opencv(rotations[source_index], positions[source_index], meters_per_unit)
        padded_source = F.pad(source, (pad, pad, pad, pad), mode="replicate")[None]
        inputs: dict[object, object] = {
            "target_frame_ids": ["left", "right"],
            ("color", 0, 0): source[None].to(device),
            ("color_aug", 0, 0): padded_source.to(device),
            ("K_src", 0): torch.from_numpy(k_source).float()[None].to(device),
            ("K_tgt", 0): torch.from_numpy(k_target).float()[None].to(device),
            ("depth_sparse", 0): torch.zeros((1, 10, 3), dtype=torch.float32, device=device),
            ("scale_colmap", 0): torch.tensor([1.0], dtype=torch.float32, device=device),
            ("T_c2w", 0): torch.from_numpy(source_c2w).float()[None].to(device),
            ("T_w2c", 0): torch.from_numpy(np.linalg.inv(source_c2w)).float()[None].to(device),
        }
        for side in ("left", "right"):
            target_frame = int(triplet[f"{side}_frame"])
            target_index = index_by_frame[target_frame]
            target_c2w = c2w_opencv(rotations[target_index], positions[target_index], meters_per_unit)
            inputs[("T_c2w", side)] = torch.from_numpy(target_c2w).float()[None].to(device)
            inputs[("T_w2c", side)] = torch.from_numpy(np.linalg.inv(target_c2w)).float()[None].to(device)
            inputs[("K_tgt", side)] = torch.from_numpy(k_target).float()[None].to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(inputs)
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - started
        predicted_source_z = outputs[("depth", 0)][0, 0].detach().cpu().numpy()
        if pad:
            predicted_source_z = predicted_source_z[pad:-pad, pad:-pad]
        if predicted_source_z.shape != (target_h, target_w):
            raise ValueError(f"unexpected cropped source depth shape: {predicted_source_z.shape}")
        source_radial = resize_array(source_depth.astype(np.float32), target_h, target_w, "nearest")
        source_yy, source_xx = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing="ij")
        source_ray_norm = np.sqrt(
            ((source_xx - k_target[0, 2]) / k_target[0, 0]) ** 2
            + ((source_yy - k_target[1, 2]) / k_target[1, 1]) ** 2
            + 1.0
        )
        source_truth_z = source_radial / source_ray_norm
        source_valid = (
            np.isfinite(predicted_source_z)
            & (predicted_source_z > 0)
            & np.isfinite(source_truth_z)
            & (source_truth_z > 0)
        )
        source_scale_ratios = predicted_source_z[source_valid] / source_truth_z[source_valid]
        scene_depth_scale = float(np.median(source_scale_ratios))
        source_metrics = geometry_metrics(predicted_source_z, source_truth_z, source_valid)
        save_depth(output_scene / "source_layer1_depth.png", predicted_source_z, source_valid)
        source_diagnostics.append(
            {
                "scene": scene,
                "source_frame": source_frame,
                "metrics": source_metrics,
                "median_predicted_z_over_truth_z": scene_depth_scale,
                "p10_predicted_z_over_truth_z": float(np.quantile(source_scale_ratios, 0.10)),
                "p90_predicted_z_over_truth_z": float(np.quantile(source_scale_ratios, 0.90)),
            }
        )

        layer_count = int(cfg.model.gaussians_per_pixel)
        padded_h, padded_w = target_h + 2 * pad, target_w + 2 * pad
        source_depth_layers = outputs[("depth", 0)].reshape(1, layer_count, 1, padded_h, padded_w)
        source_alpha_layers = outputs["gauss_opacity"].reshape(1, layer_count, 1, padded_h, padded_w)
        source_dc_layers = outputs["gauss_features_dc"].reshape(1, layer_count, 3, padded_h, padded_w)
        layer1_source = source_depth_layers[0, 0, 0, pad : pad + target_h, pad : pad + target_w]
        layer2_source = source_depth_layers[0, 1, 0, pad : pad + target_h, pad : pad + target_w]
        layer2_alpha_source = source_alpha_layers[0, 1, 0, pad : pad + target_h, pad : pad + target_w]
        layer2_dc_source = source_dc_layers[0, 1, :, pad : pad + target_h, pad : pad + target_w]
        array_output = Flash3DSecondLayerOutput(
            layer1_depth_z_float32=layer1_source.detach().cpu().numpy().astype(np.float32),
            layer2_depth_z_float32=layer2_source.detach().cpu().numpy().astype(np.float32),
            layer2_alpha_float32=layer2_alpha_source.detach().cpu().numpy().astype(np.float32),
            layer2_features_dc_hwc_float32=layer2_dc_source.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32),
            intrinsics_3x3_float64=k_target.astype(np.float64),
            world_to_camera_4x4_float64=np.linalg.inv(source_c2w).astype(np.float64),
            metadata={"anchor_camera": "center", "checkpoint_variant": "re10k_v2"},
        )
        source_rgb_uint8 = np.round(source.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        provider_context = ProviderContext(
            sample_id=f"{scene}-{camera}-{source_frame:04d}",
            rgb_uint8=source_rgb_uint8,
            intrinsics_3x3_float64=k_target.astype(np.float64),
            world_to_camera_4x4_float64=np.linalg.inv(source_c2w).astype(np.float64),
            depth_z_float32=None,
            metadata={"dataset": "HLP-GEO-01", "source_depth_truth_exposed": False},
        )
        provider_result = Flash3DReadOnlyAdapter(FrozenFlash3DArrayBackend(array_output)).predict(provider_context)
        provider_validation = validate_result(
            provider_result,
            expected_support=SupportType.OCCLUSION_HIDDEN,
            deployment=True,
        )
        center_camera = Camera(
            camera_id="center",
            intrinsics_3x3_float64=k_target.astype(np.float64),
            world_to_camera_4x4_float64=np.linalg.inv(source_c2w).astype(np.float64),
            angle_deg=0.0,
        )
        bundle = CanonicalLayerBundle(
            bundle_id=f"HLP-GEO-01-{scene}-flash3d-read-only-v1",
            deployment=True,
            coordinate_convention={
                "camera_extrinsics": "world_to_camera_4x4",
                "camera_forward_axis": "+Z",
                "pixel_origin": "top_left",
                "depth_type": "positive_camera_z",
                "depth_unit": "Flash3D_UniDepth_metric_estimate",
            },
            cameras={"center": center_camera},
            patches=provider_result.patches,
            metadata={"provider_id": provider_result.provider_id, "source_depth_truth_exposed": False},
        )
        clb_manifest = save_bundle(bundle, output_scene / "clb")
        clb_diagnostics.append(
            {
                "scene": scene,
                "manifest": portable(clb_manifest),
                "manifest_sha256": sha256(clb_manifest),
                "patch_count": provider_validation.patch_count,
                "valid_hidden_pixels": provider_validation.valid_pixel_count,
                "geometry_confidence_calibrated": False,
                "outside_source_fov_supported": False,
            }
        )
        if args.oracle_source_scale_align:
            for side in ("left", "right"):
                scaled_transform = outputs[("cam_T_cam", 0, side)].clone()
                scaled_transform[:, :3, 3] *= scene_depth_scale
                outputs[("cam_T_cam", 0, side)] = scaled_transform

        layer2 = dict(outputs)
        layer2_opacity = outputs["gauss_opacity"].clone()
        layer2_opacity[0:: int(cfg.model.gaussians_per_pixel)] = 0.0
        layer2["gauss_opacity"] = layer2_opacity
        with torch.inference_mode():
            model.render_images(inputs, layer2)
        layer2_rgb = {side: layer2[("color_gauss", side, 0)].detach().clone() for side in ("left", "right")}
        layer2_depth = {side: layer2[("depth_gauss", side, 0)].detach().clone() for side in ("left", "right")}

        alpha_outputs = dict(layer2)
        alpha_outputs["gauss_features_dc"] = torch.full_like(outputs["gauss_features_dc"], 0.5 / SH_C0)
        if "gauss_features_rest" in outputs:
            alpha_outputs["gauss_features_rest"] = torch.zeros_like(outputs["gauss_features_rest"])
        original_background = list(cfg.model.bg_colour)
        cfg.model.bg_colour = [0.0, 0.0, 0.0]
        with torch.inference_mode():
            model.render_images(inputs, alpha_outputs)
        cfg.model.bg_colour = original_background
        torch.cuda.synchronize(device)

        for side in ("left", "right"):
            target_frame = int(triplet[f"{side}_frame"])
            target_color_path = frame_path(args.dataset_root, scene, camera, target_frame, "color")
            target_depth_path = frame_path(args.dataset_root, scene, camera, target_frame, "depth_meters")
            target_color_raw = load_hdf5(target_color_path)
            target_depth_radial = load_hdf5(target_depth_path).astype(np.float32)
            mapped_target, target_tonemap_scale = tonemap_official_proxy(target_color_raw, target_depth_radial)
            target_rgb = resize_rgb(mapped_target, target_h, target_w).permute(1, 2, 0).numpy()
            rgb = layer2_rgb[side][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            weighted_z = layer2_depth[side][0, 0].cpu().numpy()
            alpha_rgb = alpha_outputs[("color_gauss", side, 0)][0].clamp(0, 1).cpu().numpy()
            alpha = alpha_rgb.mean(axis=0)
            predicted_z_model_units = weighted_z / np.maximum(alpha, 1e-8)
            predicted_z = (
                predicted_z_model_units / scene_depth_scale
                if args.oracle_source_scale_align
                else predicted_z_model_units
            )

            truth_file = args.visibility_root / scene / f"{side}_visibility_truth.npz"
            truth = np.load(truth_file)
            observed_native = truth["observed_from_source_uint8"].astype(bool)
            hidden_native = truth["occlusion_hidden_uint8"].astype(bool)
            outside_native = truth["outside_source_fov_uint8"].astype(bool)
            observed = resize_array(observed_native, target_h, target_w, "nearest") > 0.5
            hidden = resize_array(hidden_native, target_h, target_w, "nearest") > 0.5
            outside = resize_array(outside_native, target_h, target_w, "nearest") > 0.5
            radial = resize_array(target_depth_radial, target_h, target_w, "nearest")
            yy, xx = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing="ij")
            ray_norm = np.sqrt(((xx - k_target[0, 2]) / k_target[0, 0]) ** 2 + ((yy - k_target[1, 2]) / k_target[1, 1]) ** 2 + 1.0)
            truth_z = radial / ray_norm
            prediction_valid = np.isfinite(predicted_z) & (predicted_z > 0) & (alpha >= args.alpha_threshold)
            hidden_supported = hidden & prediction_valid
            hidden_correct_10 = np.zeros_like(hidden)
            hidden_correct_10[hidden_supported] = np.maximum(
                predicted_z[hidden_supported] / truth_z[hidden_supported],
                truth_z[hidden_supported] / predicted_z[hidden_supported],
            ) < 1.10
            adaptive_prediction = flash3d_render_to_geometry_prediction(
                weighted_z.astype(np.float32),
                alpha.astype(np.float32),
                model_depth_units_per_metric_unit=scene_depth_scale if args.oracle_source_scale_align else 1.0,
            )
            adaptive_target = GeometryTarget(
                depth_z_float32=truth_z.astype(np.float32),
                observed_from_source_mask=observed,
                occlusion_hidden_mask=hidden,
                outside_source_fov_mask=outside,
            )
            adaptive_evaluation = evaluate_occlusion_geometry(
                adaptive_target,
                adaptive_prediction,
                support_threshold=args.alpha_threshold,
            )
            save_rgb(output_scene / f"{side}_target_tonemap.png", target_rgb)
            save_rgb(output_scene / f"{side}_layer2.png", rgb)
            Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(output_scene / f"{side}_layer2_alpha.png")
            save_depth(output_scene / f"{side}_layer2_depth.png", predicted_z, prediction_valid)
            review_image(output_scene / f"{side}_review.png", target_rgb, rgb, alpha, hidden)
            metrics = geometry_metrics(predicted_z, truth_z, hidden_supported)
            results.append(
                {
                    "scene": scene,
                    "side": side,
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "yaw_deg": float(triplet[f"{side}_yaw_deg"]),
                    "translation_m": float(triplet[f"{side}_translation_m"]),
                    "source_tonemap_scale": source_tonemap_scale,
                    "target_tonemap_scale": target_tonemap_scale,
                    "inference_and_full_render_seconds": inference_seconds,
                    "occlusion_hidden_pixels_resized": int(hidden.sum()),
                    "outside_source_fov_pixels_resized": int(outside.sum()),
                    "layer2_supported_hidden_pixels": int(hidden_supported.sum()),
                    "layer2_hidden_coverage_at_alpha_threshold": float(hidden_supported.sum() / max(hidden.sum(), 1)),
                    "layer2_hidden_geometry": metrics,
                    "adaptive3dgs_evaluation": adaptive_evaluation,
                    "adaptive3dgs_vs_legacy_same_arrays": {
                        "hidden_coverage_abs_diff": abs(
                            float(adaptive_evaluation["hidden_coverage"])
                            - float(hidden_supported.sum() / max(hidden.sum(), 1))
                        ),
                        "abs_rel_abs_diff": abs(
                            float(adaptive_evaluation["geometry"]["abs_rel"]) - float(metrics["abs_rel"])
                        ),
                    },
                    "rendered_alpha_diagnostic_bins": calibration_bins(alpha, hidden_correct_10, hidden & prediction_valid),
                    "truth": portable(truth_file),
                    "review": portable(output_scene / f"{side}_review.png"),
                }
            )

    coverages = [item["layer2_hidden_coverage_at_alpha_threshold"] for item in results]
    abs_rels = [item["layer2_hidden_geometry"]["abs_rel"] for item in results if item["layer2_hidden_geometry"]["abs_rel"] is not None]
    reference_regression = None
    regression_pass = True
    if args.frozen_record is not None:
        frozen = json.loads(args.frozen_record.read_text(encoding="utf-8"))
        frozen_by_view = {(item["scene"], item["side"]): item for item in frozen["results"]}
        regression_views = []
        for item in results:
            key = (item["scene"], item["side"])
            if key not in frozen_by_view:
                raise KeyError(f"view missing from frozen record: {key}")
            frozen_item = frozen_by_view[key]
            current_coverage = float(item["adaptive3dgs_evaluation"]["hidden_coverage"])
            current_abs_rel = float(item["adaptive3dgs_evaluation"]["geometry"]["abs_rel"])
            frozen_coverage = float(frozen_item["layer2_hidden_coverage_at_alpha_threshold"])
            frozen_abs_rel = float(frozen_item["layer2_hidden_geometry"]["abs_rel"])
            coverage_diff = abs(current_coverage - frozen_coverage)
            abs_rel_diff = abs(current_abs_rel - frozen_abs_rel)
            view_pass = (
                coverage_diff <= args.coverage_regression_tolerance
                and abs_rel_diff <= args.abs_rel_regression_tolerance
                and float(item["adaptive3dgs_vs_legacy_same_arrays"]["hidden_coverage_abs_diff"]) <= 1e-12
                and float(item["adaptive3dgs_vs_legacy_same_arrays"]["abs_rel_abs_diff"]) <= 1e-7
            )
            regression_pass = regression_pass and view_pass
            regression_views.append(
                {
                    "scene": item["scene"],
                    "side": item["side"],
                    "current_hidden_coverage": current_coverage,
                    "frozen_hidden_coverage": frozen_coverage,
                    "hidden_coverage_abs_diff": coverage_diff,
                    "current_abs_rel": current_abs_rel,
                    "frozen_abs_rel": frozen_abs_rel,
                    "abs_rel_abs_diff": abs_rel_diff,
                    "pass": view_pass,
                }
            )
        if len(frozen_by_view) != len(results):
            regression_pass = False
        reference_regression = {
            "frozen_record": portable(args.frozen_record),
            "frozen_record_sha256": sha256(args.frozen_record),
            "coverage_abs_tolerance": args.coverage_regression_tolerance,
            "abs_rel_abs_tolerance": args.abs_rel_regression_tolerance,
            "views": regression_views,
            "all_views_pass": regression_pass,
        }
    record = {
        "schema_version": (
            "stage1.5-flash3d-reference-regression-v1"
            if args.frozen_record is not None
            else "stage1.4-flash3d-hypersim-geometry-v1"
        ),
        "status": (
            "pass_reference_regression"
            if args.frozen_record is not None and regression_pass
            else "failed_reference_regression"
            if args.frozen_record is not None
            else "complete_raw_geometry_uncalibrated"
        ),
        "producer_machine_id": "linux5080",
        "physical_gpu_index": args.physical_gpu_index,
        "cuda_visible_devices_expected": str(args.physical_gpu_index),
        "scope": "Flash3D second-layer-only geometry on HLP-GEO-01 occlusion_hidden truth",
        "oracle_source_scale_align": args.oracle_source_scale_align,
        "oracle_boundary": (
            "uses source depth truth to align each scene's translation and predicted depth scale; diagnostic only, not deployable RGB-only performance"
            if args.oracle_source_scale_align
            else "disabled; raw metric-depth prediction and metric camera translation"
        ),
        "config": portable(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint": portable(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "coordinate_conversion": "Hypersim right/up/back -> OpenCV right/down/forward using diag(1,-1,-1); translations in meters",
        "depth_normalization": "rasterizer opacity-weighted Z divided by separately rendered white-layer accumulated alpha",
        "alpha_threshold": args.alpha_threshold,
        "tonemap": {
            "formula": "official Hypersim CGIntrinsics: 90th percentile brightness -> 0.8 after gamma 1/2.2",
            "valid_mask_deviation": "finite positive depth proxy used instead of render_entity_id!=-1 because targeted subset does not include entity-id files",
        },
        "source_layer1_diagnostics": source_diagnostics,
        "clb_adapter_diagnostics": clb_diagnostics,
        "reference_regression": reference_regression,
        "results": results,
        "summary": {
            "views": len(results),
            "mean_hidden_coverage": float(np.mean(coverages)),
            "min_hidden_coverage": float(np.min(coverages)),
            "mean_abs_rel_on_supported_hidden": float(np.mean(abs_rels)) if abs_rels else None,
            "median_abs_rel_on_supported_hidden": float(np.median(abs_rels)) if abs_rels else None,
            "mean_source_layer1_abs_rel": float(np.mean([item["metrics"]["abs_rel"] for item in source_diagnostics])),
            "source_median_scale_ratios": [item["median_predicted_z_over_truth_z"] for item in source_diagnostics],
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "total_runtime_seconds_after_model_load": time.perf_counter() - started_all,
        },
        "quality_boundary": "raw measurement only; rendered alpha is not confidence, bins are diagnostics, no provider pass/fail or calibrated fusion confidence is claimed",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "summary": record["summary"]}, indent=2))
    return 0 if regression_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
