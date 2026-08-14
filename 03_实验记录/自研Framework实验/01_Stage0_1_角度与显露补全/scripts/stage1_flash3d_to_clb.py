#!/usr/bin/env python3
"""Adapt Flash3D's second Gaussian layer into uncalibrated CLB patches."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from hydra import compose, initialize_config_dir
from PIL import Image

from flash3d_xformers_compat import (
    XFORMERS_SOURCE_COMMIT,
    force_dino_reference_attention,
    install_unidepth_compatibility,
)


SH_C0 = 0.28209479177387814


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def metadata_rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line.split() for line in lines[1:] if line.strip()]
    if not rows or any(len(row) != 19 for row in rows):
        raise ValueError(f"unexpected RealEstate10K metadata format: {path}")
    return rows


def normalized_k(row: list[str], width: int, height: int) -> np.ndarray:
    fx, fy, cx, cy = [float(value) for value in row[1:5]]
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = fx * width
    matrix[1, 1] = fy * height
    matrix[0, 2] = cx * width
    matrix[1, 2] = cy * height
    return matrix


def w2c(row: list[str]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3] = np.asarray([float(value) for value in row[7:]], dtype=np.float64).reshape(3, 4)
    return matrix


def relative_scaled_w2c(source: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
    relative = target @ np.linalg.inv(source)
    relative[:3, 3] *= scale
    return relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-record", type=Path, required=True)
    parser.add_argument("--scale-record", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("repo", "checkpoint", "dataset_record", "scale_record", "protocol", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    dataset = json.loads(args.dataset_record.read_text(encoding="utf-8"))
    scale_record = json.loads(args.scale_record.read_text(encoding="utf-8"))
    if dataset.get("status") != "pass" or scale_record.get("status") != "pass":
        raise ValueError("dataset and scale records must both have status=pass")
    scales = {
        item["sequence"]: item["translation_scale_predicted_depth_per_colmap_unit"]
        for item in scale_record["sequences"]
    }
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    original_hub_load = torch.hub.load

    @functools.wraps(original_hub_load)
    def local_unidepth_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs):
        if repo_or_dir == "lpiccinelli-eth/UniDepth":
            local_repo = args.repo.parent / "UniDepth"
            hub_kwargs.pop("trust_repo", None)
            hub_kwargs.pop("force_reload", None)
            hub_kwargs["source"] = "local"
            return original_hub_load(str(local_repo), model, *hub_args, **hub_kwargs)
        return original_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs)

    torch.hub.load = local_unidepth_hub_load
    install_unidepth_compatibility()
    with initialize_config_dir(version_base=None, config_dir=str(args.repo / "configs")):
        cfg = compose(config_name="config", overrides=["+experiment=layered_re10k"])
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 0
    cfg.model.gaussian_rendering = False
    cfg.model.backbone.weights_init = "scratch"
    sys.path.insert(0, str(args.repo))
    from models.model import GaussianPredictor

    model = GaussianPredictor(cfg)
    force_dino_reference_attention()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    current_state = model.state_dict()
    adapted_state = {
        key: current_state[key].clone() if "backproject_depth" in key else value
        for key, value in checkpoint["model"].items()
    }
    loaded = model.load_state_dict(adapted_state, strict=False)
    external_prefix = "models.unidepth_extended.unidepth."
    unexpected_missing = [key for key in loaded.missing_keys if not key.startswith(external_prefix)]
    if unexpected_missing or loaded.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={unexpected_missing}, unexpected={loaded.unexpected_keys}"
        )
    device = torch.device("cuda:0")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    target_h, target_w = int(cfg.dataset.height), int(cfg.dataset.width)
    pad = int(cfg.dataset.pad_border_aug)
    results: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for sequence in dataset["sequences"]:
        sequence_id = sequence["sequence"]
        if sequence_id not in scales:
            raise KeyError(f"missing recovered scale for {sequence_id}")
        sequence_dir = args.output_dir / sequence_id
        sequence_dir.mkdir()
        rows = metadata_rows(Path.cwd() / sequence["metadata"]["path"])
        source_info = sequence["frames"]["source"]
        frame_indices = {
            "center": int(source_info["frame_index"]),
            "left": int(sequence["frames"]["left"]["frame_index"]),
            "right": int(sequence["frames"]["right"]["frame_index"]),
        }
        source_path = Path.cwd() / source_info["path"]
        image = Image.open(source_path).convert("RGB")
        resized = TF.resize(image, [target_h, target_w], interpolation=TF.InterpolationMode.LANCZOS)
        color = TF.to_tensor(resized)
        color_padded = F.pad(color, (pad, pad, pad, pad), mode="replicate")[None]
        k_padded = normalized_k(rows[frame_indices["center"]], target_w + 2 * pad, target_h + 2 * pad)
        # Match the official loader: focal lengths use the unpadded size while
        # principal points use the padded size.
        k_padded[0, 0] = float(rows[frame_indices["center"]][1]) * target_w
        k_padded[1, 1] = float(rows[frame_indices["center"]][2]) * target_h
        inputs = {
            ("color_aug", 0, 0): color_padded.to(device),
            ("K_src", 0): torch.from_numpy(k_padded).float()[None].to(device),
        }
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.models["unidepth_extended"](inputs)
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
        layers = int(cfg.model.gaussians_per_pixel)
        depth = outputs[("depth", 0)].reshape(1, layers, 1, target_h + 2 * pad, target_w + 2 * pad)
        opacity = outputs["gauss_opacity"].reshape(1, layers, 1, target_h + 2 * pad, target_w + 2 * pad)
        dc = outputs["gauss_features_dc"].reshape(1, layers, 3, target_h + 2 * pad, target_w + 2 * pad)
        first_depth = depth[0, 0, 0, pad : pad + target_h, pad : pad + target_w]
        hidden_depth = depth[0, 1, 0, pad : pad + target_h, pad : pad + target_w]
        hidden_alpha = opacity[0, 1, 0, pad : pad + target_h, pad : pad + target_w]
        hidden_dc = dc[0, 1, :, pad : pad + target_h, pad : pad + target_w]
        hidden_rgb_float = (0.5 + SH_C0 * hidden_dc).clamp(0.0, 1.0)
        finite = (
            torch.isfinite(hidden_depth)
            & torch.isfinite(hidden_alpha)
            & torch.isfinite(hidden_rgb_float).all(dim=0)
            & (hidden_depth > first_depth)
            & (first_depth > 0)
        )
        valid = finite.detach().cpu().numpy()
        rgb_float_np = hidden_rgb_float.permute(1, 2, 0).detach().cpu().numpy()
        rgb_uint8 = np.round(rgb_float_np * 255.0).astype(np.uint8)
        depth_np = hidden_depth.detach().cpu().numpy().astype(np.float32)
        alpha_np = hidden_alpha.detach().cpu().numpy().astype(np.float32)
        depth_np[~valid] = np.nan
        alpha_np[~valid] = 0.0
        provenance = np.zeros((target_h, target_w), dtype=np.uint8)
        provenance[valid] = 3  # predicted_hidden_geometry; appearance is declared in manifest.
        zero_confidence = np.zeros((target_h, target_w), dtype=np.float32)
        arrays_path = sequence_dir / "occlusion_hidden_raw.npz"
        np.savez_compressed(
            arrays_path,
            rgb_uint8=rgb_uint8,
            depth_z_float32=depth_np,
            alpha_float32=alpha_np,
            valid_mask_uint8=valid.astype(np.uint8),
            provenance_uint8=provenance,
            geometry_confidence_float32=zero_confidence,
            appearance_confidence_float32=zero_confidence,
        )

        source_pose = w2c(rows[frame_indices["center"]])
        scale = float(scales[sequence_id])
        cameras = {}
        for camera_name in ("center", "left", "right"):
            row = rows[frame_indices[camera_name]]
            camera_w2c = (
                np.eye(4, dtype=np.float64)
                if camera_name == "center"
                else relative_scaled_w2c(source_pose, w2c(row), scale)
            )
            frame_record = sequence["frames"]["source" if camera_name == "center" else camera_name]
            cameras[camera_name] = {
                "angle_deg": float(frame_record["angle_deg"]),
                "intrinsics_3x3": normalized_k(row, target_w, target_h).tolist(),
                "extrinsics_world_to_camera_4x4": camera_w2c.tolist(),
            }
        manifest = {
            "schema_version": "1.0-canonical-layer-bundle",
            "bundle_id": f"HLP-APP-01-{sequence_id}-flash3d-raw-v1",
            "deployment": True,
            "coordinate_convention": {
                "world_frame": "shared_right_handed_world_frame",
                "camera_extrinsics": "world_to_camera_4x4",
                "camera_forward_axis": "+Z",
                "pixel_origin": "top_left",
                "depth_type": "positive_camera_z",
                "depth_unit": "meter",
            },
            "depth_type": "positive_camera_z",
            "depth_unit": "UniDepth_metric_estimate",
            "cameras": cameras,
            "layers": [
                {
                    "layer_id": "flash3d_second_layer_raw",
                    "support_type": "occlusion_hidden",
                    "anchor_camera": "center",
                    "arrays_npz": arrays_path.name,
                    "geometry_provenance": "predicted_hidden_geometry",
                    "appearance_provenance": "predicted_hidden_appearance",
                    "appearance_encoding": "sRGB_uint8_from_clamp(0.5+SH_C0*features_dc)",
                    "alpha_semantics": "Flash3D Gaussian opacity; not confidence",
                    "confidence_status": "uncalibrated_zero_not_accepted_for_fusion",
                    "outside_source_fov_supported": False,
                }
            ],
        }
        manifest_path = sequence_dir / "clb_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        validation = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("stage1_validate_clb.py")), str(manifest_path), "--protocol", str(args.protocol)],
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "sequence": sequence_id,
                "translation_scale": scale,
                "runtime_seconds": runtime,
                "manifest": portable(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "arrays": portable(arrays_path),
                "arrays_sha256": sha256(arrays_path),
                "valid_pixels": int(valid.sum()),
                "total_pixels": int(valid.size),
                "valid_fraction": float(valid.mean()),
                "hidden_depth_increment_min": float((hidden_depth - first_depth)[finite].min().item()),
                "alpha_min_valid": float(hidden_alpha[finite].min().item()),
                "alpha_max_valid": float(hidden_alpha[finite].max().item()),
                "sh_rgb_preclamp_out_of_range_fraction": float(
                    (((0.5 + SH_C0 * hidden_dc) < 0) | ((0.5 + SH_C0 * hidden_dc) > 1)).float().mean().item()
                ),
                "geometry_confidence": "all_zero_uncalibrated",
                "appearance_confidence": "all_zero_uncalibrated",
                "validator_status": "pass" if validation.returncode == 0 else "fail",
                "validator_stdout": validation.stdout,
                "validator_stderr": validation.stderr,
            }
        )

    status = "schema_pass_uncalibrated" if all(item["validator_status"] == "pass" for item in results) else "fail"
    record = {
        "schema_version": "stage1.4-flash3d-to-clb-v1",
        "status": status,
        "scope": "raw Flash3D second-layer to CLB schema conversion; no quality or fusion acceptance",
        "producer_machine_id": "linux5080",
        "device": {
            "logical": "cuda:0",
            "physical_selected_by_cuda_visible_devices": args.physical_gpu_index,
            "name": torch.cuda.get_device_name(device),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "inputs": {
            "dataset_record": portable(args.dataset_record),
            "dataset_record_sha256": sha256(args.dataset_record),
            "scale_record": portable(args.scale_record),
            "scale_record_sha256": sha256(args.scale_record),
            "checkpoint": portable(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "protocol": portable(args.protocol),
            "protocol_sha256": sha256(args.protocol),
        },
        "compatibility": {
            "nystrom_source_commit": XFORMERS_SOURCE_COMMIT,
            "outside_source_fov_supported": False,
            "opacity_is_confidence": False,
            "confidence_policy": "zero until calibrated against held-out multiview truth",
        },
        "summary": {
            "sequence_count": len(results),
            "all_clb_schema_valid": all(item["validator_status"] == "pass" for item in results),
            "all_confidences_uncalibrated_zero": True,
            "total_runtime_seconds": time.perf_counter() - total_started,
        },
        "sequences": results,
        "quality_boundary": (
            "schema_pass_uncalibrated is not provider quality pass; zero confidence prevents these raw patches "
            "from entering accepted fusion, and OutsideFOV remains unsupported"
        ),
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "device": record["device"], "summary": record["summary"]}, indent=2))
    return 0 if status == "schema_pass_uncalibrated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
