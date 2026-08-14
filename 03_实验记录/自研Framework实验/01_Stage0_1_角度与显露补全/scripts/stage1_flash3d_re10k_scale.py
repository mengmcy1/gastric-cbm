#!/usr/bin/env python3
"""Recover RealEstate10K translation scale using Flash3D's official method."""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import pickle
import random
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


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def metadata_rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[0].startswith(("http://", "https://")):
        raise ValueError(f"unexpected RealEstate10K metadata format: {path}")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if any(len(row) != 19 for row in rows):
        raise ValueError(f"expected timestamp + 18 values in every row: {path}")
    return rows


def sparse_xyd(
    metadata_row: list[str], sparse_pcl: dict[str, object], frame_index: int, image_wh: tuple[int, int]
) -> tuple[torch.Tensor, dict[str, object]]:
    pose = np.asarray([float(value) for value in metadata_row[7:]], dtype=np.float64).reshape(3, 4)
    point_ids = np.asarray(sparse_pcl["p3D_ids"][frame_index])
    xys = np.asarray(sparse_pcl["xys"][frame_index])
    visible = point_ids != -1
    point_ids = point_ids[visible]
    xys = xys[visible]
    xyz = np.asarray(sparse_pcl["xyz"])[point_ids]
    xyz_camera = xyz @ pose[:, :3].T + pose[:, 3]
    z = xyz_camera[:, 2:3]
    wh = np.asarray(image_wh, dtype=np.float64)[None]
    xy_grid = (xys / wh - 0.5) * 2.0
    xyd = np.concatenate([xy_grid, z], axis=1)
    return torch.from_numpy(xyd).to(torch.float32), {
        "visible_sparse_points": int(visible.sum()),
        "positive_colmap_z_points": int((z[:, 0] > 0).sum()),
        "positive_colmap_z_fraction": float((z[:, 0] > 0).mean()),
    }


def ransac_scale_with_diagnostics(
    depth: torch.Tensor,
    sparse_depth: torch.Tensor,
    *,
    seed: int,
    num_iterations: int = 1000,
    sample_size: int = 5,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Equivalent to Flash3D estimate_depth_scale_ransac plus diagnostics."""
    sparse_depth = sparse_depth.to(depth.device)
    xy = sparse_depth[:, :2][None, None]
    z = sparse_depth[:, 2]
    predicted = F.grid_sample(depth, xy, align_corners=False).squeeze()
    valid = (z > 1e-7) & (predicted > 1e-7) & torch.isfinite(z) & torch.isfinite(predicted)
    z = z[valid]
    predicted = predicted[valid]
    if z.numel() < 10:
        raise RuntimeError(f"only {z.numel()} positive sparse correspondences")

    generator = random.Random(seed)
    best_scale: torch.Tensor | None = None
    best_count = -1
    for _ in range(num_iterations):
        indices = generator.sample(range(z.shape[0]), min(sample_size, z.shape[0]))
        scale = (predicted[indices].log() - z[indices].log()).mean().exp()
        inliers = torch.abs(predicted.log() - (z * scale).log()) < threshold
        count = int(inliers.sum().item())
        if count > best_count:
            best_scale = scale
            best_count = count
    assert best_scale is not None
    residual = torch.abs(predicted.log() - (z * best_scale).log())
    return best_scale, {
        "seed": seed,
        "num_iterations": num_iterations,
        "sample_size": sample_size,
        "log_residual_threshold": threshold,
        "valid_correspondences": int(z.numel()),
        "inlier_count": int((residual < threshold).sum().item()),
        "inlier_fraction": float((residual < threshold).float().mean().item()),
        "median_abs_log_residual": float(residual.median().item()),
        "p95_abs_log_residual": float(torch.quantile(residual, 0.95).item()),
        "colmap_z_median": float(z.median().item()),
        "predicted_depth_median": float(predicted.median().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-record", type=Path, required=True)
    parser.add_argument("--pcl-dir", type=Path, required=True)
    parser.add_argument("--pcl-record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.dataset_record = args.dataset_record.resolve()
    args.pcl_dir = args.pcl_dir.resolve()
    args.pcl_record = args.pcl_record.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    dataset = json.loads(args.dataset_record.read_text(encoding="utf-8"))
    pcl_receipt = json.loads(args.pcl_record.read_text(encoding="utf-8"))
    if dataset.get("status") != "pass" or pcl_receipt.get("status") != "pass":
        raise ValueError("dataset and sparse-COLMAP receipts must both have status=pass")

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
    load_result = model.load_state_dict(adapted_state, strict=False)
    external_prefix = "models.unidepth_extended.unidepth."
    unexpected_missing = [key for key in load_result.missing_keys if not key.startswith(external_prefix)]
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={unexpected_missing}, unexpected={load_result.unexpected_keys}"
        )

    device = torch.device("cuda:0")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    target_h, target_w = int(cfg.dataset.height), int(cfg.dataset.width)
    pad = int(cfg.dataset.pad_border_aug)
    sequence_results: list[dict[str, object]] = []
    total_started = time.perf_counter()

    for sequence_number, sequence in enumerate(dataset["sequences"]):
        sequence_id = sequence["sequence"]
        source = sequence["frames"]["source"]
        source_path = Path.cwd() / source["path"]
        metadata_path = Path.cwd() / sequence["metadata"]["path"]
        pcl_path = args.pcl_dir / f"{sequence_id}.pickle.gz"
        rows = metadata_rows(metadata_path)
        frame_index = int(source["frame_index"])
        if rows[frame_index][0] != str(source["timestamp_microseconds"]):
            raise ValueError(f"timestamp/index mismatch for {sequence_id}")
        with gzip.open(pcl_path, "rb") as stream:
            sparse_pcl = pickle.load(stream)
        if len(rows) != len(sparse_pcl["xys"]):
            raise ValueError(f"metadata/PCL frame count mismatch for {sequence_id}")

        image = Image.open(source_path).convert("RGB")
        original_wh = image.size
        normalized_intrinsics = [float(value) for value in rows[frame_index][1:5]]
        resized = TF.resize(
            image, [target_h, target_w], interpolation=TF.InterpolationMode.LANCZOS
        )
        color = TF.to_tensor(resized)
        color_padded = F.pad(color, (pad, pad, pad, pad), mode="replicate")[None]
        k = torch.eye(3, dtype=torch.float32)[None]
        k[:, 0, 0] = normalized_intrinsics[0] * target_w
        k[:, 1, 1] = normalized_intrinsics[1] * target_h
        k[:, 0, 2] = normalized_intrinsics[2] * (target_w + 2 * pad)
        k[:, 1, 2] = normalized_intrinsics[3] * (target_h + 2 * pad)
        inputs = {
            ("color_aug", 0, 0): color_padded.to(device),
            ("K_src", 0): k.to(device),
        }
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.models["unidepth_extended"](inputs)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        depth_padded = outputs[("depth", 0)].float()
        depth_first = depth_padded[[0], :, pad : pad + target_h, pad : pad + target_w]
        xyd, sparse_summary = sparse_xyd(rows[frame_index], sparse_pcl, frame_index, original_wh)
        scale, diagnostics = ransac_scale_with_diagnostics(
            depth_first, xyd, seed=20260814 + sequence_number
        )
        sequence_results.append(
            {
                "sequence": sequence_id,
                "source_frame_index": frame_index,
                "source_timestamp_microseconds": source["timestamp_microseconds"],
                "source_image": portable(source_path),
                "source_image_sha256": sha256(source_path),
                "metadata": portable(metadata_path),
                "metadata_sha256": sha256(metadata_path),
                "sparse_pcl": portable(pcl_path),
                "sparse_pcl_sha256": sha256(pcl_path),
                "metadata_frames": len(rows),
                "pcl_frames": len(sparse_pcl["xys"]),
                "image_wh": list(original_wh),
                "normalized_intrinsics_fx_fy_cx_cy": normalized_intrinsics,
                "network_unpadded_wh": [target_w, target_h],
                "translation_scale_predicted_depth_per_colmap_unit": float(scale.item()),
                "runtime_seconds": elapsed,
                "sparse_geometry": sparse_summary,
                "ransac": diagnostics,
            }
        )

    torch.cuda.synchronize(device)
    scales = [result["translation_scale_predicted_depth_per_colmap_unit"] for result in sequence_results]
    status = "pass"
    if not all(np.isfinite(scales)) or not all(scale > 0 for scale in scales):
        status = "fail"
    if not all(result["ransac"]["valid_correspondences"] >= 10 for result in sequence_results):
        status = "fail"
    record = {
        "schema_version": "stage1.4-re10k-scale-v1",
        "status": status,
        "scope": "official Flash3D source-frame UniDepth/COLMAP RANSAC translation scale",
        "producer_machine_id": "linux5080",
        "device": {
            "logical": "cuda:0",
            "physical_selected_by_cuda_visible_devices": args.physical_gpu_index,
            "name": torch.cuda.get_device_name(device),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flash3d_commit": git_commit(args.repo),
            "unidepth_commit": git_commit(args.repo.parent / "UniDepth"),
            "nystrom_source_commit": XFORMERS_SOURCE_COMMIT,
        },
        "inputs": {
            "dataset_record": portable(args.dataset_record),
            "dataset_record_sha256": sha256(args.dataset_record),
            "pcl_record": portable(args.pcl_record),
            "pcl_record_sha256": sha256(args.pcl_record),
            "checkpoint": portable(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "method": {
            "depth_layer": "first of two Flash3D Gaussian depth layers",
            "padding_crop": pad,
            "grid_sample_align_corners": False,
            "ransac_iterations": 1000,
            "ransac_sample_size": 5,
            "ransac_log_residual_threshold": 0.1,
            "pose_application": "multiply relative camera translation by recovered scale",
        },
        "summary": {
            "sequence_count": len(sequence_results),
            "all_metadata_pcl_frame_counts_match": all(
                result["metadata_frames"] == result["pcl_frames"] for result in sequence_results
            ),
            "all_scales_positive_finite": all(np.isfinite(scales)) and all(scale > 0 for scale in scales),
            "minimum_ransac_inlier_fraction": min(
                result["ransac"]["inlier_fraction"] for result in sequence_results
            ),
            "total_runtime_seconds": time.perf_counter() - total_started,
        },
        "sequences": sequence_results,
        "quality_boundary": (
            "scale pass enables metric-consistent target pose; it does not establish novel-view RGB quality, "
            "dense hidden-depth accuracy, or OutsideFOV support"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "device": record["device"], "summary": record["summary"], "scales": scales}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
