#!/usr/bin/env python3
"""Minimal Flash3D layer-output smoke without target-view rendering."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from hydra import compose, initialize_config_dir
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--focal-px", type=float, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.image = args.image.resolve()
    args.output = args.output.resolve()

    original_hub_load = torch.hub.load

    @functools.wraps(original_hub_load)
    def local_unidepth_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs):
        if repo_or_dir == "lpiccinelli-eth/UniDepth":
            local_repo = args.repo.parent / "UniDepth"
            if not local_repo.is_dir():
                raise FileNotFoundError(f"local UniDepth checkout missing: {local_repo}")
            hub_kwargs.pop("trust_repo", None)
            hub_kwargs.pop("force_reload", None)
            hub_kwargs["source"] = "local"
            return original_hub_load(str(local_repo), model, *hub_args, **hub_kwargs)
        return original_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs)

    torch.hub.load = local_unidepth_hub_load

    with initialize_config_dir(version_base=None, config_dir=str(args.repo / "configs")):
        cfg = compose(config_name="config", overrides=["+experiment=layered_re10k"])
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 0
    cfg.model.gaussian_rendering = False
    # The official checkpoint contains these weights; avoid a redundant
    # torchvision download during construction.
    cfg.model.backbone.weights_init = "scratch"

    from models.model import GaussianPredictor

    model = GaussianPredictor(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Match Flash3D's official load_model(): these non-learned buffers encode
    # the configured batch size and therefore cannot be copied from its
    # training checkpoint (batch 16) into this batch-1 smoke run.
    current_state = model.state_dict()
    adapted_state = {
        key: current_state[key].clone() if "backproject_depth" in key else value
        for key, value in checkpoint["model"].items()
    }
    load_result = model.load_state_dict(adapted_state, strict=False)

    image = Image.open(args.image).convert("RGB")
    original_w, original_h = image.size
    target_h, target_w = int(cfg.dataset.height), int(cfg.dataset.width)
    pad = int(cfg.dataset.pad_border_aug)
    resized = TF.resize(image, [target_h, target_w], interpolation=TF.InterpolationMode.LANCZOS)
    color = TF.to_tensor(resized)
    color_padded = torch.nn.functional.pad(color, (pad, pad, pad, pad), mode="replicate")[None]

    k = torch.eye(3, dtype=torch.float32)[None]
    k[:, 0, 0] = args.focal_px * target_w / original_w
    k[:, 1, 1] = args.focal_px * target_h / original_h
    k[:, 0, 2] = target_w / 2.0 + pad
    k[:, 1, 2] = target_h / 2.0 + pad
    inputs = {
        ("color_aug", 0, 0): color_padded,
        ("K_src", 0): k,
    }

    device = torch.device("cuda:0")
    model.to(device).eval()
    inputs = {key: value.to(device) for key, value in inputs.items()}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.models["unidepth_extended"](inputs)
        model.compute_gauss_means(inputs, outputs)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    depth = outputs[("depth", 0)].float()
    depth_layers = depth.reshape(1, int(cfg.model.gaussians_per_pixel), 1, *depth.shape[-2:])
    layer_1 = depth_layers[:, 0]
    layer_2 = depth_layers[:, 1]
    increment = layer_2 - layer_1
    opacities = outputs["gauss_opacity"].float().reshape(
        1, int(cfg.model.gaussians_per_pixel), 1, *depth.shape[-2:]
    )
    tensors = {
        "depth": depth,
        "gauss_means": outputs["gauss_means"],
        "gauss_opacity": outputs["gauss_opacity"],
        "gauss_scaling": outputs["gauss_scaling"],
        "gauss_rotation": outputs["gauss_rotation"],
        "gauss_features_dc": outputs["gauss_features_dc"],
    }
    tensor_summary = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite_fraction": float(torch.isfinite(value.float()).float().mean().cpu()),
        }
        for name, value in tensors.items()
    }
    result = {
        "schema_version": "stage1.4-flash3d-layer-smoke-v1",
        "status": "pass",
        "scope": "official checkpoint layer schema; no novel-view quality conclusion",
        "device": {
            "logical": "cuda:0",
            "physical_selected_by_cuda_visible_devices": args.physical_gpu_index,
            "name": torch.cuda.get_device_name(device),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "environment": {
            "python_torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flash3d_commit": git_commit(args.repo),
            "unidepth_commit": git_commit(args.repo.parent / "UniDepth"),
        },
        "inputs": {
            "image": portable_path(args.image),
            "image_sha256": sha256(args.image),
            "image_original_wh": [original_w, original_h],
            "image_network_wh_with_padding": [target_w + 2 * pad, target_h + 2 * pad],
            "focal_px_original": args.focal_px,
            "checkpoint": portable_path(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "checkpoint_load": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "runtime_seconds": elapsed,
        "outputs": tensor_summary,
        "layer_checks": {
            "layer_count": int(cfg.model.gaussians_per_pixel),
            "positive_finite_depth_fraction": float(((depth > 0) & torch.isfinite(depth)).float().mean().cpu()),
            "positive_second_layer_increment_fraction": float((increment > 0).float().mean().cpu()),
            "minimum_second_layer_increment": float(increment.min().cpu()),
            "opacity_min": float(opacities.min().cpu()),
            "opacity_max": float(opacities.max().cpu()),
        },
        "capability_boundary": {
            "outside_source_fov_supported": False,
            "native_independent_confidence": False,
            "opacity_is_confidence": False,
        },
    }
    if load_result.missing_keys or load_result.unexpected_keys or result["layer_checks"]["positive_second_layer_increment_fraction"] < 1.0:
        result["status"] = "fail"
    if any(summary["finite_fraction"] < 1.0 for summary in tensor_summary.values()):
        result["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "device": result["device"], "runtime_seconds": elapsed, "layer_checks": result["layer_checks"]}, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
