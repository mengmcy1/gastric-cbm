#!/usr/bin/env python3
"""Render RealEstate10K targets with Flash3D layer-1 vs layer-1+2."""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from flash3d_xformers_compat import force_dino_reference_attention, install_unidepth_compatibility
from stage1_flash3d_to_clb import metadata_rows, normalized_k, portable, sha256, w2c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-record", type=Path, required=True)
    parser.add_argument("--scale-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    return parser.parse_args()


def image_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    resized = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.LANCZOS)
    return TF.to_tensor(resized)


def metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    if mask is None:
        error = prediction - target
        mse = float(np.mean(error ** 2))
        mae = float(np.mean(np.abs(error)))
        ssim = float(structural_similarity(target, prediction, data_range=1.0, channel_axis=2))
        count = int(target.shape[0] * target.shape[1])
    else:
        count = int(mask.sum())
        if count == 0:
            return {"pixels": 0, "mae": None, "mse": None, "psnr": None, "ssim": None}
        error = prediction[mask] - target[mask]
        mse = float(np.mean(error ** 2))
        mae = float(np.mean(np.abs(error)))
        ssim = None
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    return {"pixels": count, "mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(path)


def make_review(path: Path, target: np.ndarray, baseline: np.ndarray, full: np.ndarray, effect: np.ndarray) -> None:
    height, width = target.shape[:2]
    canvas = Image.new("RGB", (width * 4, height), "black")
    arrays = [target, baseline, full, np.clip(effect * 4.0, 0, 1)]
    labels = ["target", "layer 1", "layers 1+2", "|delta| x4"]
    for column, array in enumerate(arrays):
        canvas.paste(Image.fromarray(np.round(array * 255).astype(np.uint8)), (column * width, 0))
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * width + 7, 6), label, fill="white", stroke_width=2, stroke_fill="black")
    canvas.save(path)


def main() -> int:
    args = parse_args()
    for name in ("repo", "checkpoint", "dataset_record", "scale_record", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    dataset = json.loads(args.dataset_record.read_text(encoding="utf-8"))
    scale_record = json.loads(args.scale_record.read_text(encoding="utf-8"))
    scales = {
        value["sequence"]: float(value["translation_scale_predicted_depth_per_colmap_unit"])
        for value in scale_record["sequences"]
    }
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
    adapted = {
        key: current[key].clone() if "backproject_depth" in key else value
        for key, value in checkpoint["model"].items()
    }
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
    total_started = time.perf_counter()

    for sequence in dataset["sequences"]:
        sequence_id = sequence["sequence"]
        sequence_dir = args.output_dir / sequence_id
        sequence_dir.mkdir()
        rows = metadata_rows(Path.cwd() / sequence["metadata"]["path"])
        indices = {
            "center": int(sequence["frames"]["source"]["frame_index"]),
            "left": int(sequence["frames"]["left"]["frame_index"]),
            "right": int(sequence["frames"]["right"]["frame_index"]),
        }
        source = image_tensor(Path.cwd() / sequence["frames"]["source"]["path"], target_h, target_w)
        padded_source = F.pad(source, (pad, pad, pad, pad), mode="replicate")[None]
        source_row = rows[indices["center"]]
        k_source = normalized_k(source_row, target_w + 2 * pad, target_h + 2 * pad)
        k_source[0, 0] = float(source_row[1]) * target_w
        k_source[1, 1] = float(source_row[2]) * target_h
        inputs: dict[object, object] = {
            "target_frame_ids": ["left", "right"],
            ("color", 0, 0): source[None].to(device),
            ("color_aug", 0, 0): padded_source.to(device),
            ("K_src", 0): torch.from_numpy(k_source).float()[None].to(device),
            ("depth_sparse", 0): torch.zeros((1, 10, 3), dtype=torch.float32, device=device),
            ("scale_colmap", 0): torch.tensor([scales[sequence_id]], dtype=torch.float32, device=device),
        }
        for name in ("center", "left", "right"):
            row = rows[indices[name]]
            pose_w2c = torch.from_numpy(w2c(row)).float()[None].to(device)
            inputs[("T_w2c", name if name != "center" else 0)] = pose_w2c
            inputs[("T_c2w", name if name != "center" else 0)] = torch.linalg.inv(pose_w2c)
            inputs[("K_tgt", name if name != "center" else 0)] = torch.from_numpy(
                normalized_k(row, target_w, target_h)
            ).float()[None].to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(inputs)
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - started
        full = {
            side: outputs[("color_gauss", side, 0)].detach().clone()
            for side in ("left", "right")
        }
        baseline_outputs = dict(outputs)
        baseline_opacity = outputs["gauss_opacity"].clone()
        baseline_opacity[1:: int(cfg.model.gaussians_per_pixel)] = 0.0
        baseline_outputs["gauss_opacity"] = baseline_opacity
        with torch.inference_mode():
            model.render_images(inputs, baseline_outputs)
        torch.cuda.synchronize(device)

        for side in ("left", "right"):
            target_path = Path.cwd() / sequence["frames"][side]["path"]
            target = image_tensor(target_path, target_h, target_w).permute(1, 2, 0).numpy()
            full_np = full[side][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            baseline_np = baseline_outputs[("color_gauss", side, 0)][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            effect = np.abs(full_np - baseline_np)
            effect_mask = np.max(effect, axis=2) > (1.0 / 255.0)
            crop_mask = np.zeros((target_h, target_w), dtype=bool)
            y0, y1 = math.ceil(0.05 * target_h), math.floor(0.95 * target_h)
            x0, x1 = math.ceil(0.05 * target_w), math.floor(0.95 * target_w)
            crop_mask[y0:y1, x0:x1] = True
            baseline_whole = metrics(baseline_np[crop_mask].reshape(y1-y0, x1-x0, 3), target[crop_mask].reshape(y1-y0, x1-x0, 3))
            full_whole = metrics(full_np[crop_mask].reshape(y1-y0, x1-x0, 3), target[crop_mask].reshape(y1-y0, x1-x0, 3))
            contribution = effect_mask & crop_mask
            baseline_effect = metrics(baseline_np, target, contribution)
            full_effect = metrics(full_np, target, contribution)
            save_rgb(sequence_dir / f"{side}_target.png", target)
            save_rgb(sequence_dir / f"{side}_layer1.png", baseline_np)
            save_rgb(sequence_dir / f"{side}_layers12.png", full_np)
            save_rgb(sequence_dir / f"{side}_delta_x4.png", np.clip(effect * 4, 0, 1))
            make_review(sequence_dir / f"{side}_review.png", target, baseline_np, full_np, effect)
            results.append(
                {
                    "sequence": sequence_id,
                    "side": side,
                    "angle_deg": float(sequence["frames"][side]["angle_deg"]),
                    "target_image": portable(target_path),
                    "target_image_sha256": sha256(target_path),
                    "translation_scale": scales[sequence_id],
                    "inference_and_full_render_seconds": inference_seconds,
                    "effect_region_definition": "max_abs(full_layers12-layer1)>1/255 within official 5% crop",
                    "effect_region_pixels": int(contribution.sum()),
                    "effect_region_fraction_of_crop": float(contribution.sum() / crop_mask.sum()),
                    "official_crop_metrics": {"layer1": baseline_whole, "layers12": full_whole},
                    "effect_region_metrics": {"layer1": baseline_effect, "layers12": full_effect},
                    "delta_layers12_minus_layer1": {
                        "psnr_db": full_whole["psnr"] - baseline_whole["psnr"],
                        "ssim": full_whole["ssim"] - baseline_whole["ssim"],
                        "mae_effect_region": full_effect["mae"] - baseline_effect["mae"] if full_effect["mae"] is not None else None,
                    },
                    "review": portable(sequence_dir / f"{side}_review.png"),
                }
            )

    psnr_deltas = [item["delta_layers12_minus_layer1"]["psnr_db"] for item in results]
    ssim_deltas = [item["delta_layers12_minus_layer1"]["ssim"] for item in results]
    effect_mae_deltas = [item["delta_layers12_minus_layer1"]["mae_effect_region"] for item in results]
    record = {
        "schema_version": "stage1.4-flash3d-layer-ab-v1",
        "status": "complete_metrics_partial_protocol",
        "scope": "same Flash3D source prediction and scaled poses; layer1 baseline vs layers1+2",
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
        },
        "metrics_scope": {
            "computed": ["PSNR", "SSIM", "MAE"],
            "pending_before_protocol_quality_gate": ["LPIPS", "DISTS", "independent occlusion_hidden truth mask"],
            "crop": "official Flash3D 5% border crop",
        },
        "summary": {
            "target_views": len(results),
            "mean_psnr_delta_db": float(np.mean(psnr_deltas)),
            "median_psnr_delta_db": float(np.median(psnr_deltas)),
            "psnr_improved_views": int(np.sum(np.asarray(psnr_deltas) > 0)),
            "mean_ssim_delta": float(np.mean(ssim_deltas)),
            "ssim_improved_views": int(np.sum(np.asarray(ssim_deltas) > 0)),
            "mean_effect_region_mae_delta": float(np.mean(effect_mae_deltas)),
            "effect_region_mae_improved_views": int(np.sum(np.asarray(effect_mae_deltas) < 0)),
            "total_runtime_seconds": time.perf_counter() - total_started,
        },
        "views": results,
        "quality_boundary": (
            "PSNR/SSIM/MAE A/B is diagnostic; protocol quality pass requires LPIPS, DISTS, semantic region truth, "
            "and confidence calibration. OutsideFOV remains unsupported."
        ),
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "device": record["device"], "summary": record["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
