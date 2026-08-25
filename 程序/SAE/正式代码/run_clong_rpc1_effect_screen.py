#!/usr/bin/env python3
"""对1150个RP-B Anchor执行三seed residual-preserving effect screen。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from clong_rpa_gpu_identity import LEVELS, downstream_levels, identity_passes, residual_preserving_noop
from clong_rpa_prepare_seed import checkpoint_path, load_sae
from clong_rpb_core import SEEDS, file_sha256
from clong_rpc_core import (
    aggregate_active_by_patient,
    aggregate_image_matrix_by_patient,
    intervention_block,
    label_seed_metrics,
)
from clong_sae_discovery import (
    FROZEN_IMAGE_THRESHOLD,
    FROZEN_PATIENT_THRESHOLD,
    load_clong_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
RPA_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824"
CACHE_ROOT = RPA_ROOT / "analysis_cache"
SPATIAL_ROOT = PROJECT_ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
RPB_ROOT = PROJECT_ROOT / "结果/SAE/RP_B_Technical_20260825"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_C1_Effect_Screen_20260825"
PROTOCOL_PATH = CODE_ROOT / "rpc_effect_screen_protocol_v1.json"
IDENTITY_PROTOCOL_PATH = CODE_ROOT / "rpa_gpu_identity_protocol_v1.json"


def parse_args() -> argparse.Namespace:
    """解析正式或隔离debug参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--anchor-block", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-tag", default="debug_v2")
    parser.add_argument("--debug-images", type=int, default=16)
    parser.add_argument("--debug-anchors", type=int, default=8)
    return parser.parse_args()


def model_weights(model: nn.Module, device: torch.device) -> tuple[torch.Tensor, ...]:
    """提取冻结attention头和cancer margin线性方向。"""
    linear = next(module for module in reversed(list(model.classifier.modules())) if isinstance(module, nn.Linear))
    attention_weight = model.attention_head.weight.reshape(-1).to(device)
    attention_bias = model.attention_head.bias.reshape(()).to(device)
    margin_weight = (linear.weight[1] - linear.weight[0]).to(device)
    margin_bias = (linear.bias[1] - linear.bias[0]).reshape(()).to(device)
    return attention_weight, attention_bias, margin_weight, margin_bias


@torch.no_grad()
def run_identity_gate(
    model: nn.Module, spatial: np.ndarray, metadata: pd.DataFrame,
    device: torch.device, debug: bool,
) -> dict:
    """三seed执行alpha=1的五级identity gate。"""
    identity_protocol = json.loads(IDENTITY_PROTOCOL_PATH.read_text(encoding="utf-8"))
    patient_ids = np.sort(metadata.patient_id.astype(str).unique())[:32]
    selected = np.flatnonzero(metadata.patient_id.astype(str).isin(patient_ids).to_numpy())
    if debug:
        selected = selected[: min(8, len(selected))]
    output = {}
    for seed in SEEDS:
        sae = load_sae(
            checkpoint_path(argparse.Namespace(
                debug=False, debug_patients_per_class=0, debug_epochs=0, seed=seed,
            )), device,
        )
        passed = {level: True for level in LEVELS}
        max_abs = {level: 0.0 for level in LEVELS}
        for start in range(0, len(selected), 16):
            features = torch.from_numpy(np.asarray(spatial[selected[start:start + 16]]).copy()).to(device)
            candidate = residual_preserving_noop(features, sae, k=1024)
            reference_levels = downstream_levels(features, model)
            candidate_levels = downstream_levels(candidate, model)
            for level in LEVELS:
                difference = float((candidate_levels[level] - reference_levels[level]).abs().max())
                max_abs[level] = max(max_abs[level], difference)
                if device.type == "cuda":
                    tolerance = identity_protocol["frozen_tolerances"][level]
                    passed[level] &= identity_passes(
                        reference_levels[level], candidate_levels[level],
                        float(tolerance["atol"]), float(tolerance["rtol"]),
                    )
        if device.type == "cuda" and not all(passed.values()):
            raise RuntimeError(f"seed{seed} alpha=1 identity失败: {passed}")
        output[str(seed)] = {"passed": passed, "max_abs_error": max_abs}
        del sae
    return output


def create_memmaps(root: Path, images: int, anchors: int) -> dict[str, np.memmap]:
    """创建一个seed的图像级输出矩阵。"""
    specs = {
        "delta_margin": np.float32,
        "delta_probability": np.float32,
        "attention_cosine": np.float32,
        "attention_l1": np.float32,
        "peak_changed": np.bool_,
        "active_image": np.bool_,
    }
    return {
        name: np.lib.format.open_memmap(root / f"image_{name}.npy", mode="w+", dtype=dtype, shape=(images, anchors))
        for name, dtype in specs.items()
    }


@torch.no_grad()
def screen_seed(
    seed: int, anchors: pd.DataFrame, spatial: np.ndarray, images: pd.DataFrame,
    patients: pd.DataFrame, model: nn.Module, device: torch.device,
    image_batch: int, anchor_block: int, target: Path,
) -> pd.DataFrame:
    """流式计算一个seed全部Anchor的图像和患者效应。"""
    cache = CACHE_ROOT / f"seed{seed}"
    activations = np.load(cache / "train_image_activations.npy", mmap_mode="r")
    decoder_all = np.load(cache / "decoder_weight.npy", mmap_mode="r")
    feature_ids = anchors[f"feature_{seed}"].to_numpy(int)
    output = target / f"seed{seed}"
    output.mkdir(parents=True)
    maps = create_memmaps(output, len(images), len(anchors))
    original_margin = np.empty(len(images), dtype=np.float32)
    original_probability = np.empty(len(images), dtype=np.float32)
    attention_weight, attention_bias, margin_weight, margin_bias = model_weights(model, device)

    for image_start in range(0, len(images), image_batch):
        image_stop = min(image_start + image_batch, len(images))
        features = torch.from_numpy(np.asarray(spatial[image_start:image_stop]).copy()).to(device)
        for anchor_start in range(0, len(anchors), anchor_block):
            anchor_stop = min(anchor_start + anchor_block, len(anchors))
            ids = feature_ids[anchor_start:anchor_stop]
            hidden_np = np.take(
                activations[image_start:image_stop], ids, axis=2,
            )
            hidden = torch.from_numpy(np.asarray(hidden_np).copy()).to(device)
            decoder = torch.from_numpy(np.asarray(decoder_all[ids]).copy()).to(device)
            result = intervention_block(
                features, hidden, decoder, attention_weight, attention_bias,
                margin_weight, margin_bias, alpha=0.0,
            )
            if anchor_start == 0:
                original_margin[image_start:image_stop] = result["original_margin"].cpu().numpy()
                original_probability[image_start:image_stop] = result["original_probability"].cpu().numpy()
            for name, destination in maps.items():
                destination[image_start:image_stop, anchor_start:anchor_stop] = result[name].cpu().numpy()
        print(f"seed{seed}: images {image_stop}/{len(images)}", flush=True)
    for destination in maps.values():
        destination.flush()
    np.save(output / "original_image_margin.npy", original_margin)
    np.save(output / "original_image_probability.npy", original_probability)

    patient_ids = patients.patient_id.astype(str).to_numpy()
    image_patient_ids = images.patient_id.astype(str).to_numpy()
    patient_outputs = {}
    for name in ("delta_margin", "delta_probability", "attention_cosine", "attention_l1", "peak_changed"):
        patient_outputs[name] = aggregate_image_matrix_by_patient(
            np.asarray(maps[name]), image_patient_ids, patient_ids,
        )
        np.save(output / f"patient_{name}.npy", patient_outputs[name])
    active_patient = aggregate_active_by_patient(
        np.asarray(maps["active_image"]), image_patient_ids, patient_ids,
    )
    np.save(output / "patient_active.npy", active_patient)
    original_patient_probability = aggregate_image_matrix_by_patient(
        original_probability[:, None], image_patient_ids, patient_ids,
    )[:, 0]
    np.save(output / "original_patient_probability.npy", original_patient_probability)

    rows = label_seed_metrics(
        patient_outputs["delta_margin"], patient_outputs["delta_probability"],
        patient_outputs["attention_cosine"], patient_outputs["attention_l1"],
        patient_outputs["peak_changed"], active_patient,
        patients.label.to_numpy(int), original_patient_probability,
        FROZEN_PATIENT_THRESHOLD,
    )
    metrics = pd.DataFrame(rows)
    metrics["anchor_id"] = anchors.anchor_id.to_numpy()
    metrics["seed"] = seed
    metrics["feature_id"] = feature_ids
    original_image_prediction = original_probability >= FROZEN_IMAGE_THRESHOLD
    image_labels = images.label.to_numpy(int)
    for column in range(len(anchors)):
        changed = original_probability + np.asarray(maps["delta_probability"][:, column])
        changed_prediction = changed >= FROZEN_IMAGE_THRESHOLD
        metrics.loc[column, "image_flip_rate"] = float((changed_prediction != original_image_prediction).mean())
        metrics.loc[column, "image_cancer_to_noncancer_flip_count"] = int(
            ((image_labels == 1) & original_image_prediction & ~changed_prediction).sum()
        )
        metrics.loc[column, "image_noncancer_to_cancer_flip_count"] = int(
            ((image_labels == 0) & ~original_image_prediction & changed_prediction).sum()
        )
    metrics.to_csv(output / "seed_anchor_metrics.csv", index=False)
    return metrics


def main() -> None:
    """核验输入、执行identity后运行三seed effect screen。"""
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_effect_results_2026-08-25":
        raise RuntimeError("RP-C1协议未冻结")
    if not args.debug and args.device != "cuda":
        raise ValueError("正式RP-C1必须使用CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    device = torch.device(args.device)
    target = OUTPUT_ROOT / args.debug_tag if args.debug else OUTPUT_ROOT / "formal"
    if target.exists():
        raise FileExistsError(f"RP-C1输出已存在: {target}")

    anchors = pd.read_csv(
        RPA_ROOT / "full_train_matching/formal/development_anchors.csv", dtype={"anchor_id": str},
    )
    rpb_master = pd.read_csv(RPB_ROOT / "anchor_master/anchor_master.csv", dtype={"anchor_id": str})
    if len(anchors) != 1150 or not anchors.anchor_id.equals(rpb_master.anchor_id):
        raise RuntimeError("RP-C1 Anchor输入与RP-B不一致")
    images = pd.read_csv(CACHE_ROOT / "seed42/train_images.csv", dtype={"patient_id": str})
    patients = pd.read_csv(CACHE_ROOT / "seed42/train_patients.csv", dtype={"patient_id": str})
    metadata = pd.read_csv(SPATIAL_ROOT / "train_metadata.csv", dtype={"patient_id": str})
    spatial = np.load(SPATIAL_ROOT / "train_spatial_features.npy", mmap_mode="r")
    if not images.relative_path.astype(str).equals(metadata.relative_path.astype(str)):
        raise RuntimeError("RP-A激活与空间特征图像顺序不一致")
    for seed in SEEDS:
        current = pd.read_csv(CACHE_ROOT / f"seed{seed}/train_images.csv", dtype={"patient_id": str})
        if not images.equals(current):
            raise RuntimeError(f"seed{seed}图像表不一致")
    if args.debug:
        per_label = max(1, args.debug_images // 2)
        indices = np.sort(np.concatenate([
            np.flatnonzero(images.label.to_numpy(int) == label)[:per_label]
            for label in (0, 1)
        ]))
        anchor_count = min(args.debug_anchors, len(anchors))
        images = images.iloc[indices].reset_index(drop=True)
        metadata = metadata.iloc[indices].reset_index(drop=True)
        spatial = np.asarray(spatial[indices])
        anchors = anchors.iloc[:anchor_count].reset_index(drop=True)
        selected_patients = images.patient_id.unique()
        patients = patients[patients.patient_id.isin(selected_patients)].reset_index(drop=True)

    model = load_clong_model(device)
    identity = run_identity_gate(model, spatial, metadata, device, args.debug)
    target.mkdir(parents=True)
    (target / "identity_gate.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    seed_frames = []
    for seed in SEEDS:
        seed_frames.append(screen_seed(
            seed, anchors, spatial, images, patients, model, device,
            args.image_batch, args.anchor_block, target,
        ))
    combined = pd.concat(seed_frames, ignore_index=True)
    combined.to_csv(target / "seed_anchor_metrics.csv", index=False)
    config = {
        "status": "rpc1_effect_screen_complete",
        "debug": bool(args.debug),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "rpb_protocol_sha256": file_sha256(CODE_ROOT / "rpb_technical_protocol_v1.json"),
        "anchor_count": int(len(anchors)),
        "image_count": int(len(images)),
        "patient_count": int(len(patients)),
        "seeds": list(SEEDS),
        "identity_gate": identity,
        "input_sha256": {
            "development_anchors": file_sha256(RPA_ROOT / "full_train_matching/formal/development_anchors.csv"),
            "rpb_anchor_master": file_sha256(RPB_ROOT / "anchor_master/anchor_master.csv"),
            "spatial_metadata": file_sha256(SPATIAL_ROOT / "train_metadata.csv"),
            **{f"seed{seed}_config": file_sha256(CACHE_ROOT / f"seed{seed}/config.json") for seed in SEEDS},
        },
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (target / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(f"RP-C1 effect screen完成: {target}")


if __name__ == "__main__":
    main()
