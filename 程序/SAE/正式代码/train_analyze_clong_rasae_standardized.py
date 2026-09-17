#!/usr/bin/env python3
"""训练一次逐通道标准化RA-SAE候选，并与当前1701模型做原坐标分析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE, calibrate_initial_encoder  # noqa: E402
from clong_rpc_core import intervention_block  # noqa: E402
from clong_s2b_core import (  # noqa: E402
    attention_from_features, normalized_margin_mse, patch_position_weights,
    pooled_from_features,
)
from clong_sae_discovery import (  # noqa: E402
    FROZEN_IMAGE_THRESHOLD, FROZEN_PATIENT_THRESHOLD, patient_class_weights,
)
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
REFERENCE = BASE / "duration100"
DECISION = BASE / "decision_alignment"
SEED = 1701
K_LIST = (64, 128, 256)
K = 256
WIDTH = 2560
EPOCHS = 100
BATCH = 16
BASE_LR = 1e-4
WARMUP_EPOCHS = 2
GAMMA_MARGIN = 0.1


def json_write(path: Path, value: dict) -> None:
    """写入不含NaN的UTF-8 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def fixed_train_data(device: torch.device) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """读取当前duration100固定196张训练子集、缓存和归一化权重。"""
    frame = pd.read_csv(REFERENCE / "train_subset.csv")
    if len(frame) != 196 or frame.patient_id.nunique() != 128:
        raise RuntimeError("固定训练子集不是196图/128人")
    data = read_subset(frame, "train", device)
    weights = patient_class_weights(frame).astype(np.float64)
    weights /= weights.sum()
    return frame, data, weights


def weighted_channel_statistics(spatial: torch.Tensor, weights: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """按患者/类别图像权重及图内49位置等权计算逐通道均值和标准差。"""
    weight = torch.as_tensor(weights, dtype=torch.float64, device=spatial.device)
    values = spatial.double()
    mean = (values.mean(1) * weight[:, None]).sum(0)
    variance = ((values - mean).square().mean(1) * weight[:, None]).sum(0)
    std = variance.sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool(std.le(0).any()):
        raise RuntimeError("逐通道标准化统计非有限或存在零标准差")
    return mean.float(), std.float()


def dictionary_statistics(model: ArchetypalMatryoshkaSAE, std: torch.Tensor) -> dict:
    """记录标准化及逆变换原坐标中的字典、混合和松弛统计。"""
    dictionary_std = model.decoder_weight.detach()
    dictionary_original = dictionary_std * std[None, :]
    weights = model.mixture_logits.detach().softmax(-1)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(1)
    relax_std = model.relaxation.detach().norm(dim=1)
    relax_original = (model.relaxation.detach() * std[None, :]).norm(dim=1)

    def dist(values: torch.Tensor) -> dict:
        values = values.float()
        return {
            "min": float(values.min()), "median": float(values.median()),
            "p95": float(torch.quantile(values, 0.95)), "max": float(values.max()),
        }

    return {
        "dictionary_norm_standardized": dist(dictionary_std.norm(dim=1)),
        "dictionary_norm_original": dist(dictionary_original.norm(dim=1)),
        "relaxation_norm_standardized": dist(relax_std),
        "relaxation_norm_original": dist(relax_original),
        "relaxation_at_delta_fraction": float(relax_std.ge(model.delta - 1e-6).float().mean()),
        "mixture_max_weight": dist(weights.max(1).values),
        "mixture_effective_count": dist(entropy.exp()),
    }


def standardized_losses(
    model: ArchetypalMatryoshkaSAE, standardized: torch.Tensor,
    original_pool: torch.Tensor, original_attention: torch.Tensor,
    head: torch.nn.Module, classifier_margin: torch.Tensor, margin_std: float,
    gamma_pool: float, mean: torch.Tensor, std: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    """patch在标准化坐标计算，pool/margin在逆变换后的原坐标计算。"""
    position_weights = patch_position_weights(original_attention)
    totals, terms = {}, {}
    for k, (reconstructed_std, _hidden) in model(standardized).items():
        patch = ((reconstructed_std - standardized).square().mean(2) * position_weights).mean()
        reconstructed_original = reconstructed_std * std + mean
        rebuilt_attention = attention_from_features(reconstructed_original, head)
        rebuilt_pool = pooled_from_features(reconstructed_original, rebuilt_attention)
        pool = F.mse_loss(rebuilt_pool, original_pool)
        margin = normalized_margin_mse(
            original_pool, rebuilt_pool, classifier_margin, margin_std
        )
        terms[k] = {"patch": patch, "pool": pool, "margin": margin}
        totals[k] = patch + gamma_pool * pool + GAMMA_MARGIN * margin
    return totals, terms


def train(args: argparse.Namespace) -> None:
    """执行固定预算的单次标准化候选训练。"""
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    old_config = json.loads((REFERENCE / "config.json").read_text())
    frame, data, image_weights = fixed_train_data(device)
    mean, std = weighted_channel_statistics(data["spatial"], image_weights)
    baseline_checkpoint = torch.load(
        REFERENCE / "ra_final.pth", map_location=device, weights_only=False
    )
    baseline_center = baseline_checkpoint["state_dict"]["decoder_bias"]
    mean_difference = float((mean - baseline_center).abs().max())
    if mean_difference > 2e-5:
        raise RuntimeError("标准化均值未复现当前患者/类别平衡中心")
    standardized = (data["spatial"] - mean) / std
    km = MiniBatchKMeans(
        n_clusters=WIDTH, random_state=SEED, n_init=1,
        batch_size=1024, max_iter=20, reassignment_ratio=0,
    )
    km.fit(
        standardized.cpu().numpy().reshape(-1, 1280),
        sample_weight=np.repeat(image_weights / 49, 49),
    )
    points = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
    model = ArchetypalMatryoshkaSAE(
        points, torch.zeros(1280, device=device), WIDTH, K_LIST, 0.2,
        True, SEED, "unique",
    ).to(device)
    initialization = calibrate_initial_encoder(
        model, standardized, data["attention"],
        torch.as_tensor(image_weights, device=device), BATCH,
    )
    initial_dictionary_std = model.decoder_weight.detach().clone()
    initial_stats = dictionary_statistics(model, std)
    encoder_lr_scale = initialization["scale"]
    config = {
        "scope": "single_standardized_coordinate_candidate",
        "single_changed_factor": "train-only patient-class-weighted per-channel standardization",
        "seed": SEED, "train_images": len(frame),
        "train_patients": int(frame.patient_id.nunique()),
        "epochs": EPOCHS, "draws_per_epoch": len(frame), "batch": BATCH,
        "updates_per_epoch": 13, "last_batch_size": 4,
        "hidden_dim": WIDTH, "k_list": list(K_LIST), "delta_standardized": 0.2,
        "base_learning_rate": BASE_LR,
        "encoder_lr_scale": encoder_lr_scale,
        "encoder_learning_rate_after_warmup": BASE_LR * encoder_lr_scale,
        "dictionary_learning_rate_after_warmup": BASE_LR,
        "warmup_epochs": WARMUP_EPOCHS, "weight_decay": 0.0,
        "gamma_pool": old_config["gamma_pool"], "gamma_margin": GAMMA_MARGIN,
        "margin_std_original_coordinates": old_config["margin_std"],
        "gamma_recalibrated": False,
        "patch_loss_coordinates": "standardized",
        "pool_margin_coordinates": "inverse-transformed original C-long coordinates",
        "representative_fit": "MiniBatchKMeans on standardized fixed pilot196 tokens; random_state1701",
        "normalization_fit": "fixed pilot196 only; patient/class image weights; 49 positions equal within image",
        "checkpoint_rule": "fixed epoch100; no val selection",
        "test_read": False, "external_read": False,
    }
    json_write(args.output / "config.json", config)
    normalization = {
        "mean_max_abs_difference_vs_current_center": mean_difference,
        "std_min": float(std.min()), "std_median": float(std.median()),
        "std_p95": float(torch.quantile(std, 0.95)), "std_max": float(std.max()),
        "std_max_over_min": float(std.max() / std.min()),
        "mean": mean.cpu().tolist(), "std": std.cpu().tolist(),
    }
    json_write(args.output / "normalization.json", normalization)
    json_write(args.output / "initialization.json", {
        "encoder_calibration": initialization,
        "encoder_lr_scale": encoder_lr_scale,
        "actual_encoder_lr_after_warmup": BASE_LR * encoder_lr_scale,
        "dictionary_statistics": initial_stats,
        "kmeans_steps": int(km.n_steps_),
    })
    rng = np.random.default_rng(SEED)
    orders = [rng.choice(len(frame), len(frame), replace=True, p=image_weights)
              for _ in range(EPOCHS)]
    _baseline, head, classifier_weight, _classifier_bias = load_models(device)
    classifier_margin = classifier_weight[1] - classifier_weight[0]
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr_scale": encoder_lr_scale},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")], "lr_scale": 1.0},
    ], lr=BASE_LR, weight_decay=0)
    history = []
    for epoch, order in enumerate(orders, 1):
        model.train()
        warmup = min(1.0, epoch / WARMUP_EPOCHS)
        for group in optimizer.param_groups:
            group["lr"] = BASE_LR * warmup * group["lr_scale"]
        sums = {"total": 0.0, "patch": 0.0, "pool": 0.0, "margin": 0.0}
        for start in range(0, len(order), BATCH):
            indices = torch.as_tensor(order[start:start + BATCH], device=device)
            optimizer.zero_grad(set_to_none=True)
            totals, terms = standardized_losses(
                model, standardized[indices], data["pooled"][indices],
                data["attention"][indices], head, classifier_margin,
                old_config["margin_std"], old_config["gamma_pool"], mean, std,
            )
            loss = torch.stack(list(totals.values())).mean()
            patch = torch.stack([value["patch"] for value in terms.values()]).mean()
            pool = torch.stack([value["pool"] for value in terms.values()]).mean()
            margin = torch.stack([value["margin"] for value in terms.values()]).mean()
            loss = patch + warmup * (loss - patch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch{epoch}出现非有限损失")
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            size = len(indices)
            for key, value in (("total", loss), ("patch", patch), ("pool", pool), ("margin", margin)):
                sums[key] += float(value.detach()) * size
        row = {"epoch": epoch, **{key: value / len(order) for key, value in sums.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        print(json.dumps(row), flush=True)
    final_stats = dictionary_statistics(model, std)
    change = {
        "dictionary_rms_change_standardized": float(
            (model.decoder_weight.detach() - initial_dictionary_std).square().mean().sqrt()
        ),
        "dictionary_rms_change_original": float(
            ((model.decoder_weight.detach() - initial_dictionary_std) * std[None, :]).square().mean().sqrt()
        ),
        "initial": initial_stats, "final": final_stats,
    }
    json_write(args.output / "training_change.json", change)
    torch.save({
        "state_dict": model.state_dict(), "config": config,
        "normalization_mean": mean.cpu(), "normalization_std": std.cpu(),
        "initialization": initialization,
    }, args.output / "standardized_ra_final.pth")
    json_write(args.output / "verification.json", {
        "fixed_training_subset": True, "fixed_sampling_orders": True,
        "gamma_pool_reused": old_config["gamma_pool"],
        "gamma_margin_reused": GAMMA_MARGIN,
        "patch_loss_standardized": True,
        "pool_margin_after_inverse_transform": True,
        "encoder_lr_rule_reused": True,
        "actual_encoder_lr_after_warmup": BASE_LR * encoder_lr_scale,
        "checkpoint_epoch": EPOCHS, "test_read": False, "external_read": False,
    })


def load_candidate(path: Path, device: torch.device):
    """读取标准化候选及其逆变换参数。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    state, config = payload["state_dict"], payload["config"]
    model = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], WIDTH, K_LIST,
        config["delta_standardized"], True, SEED, "unique",
    ).to(device)
    model.load_state_dict(state)
    model.requires_grad_(False).eval()
    return model, payload["normalization_mean"].to(device), payload["normalization_std"].to(device)


def distribution(values: np.ndarray) -> dict:
    """一维有限数组的描述统计。"""
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()), "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)), "mean": float(values.mean()),
        "median": float(np.median(values)), "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
    }


def train_reference_means() -> tuple[np.ndarray, np.ndarray]:
    """完整train算术均值，仅用于原坐标FVU参照。"""
    spatial = np.load(CACHE / "train_spatial_features.npy", mmap_mode="r")
    pooled = np.load(CACHE / "train_pooled_features.npy", mmap_mode="r")
    return (
        np.asarray(spatial).sum((0, 1), dtype=np.float64) / (spatial.shape[0] * 49),
        np.asarray(pooled).sum(0, dtype=np.float64) / pooled.shape[0],
    )


@torch.no_grad()
def fidelity_and_usage(
    name: str, split: str, frame: pd.DataFrame, model: ArchetypalMatryoshkaSAE,
    head: torch.nn.Module, classifier_weight: torch.Tensor, classifier_bias: torch.Tensor,
    patch_mean: np.ndarray, pooled_mean: np.ndarray, device: torch.device,
    mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
) -> tuple[dict, pd.DataFrame]:
    """在原坐标计算纯重构保真、预测变化和实际稀疏度。"""
    rows = []
    patch_rss = patch_tss = pooled_rss = pooled_tss = 0.0
    patch_cos = pooled_cos = attention_cos = 0.0
    active_counts = np.zeros(WIDTH, dtype=np.int64)
    l0_values = []
    for start in range(0, len(frame), BATCH):
        stop = min(start + BATCH, len(frame))
        current = frame.iloc[start:stop]
        data = read_subset(current, split, device)
        spatial = data["spatial"]
        original_attention = attention_from_features(spatial, head)
        original_pool = pooled_from_features(spatial, original_attention)
        encoded_input = spatial if mean is None else (spatial - mean) / std
        hidden = model.encode(encoded_input, K)
        reconstructed_space = model.decode(hidden)
        reconstructed = reconstructed_space if mean is None else reconstructed_space * std + mean
        rebuilt_attention = attention_from_features(reconstructed, head)
        rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
        original_logits = original_pool @ classifier_weight.T + classifier_bias
        rebuilt_logits = rebuilt_pool @ classifier_weight.T + classifier_bias
        original_probability = original_logits.softmax(1)[:, 1]
        rebuilt_probability = rebuilt_logits.softmax(1)[:, 1]
        original_np = spatial.cpu().numpy().astype(np.float64)
        rebuilt_np = reconstructed.cpu().numpy().astype(np.float64)
        original_pool_np = original_pool.cpu().numpy().astype(np.float64)
        rebuilt_pool_np = rebuilt_pool.cpu().numpy().astype(np.float64)
        patch_rss += float(np.square(rebuilt_np - original_np).sum())
        patch_tss += float(np.square(original_np - patch_mean).sum())
        pooled_rss += float(np.square(rebuilt_pool_np - original_pool_np).sum())
        pooled_tss += float(np.square(original_pool_np - pooled_mean).sum())
        patch_cos += float(F.cosine_similarity(spatial, reconstructed, dim=2).sum())
        pooled_cos += float(F.cosine_similarity(original_pool, rebuilt_pool, dim=1).sum())
        attention_cos += float(F.cosine_similarity(original_attention, rebuilt_attention, dim=1).sum())
        active = hidden.gt(1e-8)
        active_counts += active.sum((0, 1)).cpu().numpy()
        l0_values.extend(active.sum(2).cpu().numpy().reshape(-1).tolist())
        for local, record in enumerate(current.itertuples()):
            rows.append({
                "patient_id": str(record.patient_id), "label": int(record.label),
                "original_probability": float(original_probability[local]),
                "reconstructed_probability": float(rebuilt_probability[local]),
            })
    images = pd.DataFrame(rows)
    patients = images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), original_probability=("original_probability", "mean"),
        reconstructed_probability=("reconstructed_probability", "mean"),
    )
    images["abs_delta"] = (images.reconstructed_probability - images.original_probability).abs()
    patients["abs_delta"] = (patients.reconstructed_probability - patients.original_probability).abs()
    image_agreement = (
        images.original_probability.ge(FROZEN_IMAGE_THRESHOLD)
        == images.reconstructed_probability.ge(FROZEN_IMAGE_THRESHOLD)
    )
    patient_agreement = (
        patients.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
        == patients.reconstructed_probability.ge(FROZEN_PATIENT_THRESHOLD)
    )
    l0 = np.asarray(l0_values)
    summary = {
        "model": name, "split": split, "images": len(images), "patients": len(patients),
        "prediction": {
            "image_probability_abs_change": distribution(images.abs_delta),
            "patient_probability_abs_change": distribution(patients.abs_delta),
            "image_prediction_agreement": float(image_agreement.mean()),
            "patient_prediction_agreement": float(patient_agreement.mean()),
            "image_flip_count": int((~image_agreement).sum()),
            "patient_flip_count": int((~patient_agreement).sum()),
            "original_patient_auc": float(roc_auc_score(patients.label, patients.original_probability)),
            "reconstructed_patient_auc": float(roc_auc_score(patients.label, patients.reconstructed_probability)),
        },
        "reconstruction": {
            "patch_fvu": patch_rss / patch_tss, "pooled_fvu": pooled_rss / pooled_tss,
            "mean_patch_cosine": patch_cos / (len(frame) * 49),
            "mean_pooled_cosine": pooled_cos / len(frame),
            "mean_attention_cosine": attention_cos / len(frame),
        },
        "sparsity": {
            "l0": distribution(l0), "dead_feature_count": int((active_counts == 0).sum()),
            "active_feature_count": int((active_counts > 0).sum()),
        },
    }
    return summary, patients


def safe_spearman(left: np.ndarray, right: np.ndarray, minimum: int = 3) -> tuple[float, int, str]:
    """恒定或共同正响应不足时返回NaN及明确原因。"""
    left, right = np.asarray(left), np.asarray(right)
    mask = np.isfinite(left) & np.isfinite(right) & ((left > 0) | (right > 0))
    count = int(mask.sum())
    if count < minimum:
        return np.nan, count, "insufficient_union_positive"
    if np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return np.nan, count, "constant_response"
    value = float(spearmanr(left[mask], right[mask]).statistic)
    return value, count, "evaluable" if np.isfinite(value) else "nonfinite"


@torch.no_grad()
def response_matching(
    baseline: ArchetypalMatryoshkaSAE, candidate: ArchetypalMatryoshkaSAE,
    mean: torch.Tensor, std: torch.Tensor, device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    """在原坐标匹配方向，并在完整val比较患者、空间和删除响应。"""
    baseline_direction = F.normalize(baseline.decoder_weight.detach(), dim=1)
    candidate_original_direction = candidate.decoder_weight.detach() * std[None, :]
    candidate_direction = F.normalize(candidate_original_direction, dim=1)
    similarity = (baseline_direction @ candidate_direction.T).cpu().numpy()
    baseline_ids, candidate_ids = linear_sum_assignment(-similarity)
    order = np.argsort(baseline_ids)
    candidate_for_baseline = candidate_ids[order]
    matched_cosine = similarity[np.arange(WIDTH), candidate_for_baseline]

    frame = pd.read_csv(CACHE / "val_metadata.csv").reset_index().rename(columns={"index": "source_row"})
    patient_ids = np.sort(frame.patient_id.astype(str).unique())
    patient_lookup = {patient: index for index, patient in enumerate(patient_ids)}
    base_peak_sum = np.zeros((len(patient_ids), WIDTH), dtype=np.float64)
    cand_peak_sum = np.zeros_like(base_peak_sum)
    image_count = np.zeros(len(patient_ids), dtype=np.int32)
    spatial_sum = np.zeros_like(base_peak_sum)
    spatial_count = np.zeros((len(patient_ids), WIDTH), dtype=np.int32)
    match_tensor = torch.as_tensor(candidate_for_baseline, device=device)
    for start in range(0, len(frame), 4):
        stop = min(start + 4, len(frame))
        current = frame.iloc[start:stop]
        spatial = read_subset(current, "val", device)["spatial"]
        base_hidden = baseline.encode(spatial, K)
        cand_hidden = candidate.encode((spatial - mean) / std, K).index_select(2, match_tensor)
        base_peak = base_hidden.max(1).values.cpu().numpy()
        cand_peak = cand_hidden.max(1).values.cpu().numpy()
        dot = (base_hidden * cand_hidden).sum(1)
        norm = base_hidden.square().sum(1).sqrt() * cand_hidden.square().sum(1).sqrt()
        valid = norm.gt(0)
        cosine = torch.where(valid, dot / norm.clamp_min(1e-12), torch.zeros_like(dot)).cpu().numpy()
        valid_np = valid.cpu().numpy()
        for local, record in enumerate(current.itertuples()):
            patient = patient_lookup[str(record.patient_id)]
            base_peak_sum[patient] += base_peak[local]
            cand_peak_sum[patient] += cand_peak[local]
            image_count[patient] += 1
            spatial_sum[patient] += cosine[local]
            spatial_count[patient] += valid_np[local]
    base_patient = base_peak_sum / image_count[:, None]
    cand_patient = cand_peak_sum / image_count[:, None]
    spatial_patient = np.divide(
        spatial_sum, spatial_count, out=np.full_like(spatial_sum, np.nan),
        where=spatial_count > 0,
    )

    val98 = pd.read_csv(REFERENCE / "val_subset.csv")
    baseline_delta = np.load(DECISION / "ra_val_delta_margin.npy", mmap_mode="r")
    if baseline_delta.shape != (len(val98), WIDTH):
        raise RuntimeError("既有val98删除矩阵shape不符")
    candidate_delta = np.zeros_like(baseline_delta)
    _current, head, classifier_weight, classifier_bias = load_models(device)
    attention_weight = head.weight.flatten()
    attention_bias = head.bias.flatten()[0]
    margin_weight = classifier_weight[1] - classifier_weight[0]
    margin_bias = classifier_bias[1] - classifier_bias[0]
    decoder_original = candidate.decoder_weight.detach() * std[None, :]
    for start in range(0, len(val98), 2):
        stop = min(start + 2, len(val98))
        spatial = read_subset(val98.iloc[start:stop], "val", device)["spatial"]
        hidden = candidate.encode((spatial - mean) / std, K)
        effects = intervention_block(
            spatial, hidden, decoder_original, attention_weight, attention_bias,
            margin_weight, margin_bias, 0.0,
        )
        candidate_delta[start:stop] = effects["delta_margin"].cpu().numpy()
    candidate_delta = candidate_delta[:, candidate_for_baseline]
    effect_patients = np.sort(val98.patient_id.astype(str).unique())
    effect_lookup = {patient: index for index, patient in enumerate(effect_patients)}
    base_effect = np.zeros((len(effect_patients), WIDTH), dtype=np.float64)
    cand_effect = np.zeros_like(base_effect)
    effect_count = np.zeros(len(effect_patients), dtype=np.int32)
    for row, record in enumerate(val98.itertuples()):
        patient = effect_lookup[str(record.patient_id)]
        base_effect[patient] += baseline_delta[row]
        cand_effect[patient] += candidate_delta[row]
        effect_count[patient] += 1
    base_effect /= effect_count[:, None]
    cand_effect /= effect_count[:, None]

    rows = []
    for feature in range(WIDTH):
        response_corr, response_n, response_status = safe_spearman(
            base_patient[:, feature], cand_patient[:, feature]
        )
        effect_corr, effect_n, effect_status = safe_spearman(
            np.abs(base_effect[:, feature]), np.abs(cand_effect[:, feature])
        )
        base_order = np.argsort(-base_patient[:, feature], kind="stable")[:5]
        cand_order = np.argsort(-cand_patient[:, feature], kind="stable")[:5]
        spatial_values = spatial_patient[:, feature]
        spatial_n = int(np.isfinite(spatial_values).sum())
        spatial_mean = float(np.nanmean(spatial_values)) if spatial_n else np.nan
        rows.append({
            "baseline_feature_id": feature,
            "candidate_feature_id": int(candidate_for_baseline[feature]),
            "decoder_cosine_original_coordinates": float(matched_cosine[feature]),
            "val_patient_response_spearman": response_corr,
            "val_patient_response_evaluable_count": response_n,
            "val_patient_response_status": response_status,
            "val_top5_patient_overlap": len(set(base_order) & set(cand_order)) / 5,
            "val_same_image_spatial_cosine_patient_mean": spatial_mean,
            "val_spatial_evaluable_patients": spatial_n,
            "val98_abs_deletion_effect_spearman": effect_corr,
            "val98_deletion_effect_evaluable_patients": effect_n,
            "val98_deletion_effect_status": effect_status,
            "val98_signed_deletion_effect_same_direction_fraction": float(
                np.mean(np.sign(base_effect[:, feature]) == np.sign(cand_effect[:, feature]))
            ),
        })
    pairs = pd.DataFrame(rows)

    def finite_distribution(column: str) -> dict:
        values = pairs[column].dropna().to_numpy()
        return {**distribution(values), "evaluable_features": len(values),
                "not_evaluable_features": WIDTH - len(values)}

    summary = {
        "matching": "forced Hungarian one-to-one on original-coordinate decoder cosine; low similarities retained",
        "decoder_cosine": distribution(pairs.decoder_cosine_original_coordinates),
        "val_patient_response_spearman": finite_distribution("val_patient_response_spearman"),
        "val_top5_patient_overlap": distribution(pairs.val_top5_patient_overlap),
        "val_same_image_spatial_cosine": finite_distribution(
            "val_same_image_spatial_cosine_patient_mean"
        ),
        "val98_abs_deletion_effect_spearman": finite_distribution(
            "val98_abs_deletion_effect_spearman"
        ),
        "response_status_counts": pairs.val_patient_response_status.value_counts().to_dict(),
        "deletion_status_counts": pairs.val98_deletion_effect_status.value_counts().to_dict(),
        "limits": [
            "matching does not establish equivalent semantics",
            "this is extraction change, not cross-seed stability",
            "medical feedback not used",
        ],
    }
    return pairs, summary


def analyze_command(args: argparse.Namespace) -> None:
    """执行完整train/val原坐标保真与强制配对分析。"""
    args.analysis_output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    baseline, head, classifier_weight, classifier_bias = load_models(device)
    candidate, mean, std = load_candidate(args.checkpoint, device)
    patch_mean, pooled_mean = train_reference_means()
    summaries = {}
    for split in ("train", "val"):
        frame = pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(
            columns={"index": "source_row"}
        )
        summaries[split] = {}
        for name, model, current_mean, current_std in (
            ("baseline", baseline, None, None),
            ("standardized_candidate", candidate, mean, std),
        ):
            summary, patients = fidelity_and_usage(
                name, split, frame, model, head, classifier_weight, classifier_bias,
                patch_mean, pooled_mean, device, current_mean, current_std,
            )
            summaries[split][name] = summary
            if split == "val":
                patients.to_csv(
                    args.analysis_output / f"val_{name}_patient_fidelity.csv", index=False
                )
            print(f"{split}/{name} complete", flush=True)
    json_write(args.analysis_output / "original_coordinate_fidelity.json", summaries)
    comparison = {}
    for split in ("train", "val"):
        old, new = summaries[split]["baseline"], summaries[split]["standardized_candidate"]
        comparison[split] = {
            "patch_fvu_delta": new["reconstruction"]["patch_fvu"] - old["reconstruction"]["patch_fvu"],
            "pooled_fvu_delta": new["reconstruction"]["pooled_fvu"] - old["reconstruction"]["pooled_fvu"],
            "patient_probability_mae_delta": (
                new["prediction"]["patient_probability_abs_change"]["mean"]
                - old["prediction"]["patient_probability_abs_change"]["mean"]
            ),
            "patient_agreement_delta": (
                new["prediction"]["patient_prediction_agreement"]
                - old["prediction"]["patient_prediction_agreement"]
            ),
            "patient_auc_delta": (
                new["prediction"]["reconstructed_patient_auc"]
                - old["prediction"]["reconstructed_patient_auc"]
            ),
        }
    val = comparison["val"]
    conditions = {
        "patch_fvu_lower": val["patch_fvu_delta"] < 0,
        "pooled_fvu_lower": val["pooled_fvu_delta"] < 0,
        "patient_probability_mae_lower": val["patient_probability_mae_delta"] < 0,
        "patient_agreement_not_lower": val["patient_agreement_delta"] >= 0,
    }
    comparison["fidelity_conditions"] = conditions
    comparison["fidelity_status"] = (
        "coherent_original_coordinate_fidelity_improvement"
        if all(conditions.values()) else "original_coordinate_fidelity_improvement_inconsistent"
    )
    json_write(args.analysis_output / "fidelity_comparison.json", comparison)
    pairs, match_summary = response_matching(
        baseline, candidate, mean, std, device
    )
    pairs.to_csv(args.analysis_output / "feature_matching.csv", index=False)
    json_write(args.analysis_output / "matching_summary.json", match_summary)
    json_write(args.analysis_output / "analysis_definition.json", {
        "fidelity_coordinates": "original C-long coordinates after inverse transform",
        "candidate_decoder_original": "normalization_std * decoder_standardized",
        "matching": "train-parameter decoder Hungarian, forced one-to-one; not semantic equivalence",
        "response_na": "constant or insufficient union-positive responses are NA and counted",
        "deletion": "candidate direction inverse-transformed; original attention/pooling/classifier recomputed",
        "medical_feedback_used": False,
        "test_read": False, "external_read": False,
    })


def main() -> None:
    """选择训练或分析阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--checkpoint", type=Path, required=True)
    analyze_parser.add_argument("--analysis-output", type=Path, required=True)
    analyze_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        analyze_command(args)


if __name__ == "__main__":
    main()
