#!/usr/bin/env python3
"""在共同train图片上拆解RA-SAE扩大抽样池前后的误差路径。

本诊断只比较两个已完成checkpoint，不训练模型。共同队列为扩大池的
1673张train图，其中完整包含原pilot 196图/128人，其余1477图/1084人
标记为新增患者。同时报告原训练损失和固定原注意力/重算注意力路径，
用于定位误差变化；不检验训练次数或代表点是否为成因。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE  # noqa: E402
from clong_s2b_core import (  # noqa: E402
    attention_from_features,
    patch_position_weights,
    pooled_from_features,
)
from clong_sae_discovery import patient_class_weights  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
BASELINE_ROOT = BASE / "duration100"
CANDIDATE_ROOT = BASE / "expanded_pool_fidelity_20260916"
K_LIST = (64, 128, 256)
GAMMA_MARGIN = 0.1


def load_candidate(device: torch.device) -> ArchetypalMatryoshkaSAE:
    """读取扩大抽样池后的固定第100轮RA-SAE。

    Args:
        device (torch.device): 已核对的CUDA或CPU设备。

    Returns:
        ArchetypalMatryoshkaSAE: 冻结的2560宽字典。
    """
    checkpoint = torch.load(
        CANDIDATE_ROOT / "coverage_ra_final.pth", map_location=device, weights_only=False
    )
    state, config = checkpoint["state_dict"], checkpoint["config"]
    if config["hidden_dim"] != 2560 or tuple(config["k_list"]) != K_LIST:
        raise RuntimeError("扩大池checkpoint结构不符合预期")
    model = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], config["hidden_dim"],
        tuple(config["k_list"]), config["delta"], True, config["seed"],
        config["initialization"],
    ).to(device)
    model.load_state_dict(state)
    model.requires_grad_(False).eval()
    return model


def common_cohort() -> tuple[pd.DataFrame, set[str]]:
    """构建共同1673图train队列并核验原pilot完整包含关系。

    Args:
        None.

    Returns:
        tuple[pd.DataFrame, set[str]]: 固定队列和128位pilot患者ID。
    """
    pool = pd.read_csv(CANDIDATE_ROOT / "expanded_train_pool.csv")
    pilot = pd.read_csv(BASELINE_ROOT / "train_subset.csv")
    pilot_patients = set(pilot.patient_id.astype(str))
    pool_keys = set(zip(pool.patient_id.astype(str), pool.relative_path.astype(str)))
    pilot_keys = set(zip(pilot.patient_id.astype(str), pilot.relative_path.astype(str)))
    if len(pool) != 1673 or pool.patient_id.nunique() != 1212:
        raise RuntimeError("共同队列不是1673图/1212人")
    if len(pilot) != 196 or len(pilot_patients) != 128 or not pilot_keys <= pool_keys:
        raise RuntimeError("共同队列没有完整包含原pilot 196图/128人")
    pool = pool.copy()
    pool["patient_id"] = pool.patient_id.astype(str)
    pool["cohort_role"] = np.where(
        pool.patient_id.isin(pilot_patients), "original_pilot_patient", "new_patient"
    )
    counts = pool.groupby("cohort_role").agg(
        images=("source_row", "size"), patients=("patient_id", "nunique")
    )
    if tuple(counts.loc["original_pilot_patient"]) != (196, 128):
        raise RuntimeError("原pilot患者分层不是196图/128人")
    if tuple(counts.loc["new_patient"]) != (1477, 1084):
        raise RuntimeError("新增患者分层不是1477图/1084人")
    return pool, pilot_patients


def evaluation_weights(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """为总体、原pilot患者和新增患者生成患者/类别平衡权重。

    Args:
        frame (pd.DataFrame): 1673行共同train队列，含``cohort_role``。

    Returns:
        dict[str, np.ndarray]: 三个长度1673且各自和为1的固定权重。
    """
    scopes = {
        "all": np.ones(len(frame), dtype=bool),
        "original_pilot_patient": frame.cohort_role.eq("original_pilot_patient").to_numpy(),
        "new_patient": frame.cohort_role.eq("new_patient").to_numpy(),
    }
    result = {}
    for name, mask in scopes.items():
        local = frame.loc[mask].reset_index(drop=True)
        weights = patient_class_weights(local).astype(np.float64)
        weights /= weights.sum()
        full = np.zeros(len(frame), dtype=np.float64)
        full[np.flatnonzero(mask)] = weights
        if not np.isclose(full.sum(), 1.0):
            raise RuntimeError(f"{name}权重和不是1")
        result[name] = full
    return result


@torch.no_grad()
def evaluate_model(
    name: str, sae: ArchetypalMatryoshkaSAE, data: dict[str, torch.Tensor],
    frame: pd.DataFrame, head: torch.nn.Module, margin_vector: torch.Tensor,
    margin_std: float, gamma_pool: float, batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    """对一个checkpoint计算三层K的原损失与两条注意力路径。

    Args:
        name (str): ``baseline``或``expanded_pool``。
        sae (ArchetypalMatryoshkaSAE): 待评价的冻结SAE。
        data (dict[str, torch.Tensor]): 共同队列的spatial/pooled/attention缓存。
        frame (pd.DataFrame): 与缓存同序的1673行元数据。
        head (torch.nn.Module): 冻结C-long注意力头。
        margin_vector (torch.Tensor): 癌减非癌分类方向``[1280]``。
        margin_std (float): 原训练使用的margin归一化尺度。
        gamma_pool (float): 原训练pool项权重。
        batch_size (int): 图像批量。

    Returns:
        tuple[pd.DataFrame, dict]: 逐图逐K指标及原缓存复算误差。
    """
    rows = []
    max_attention_error = max_pool_error = 0.0
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        spatial = data["spatial"][start:stop]
        original_attention = attention_from_features(spatial, head)
        original_pool = pooled_from_features(spatial, original_attention)
        max_attention_error = max(
            max_attention_error,
            float((original_attention - data["attention"][start:stop]).abs().max()),
        )
        max_pool_error = max(
            max_pool_error, float((original_pool - data["pooled"][start:stop]).abs().max())
        )
        position_weights = patch_position_weights(original_attention)
        layers = sae(spatial)
        for k, (reconstructed, _hidden) in layers.items():
            rebuilt_attention = attention_from_features(reconstructed, head)
            fixed_pool = pooled_from_features(reconstructed, original_attention)
            recomputed_pool = pooled_from_features(reconstructed, rebuilt_attention)
            ordinary_patch = (reconstructed - spatial).square().mean((1, 2))
            training_patch = (
                (reconstructed - spatial).square().mean(2) * position_weights
            ).mean(1)
            training_pool = (recomputed_pool - original_pool).square().mean(1)
            recomputed_margin_error = (recomputed_pool - original_pool) @ margin_vector
            fixed_margin_error = (fixed_pool - original_pool) @ margin_vector
            training_margin = (recomputed_margin_error / margin_std).square()
            training_total = (
                training_patch + gamma_pool * training_pool + GAMMA_MARGIN * training_margin
            )
            fixed_pool_mse = (fixed_pool - original_pool).square().mean(1)
            fixed_margin_loss = (fixed_margin_error / margin_std).square()
            attention_cosine = torch_functional.cosine_similarity(
                original_attention, rebuilt_attention, dim=1
            )
            fixed_cosine = torch_functional.cosine_similarity(
                original_pool, fixed_pool, dim=1
            )
            recomputed_cosine = torch_functional.cosine_similarity(
                original_pool, recomputed_pool, dim=1
            )
            arrays = {
                "ordinary_patch_mse": ordinary_patch,
                "training_patch_loss": training_patch,
                "training_pool_loss": training_pool,
                "training_margin_loss": training_margin,
                "training_patch_contribution": training_patch,
                "training_pool_contribution": gamma_pool * training_pool,
                "training_margin_contribution": GAMMA_MARGIN * training_margin,
                "training_total": training_total,
                "fixed_attention_pool_mse": fixed_pool_mse,
                "recomputed_attention_pool_mse": training_pool,
                "fixed_attention_pooled_cosine": fixed_cosine,
                "recomputed_attention_pooled_cosine": recomputed_cosine,
                "fixed_attention_margin_error": fixed_margin_error,
                "recomputed_attention_margin_error": recomputed_margin_error,
                "fixed_attention_margin_abs_error": fixed_margin_error.abs(),
                "recomputed_attention_margin_abs_error": recomputed_margin_error.abs(),
                "fixed_attention_margin_loss": fixed_margin_loss,
                "recomputed_attention_margin_loss": training_margin,
                "attention_cosine": attention_cosine,
                "attention_l1": (rebuilt_attention - original_attention).abs().sum(1),
            }
            numpy_arrays = {key: value.detach().cpu().numpy() for key, value in arrays.items()}
            for local, record in enumerate(frame.iloc[start:stop].itertuples(index=False)):
                row = {
                    "checkpoint": name, "k": int(k), "source_row": int(record.source_row),
                    "patient_id": str(record.patient_id), "label": int(record.label),
                    "cohort_role": record.cohort_role,
                }
                row.update({key: float(value[local]) for key, value in numpy_arrays.items()})
                rows.append(row)
        print(f"{name}: {stop}/{len(frame)}", flush=True)
    verification = {
        "max_attention_cache_error": max_attention_error,
        "max_pool_cache_error": max_pool_error,
    }
    if max(max_attention_error, max_pool_error) > 1e-4:
        raise RuntimeError(f"{name}原始路径与冻结缓存不一致")
    return pd.DataFrame(rows), verification


def aggregate_metrics(
    image_metrics: pd.DataFrame, weights: dict[str, np.ndarray], frame: pd.DataFrame,
) -> pd.DataFrame:
    """对三层联合目标和K=256按固定权重汇总。

    Args:
        image_metrics (pd.DataFrame): 两checkpoint的逐图逐K指标。
        weights (dict[str, np.ndarray]): 三个队列层级的患者/类别平衡权重。
        frame (pd.DataFrame): 共同1673行队列。

    Returns:
        pd.DataFrame: checkpoint x scope x ``joint_K64_128_256/K256``汇总。
    """
    metric_columns = [
        column for column in image_metrics.columns
        if column not in {"checkpoint", "k", "source_row", "patient_id", "label", "cohort_role"}
    ]
    weight_lookup = {
        scope: dict(zip(frame.source_row.astype(int), values)) for scope, values in weights.items()
    }
    rows = []
    for checkpoint in ("baseline", "expanded_pool"):
        current = image_metrics[image_metrics.checkpoint.eq(checkpoint)]
        views = {
            "joint_K64_128_256": current.groupby("source_row", as_index=False)[metric_columns].mean(),
            "K256": current[current.k.eq(256)].copy(),
        }
        for view_name, view in views.items():
            for scope, lookup in weight_lookup.items():
                local_weights = view.source_row.map(lookup).to_numpy(float)
                if not np.isclose(local_weights.sum(), 1.0):
                    raise RuntimeError(f"{checkpoint}/{view_name}/{scope}权重错位")
                row = {"checkpoint": checkpoint, "scope": scope, "view": view_name}
                for metric in metric_columns:
                    row[metric] = float(np.sum(view[metric].to_numpy(float) * local_weights))
                rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(aggregate: pd.DataFrame) -> pd.DataFrame:
    """计算扩大池checkpoint减原pilot checkpoint的配对指标差。

    Args:
        aggregate (pd.DataFrame): 两checkpoint的聚合指标。

    Returns:
        pd.DataFrame: 每个scope/view一行的candidate-minus-baseline差值。
    """
    keys = ["scope", "view"]
    metrics = [column for column in aggregate.columns if column not in {"checkpoint", *keys}]
    old = aggregate[aggregate.checkpoint.eq("baseline")].drop(columns="checkpoint")
    new = aggregate[aggregate.checkpoint.eq("expanded_pool")].drop(columns="checkpoint")
    merged = old.merge(new, on=keys, suffixes=("_baseline", "_expanded"), validate="one_to_one")
    output = merged[keys].copy()
    for metric in metrics:
        output[f"{metric}_baseline"] = merged[f"{metric}_baseline"]
        output[f"{metric}_expanded"] = merged[f"{metric}_expanded"]
        output[f"{metric}_delta"] = (
            merged[f"{metric}_expanded"] - merged[f"{metric}_baseline"]
        )
    return output


def main() -> None:
    """运行无训练的共同train损失与注意力路径诊断。

    Args:
        None: 输出、设备和批量由CLI指定。

    Returns:
        None: 写入逐图指标、聚合表、配对差值和验证JSON。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame, pilot_patients = common_cohort()
    weights = evaluation_weights(frame)
    device = torch.device(args.device)
    data = read_subset(frame, "train", device)
    baseline, head, classifier_weight, _classifier_bias = load_models(device)
    baseline_config = json.loads((BASELINE_ROOT / "config.json").read_text())
    candidate_config = json.loads((CANDIDATE_ROOT / "config.json").read_text())
    gamma_pool = float(baseline_config["gamma_pool"])
    margin_std = float(baseline_config["margin_std"])
    if not np.isclose(gamma_pool, candidate_config["gamma_pool"]):
        raise RuntimeError("两checkpoint的gamma_pool不一致")
    if not np.isclose(margin_std, candidate_config["margin_std"]):
        raise RuntimeError("两checkpoint的margin_std不一致")
    margin_vector = classifier_weight[1] - classifier_weight[0]
    baseline_rows, baseline_verification = evaluate_model(
        "baseline", baseline, data, frame, head, margin_vector,
        margin_std, gamma_pool, args.batch_size,
    )
    del baseline
    torch.cuda.empty_cache() if device.type == "cuda" else None
    candidate = load_candidate(device)
    candidate_rows, candidate_verification = evaluate_model(
        "expanded_pool", candidate, data, frame, head, margin_vector,
        margin_std, gamma_pool, args.batch_size,
    )
    image_metrics = pd.concat([baseline_rows, candidate_rows], ignore_index=True)
    if len(image_metrics) != 2 * len(frame) * len(K_LIST):
        raise RuntimeError("逐图逐K输出行数不符合预期")
    if not np.isfinite(image_metrics.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("诊断输出存在NaN或Inf")
    aggregate = aggregate_metrics(image_metrics, weights, frame)
    differences = paired_differences(aggregate)
    image_metrics.to_csv(args.output / "image_metrics_internal.csv", index=False)
    aggregate.to_csv(args.output / "aggregate_metrics.csv", index=False)
    differences.to_csv(args.output / "checkpoint_differences.csv", index=False)
    definition = {
        "question": "locate which computation path changes reconstruction errors",
        "causal_hypotheses_tested": False,
        "common_images": 1673,
        "common_patients": 1212,
        "original_pilot_patient_scope": {"images": 196, "patients": 128},
        "new_patient_scope": {"images": 1477, "patients": 1084},
        "weights": "within each scope, class-balanced and patient-equal; identical for both checkpoints",
        "joint_view": "arithmetic mean over K=64/128/256 per image, then fixed weighted mean",
        "k256_view": "K=256 per image, then fixed weighted mean",
        "training_patch_loss": "original attention-weighted patch MSE",
        "ordinary_patch_mse": "diagnostic only; not the original training patch loss",
        "fixed_attention_path": "pool the same reconstructed features with original attention",
        "recomputed_attention_path": "recompute frozen attention from reconstructed features before pooling",
        "path_difference_boundary": "descriptive path comparison, not strict causal decomposition",
        "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification = {
        "common_images": len(frame), "common_patients": int(frame.patient_id.nunique()),
        "pilot_images": int(frame.patient_id.isin(pilot_patients).sum()),
        "pilot_patients": len(pilot_patients),
        "new_patient_images": int((~frame.patient_id.isin(pilot_patients)).sum()),
        "new_patients": int(frame.loc[~frame.patient_id.isin(pilot_patients), "patient_id"].nunique()),
        "rows": len(image_metrics), "all_outputs_finite": True,
        "baseline": baseline_verification, "expanded_pool": candidate_verification,
        "identical_images_weights_and_order": True,
        "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
