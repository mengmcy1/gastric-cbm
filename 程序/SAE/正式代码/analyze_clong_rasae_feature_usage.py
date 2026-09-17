#!/usr/bin/env python3
"""统计固定RA-SAE全2560项Feature的使用覆盖和已有单项删除效应。

本入口只读100轮、K=256 RA-SAE已有train/val投影与val98单Feature
残差保留删除结果。dead/rare仅由train任意正激活的患者数定义；
固定Q99尺度下的强响应覆盖和功能排名是补充描述，不是剪枝授权。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
GROUP_ROOT = BASE / "full_dictionary_groups_20260909"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
DECISION_ROOT = BASE / "decision_alignment"
DURATION_ROOT = BASE / "duration100"
RARE_PATIENT_FRACTION = 0.01
STRONG_THRESHOLDS = (0.5, 1.0)


def patient_reduce(
    values: np.ndarray, metadata: pd.DataFrame, reduction: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """将逐图Feature矩阵按患者聚合。

    Args:
        values (np.ndarray): ``[N,F]``逐图数值。
        metadata (pd.DataFrame): N行、含``patient_id``和``label``的同序元数据。
        reduction (str): ``any``、``mean``或``mean_abs``。

    Returns:
        tuple[np.ndarray, pd.DataFrame]: ``[P,F]``患者矩阵和P行患者标签表。
    """
    if values.ndim != 2 or len(metadata) != values.shape[0]:
        raise ValueError("逐图矩阵与metadata行数不一致")
    if reduction not in {"any", "mean", "mean_abs"}:
        raise ValueError(f"未知患者聚合方式: {reduction}")
    if metadata.groupby("patient_id").label.nunique().max() != 1:
        raise ValueError("存在跨标签患者")
    patients = metadata[["patient_id", "label"]].drop_duplicates().sort_values("patient_id").reset_index(drop=True)
    lookup = {str(patient): index for index, patient in enumerate(patients.patient_id)}
    if reduction == "any":
        output = np.zeros((len(patients), values.shape[1]), dtype=bool)
        for row, patient in enumerate(metadata.patient_id):
            output[lookup[str(patient)]] |= values[row].astype(bool)
        return output, patients
    output = np.zeros((len(patients), values.shape[1]), dtype=np.float64)
    counts = np.zeros(len(patients), dtype=np.int64)
    source = np.abs(values) if reduction == "mean_abs" else values
    for row, patient in enumerate(metadata.patient_id):
        index = lookup[str(patient)]
        output[index] += source[row]
        counts[index] += 1
    if np.any(counts == 0):
        raise RuntimeError("存在无图像患者")
    return output / counts[:, None], patients


def split_usage(split: str) -> tuple[pd.DataFrame, dict]:
    """统计一个split的任意正激活与固定强响应覆盖。

    Args:
        split (str): 仅允许``train``或``val``。

    Returns:
        tuple[pd.DataFrame, dict]: 2560行Feature统计和split级核对摘要。
    """
    if split not in {"train", "val"}:
        raise ValueError("split仅允许train/val")
    peaks = np.load(GROUP_ROOT / f"{split}_peak_q99.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / f"{split}_metadata.csv", usecols=["patient_id", "label"])
    if peaks.shape != (len(metadata), 2560):
        raise RuntimeError(f"{split} peak矩阵shape异常: {peaks.shape}")
    rows = pd.DataFrame({"feature_id": np.arange(2560, dtype=int)})
    positive_image = np.asarray(peaks > 0)
    positive_patient, patients = patient_reduce(positive_image, metadata, "any")
    patient_mean_peak, _ = patient_reduce(np.asarray(peaks), metadata, "mean")
    rows[f"{split}_positive_image_count"] = positive_image.sum(0)
    rows[f"{split}_positive_image_fraction"] = positive_image.mean(0)
    rows[f"{split}_positive_patient_count"] = positive_patient.sum(0)
    rows[f"{split}_positive_patient_fraction"] = positive_patient.mean(0)
    rows[f"{split}_patient_mean_peak_q99"] = patient_mean_peak.mean(0)
    for label, name in ((1, "cancer"), (0, "noncancer")):
        selected_images = metadata.label.to_numpy() == label
        selected_patients = patients.label.to_numpy() == label
        rows[f"{split}_{name}_positive_image_count"] = positive_image[selected_images].sum(0)
        rows[f"{split}_{name}_positive_patient_count"] = positive_patient[selected_patients].sum(0)
    for threshold in STRONG_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        strong_image = np.asarray(peaks >= threshold)
        strong_patient, _ = patient_reduce(strong_image, metadata, "any")
        rows[f"{split}_ge_{suffix}q99_image_count"] = strong_image.sum(0)
        rows[f"{split}_ge_{suffix}q99_patient_count"] = strong_patient.sum(0)
        rows[f"{split}_ge_{suffix}q99_patient_fraction"] = strong_patient.mean(0)
    summary = {
        "images": int(len(metadata)), "patients": int(len(patients)),
        "patients_by_label": {str(key): int(value) for key, value in patients.groupby("label").size().items()},
        "minimum_positive_patient_count": int(positive_patient.sum(0).min()),
        "maximum_positive_patient_count": int(positive_patient.sum(0).max()),
        "features_with_zero_positive_patients": int((positive_patient.sum(0) == 0).sum()),
    }
    return rows, summary


def functional_effects() -> tuple[pd.DataFrame, dict]:
    """汇总已有val98单Feature精确删除效应，不重新forward。

    Args:
        None.

    Returns:
        tuple[pd.DataFrame, dict]: 2560行功能效应与val98队列说明。
    """
    delta_margin = np.load(DECISION_ROOT / "ra_val_delta_margin.npy")
    delta_probability = np.load(DECISION_ROOT / "ra_val_delta_probability.npy")
    metadata = pd.read_csv(DURATION_ROOT / "val_subset.csv", usecols=["patient_id", "label"])
    if delta_margin.shape != (len(metadata), 2560) or delta_probability.shape != delta_margin.shape:
        raise RuntimeError("val98已有删除矩阵shape异常")
    patient_signed_margin, patients = patient_reduce(delta_margin, metadata, "mean")
    patient_abs_margin, _ = patient_reduce(delta_margin, metadata, "mean_abs")
    patient_signed_probability, _ = patient_reduce(delta_probability, metadata, "mean")
    patient_abs_probability, _ = patient_reduce(delta_probability, metadata, "mean_abs")
    result = pd.DataFrame({
        "feature_id": np.arange(2560, dtype=int),
        "val98_patient_mean_signed_delta_margin": patient_signed_margin.mean(0),
        "val98_patient_mean_image_abs_delta_margin": patient_abs_margin.mean(0),
        "val98_patient_mean_signed_delta_probability": patient_signed_probability.mean(0),
        "val98_patient_mean_image_abs_delta_probability": patient_abs_probability.mean(0),
        "val98_max_image_abs_delta_margin": np.abs(delta_margin).max(0),
        "val98_max_image_abs_delta_probability": np.abs(delta_probability).max(0),
    })
    for label, name in ((1, "cancer"), (0, "noncancer")):
        selected = patients.label.to_numpy() == label
        result[f"val98_{name}_patient_mean_signed_delta_margin"] = patient_signed_margin[selected].mean(0)
        result[f"val98_{name}_patient_mean_image_abs_delta_margin"] = patient_abs_margin[selected].mean(0)
    ordered = result.sort_values(
        ["val98_patient_mean_image_abs_delta_margin", "feature_id"],
        ascending=[False, True], kind="stable",
    )
    rank = pd.Series(np.arange(1, 2561), index=ordered.feature_id)
    result["val98_abs_effect_rank_desc"] = result.feature_id.map(rank).astype(int)
    result["val98_abs_effect_percentile_desc"] = result.val98_abs_effect_rank_desc / 2560
    summary = {
        "images": int(len(metadata)), "patients": int(len(patients)),
        "patients_by_label": {str(key): int(value) for key, value in patients.groupby("label").size().items()},
        "effect_definition": "patient mean of per-image abs(delta_margin), then equal mean across patients",
        "effect_scope": "existing balanced val98/64-patient exploratory subset; not independent confirmation",
    }
    return result, summary


def main() -> None:
    """读取输出目录，生成独立Feature使用统计。

    Args:
        None: 输出路径由CLI提供。

    Returns:
        None: 写入CSV、JSON和简短结果说明。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train, train_summary = split_usage("train")
    val, val_summary = split_usage("val")
    effects, effect_summary = functional_effects()
    groups = pd.read_csv(GROUP_ROOT / "feature_groups.csv", usecols=["group", "feature_id", "representative"])
    result = train.merge(val, on="feature_id", validate="one_to_one").merge(
        effects, on="feature_id", validate="one_to_one"
    ).merge(groups, on="feature_id", validate="one_to_one")
    rare_max_patients = int(np.floor(train_summary["patients"] * RARE_PATIENT_FRACTION))
    result["usage_status"] = "active"
    result.loc[result.train_positive_patient_count.eq(0), "usage_status"] = "dead"
    result.loc[result.train_positive_patient_count.between(1, rare_max_patients), "usage_status"] = "rare"
    result["train_strong_0p5q99_rare_support"] = result.train_ge_0p5q99_patient_count.le(rare_max_patients)
    result["is_technical_representative"] = result.feature_id.eq(result.representative)
    result = result.sort_values("feature_id").reset_index(drop=True)
    if result.feature_id.tolist() != list(range(2560)):
        raise RuntimeError("Feature未恰好各出现一次")
    status_counts = {str(key): int(value) for key, value in result.usage_status.value_counts().items()}
    summary = {
        "model": "fixed_RA_duration100_K256_width2560",
        "definition": {
            "dead": "zero train patients with any positive activation",
            "rare": f"1 to {rare_max_patients} train patients with any positive activation (<=1%)",
            "active": f"more than {rare_max_patients} train patients with any positive activation",
            "strong_response_supplement": "peak/Q99 >= 0.5; train-only rare-support flag uses same <=1% patient cutoff",
        },
        "train": train_summary, "val": val_summary, "val98_function": effect_summary,
        "usage_status_counts": status_counts,
        "strong_0p5q99_rare_support_features": int(result.train_strong_0p5q99_rare_support.sum()),
        "minimum_train_positive_patient_fraction": float(result.train_positive_patient_fraction.min()),
        "median_train_positive_patient_fraction": float(result.train_positive_patient_fraction.median()),
        "limits": [
            "Top-K makes any-positive activation highly saturated; usage_status alone cannot reduce review workload here",
            "0.5xQ99 support and val98 functional rank are descriptive supplements, not pruning rules",
            "no retraining, regrouping, test reading, external reading, or feature deletion",
        ],
        "test_read": False, "external_read": False,
    }
    result.to_csv(args.output / "feature_usage_statistics.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    explanation = f"""# RA-SAE Feature基础使用统计（2026-09-15）

本轮固定100轮、K=256的RA-SAE，只复用现有train/val投影和val98单Feature删除结果。
未重训、未重分组、未删除Feature，未读取test或external。

## 主结果

- train为{train_summary['images']}图/{train_summary['patients']}人，val为{val_summary['images']}图/{val_summary['patients']}人。
- dead定义为train中没有任何患者出现正激活；rare定义为只在1至{rare_max_patients}位（<=1%）train患者中正激活。
- 实际结果为：active {status_counts.get('active', 0)}项、rare {status_counts.get('rare', 0)}项、dead {status_counts.get('dead', 0)}项。
- 覆盖最低的Feature也在train的311/1212位患者中有正激活；中位患者覆盖率为{result.train_positive_patient_fraction.median():.2%}。
- 因此，“是否出现任意正激活”在当前Top-K表示中高度饱和，不能用来减少医学复核数量。

## 补充统计

- 以固定train Q99尺度衡量，当峰值至少达0.5×Q99时记为强响应。按同一<=1%患者界限，有{int(result.train_strong_0p5q99_rare_support.sum())}项Feature的强响应支持较少。
- CSV同时保留>=1.0×Q99的train/val图像与患者覆盖，但不把Q99以上稀少直接命名为dead/rare。
- 已有val98/64人单Feature精确删除结果以“患者内先平均逐图绝对delta margin，再患者等权”排名。该排名是开发子集的描述，不是独立确认。

## 结论边界

本阶段不能依靠dead/rare基础状态直接缩小字典或医学工作量。后续应沿用整体路线，
使用功能相关性优先组织小批Feature图片，再请医学侧做“稳定有意义/疑似伪相关/混杂/无明显意义”的第一轮自由描述。
"""
    (args.output / "结果说明.md").write_text(explanation, encoding="utf-8")
    verification = {
        "features_exactly_once": True,
        "feature_count": len(result),
        "train_val_patient_overlap": 0,
        "status_counts_match": status_counts,
        "existing_effect_matrices_reused": True,
        "new_forward": False,
        "test_read": False, "external_read": False,
    }
    train_patients = set(pd.read_csv(CACHE / "train_metadata.csv", usecols=["patient_id"]).patient_id)
    val_patients = set(pd.read_csv(CACHE / "val_metadata.csv", usecols=["patient_id"]).patient_id)
    verification["train_val_patient_overlap"] = len(train_patients & val_patients)
    if verification["train_val_patient_overlap"]:
        raise RuntimeError("train/val患者交叉")
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
