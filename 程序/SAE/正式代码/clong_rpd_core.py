#!/usr/bin/env python3
"""RP-D病例确定性抽样与hard-negative选择纯函数。"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


BINS = ("high", "low", "mid", "zero")


def stable_hash(anchor_id: str, patient_id: str, relative_path: str) -> str:
    """返回冻结RP-D病例选择哈希。"""
    value = f"rpd_atlas_v1|{anchor_id}|{patient_id}|{relative_path}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def blind_hash(anchor_id: str, case_id: str) -> str:
    """返回医生盲审顺序哈希。"""
    value = f"rpd_blind_v1|{anchor_id}|{case_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_patient_table(
    image_frame: pd.DataFrame,
    image_peak: np.ndarray,
    image_mass: np.ndarray,
    anchor_id: str,
) -> pd.DataFrame:
    """按患者聚合peak，并确定每位患者唯一代表图。"""
    if len(image_frame) != len(image_peak) or len(image_frame) != len(image_mass):
        raise ValueError("图像metadata与激活长度不一致")
    frame = image_frame[["relative_path", "patient_id", "label", "source"]].copy()
    frame["image_index"] = np.arange(len(frame), dtype=np.int64)
    frame["image_peak_activation"] = np.asarray(image_peak, dtype=np.float64)
    frame["image_mass_activation"] = np.asarray(image_mass, dtype=np.float64)
    frame["selection_hash"] = [
        stable_hash(anchor_id, str(row.patient_id), str(row.relative_path))
        for row in frame.itertuples(index=False)
    ]
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        if group.label.nunique() != 1 or group.source.nunique() != 1:
            raise RuntimeError(f"患者标签或来源不唯一: {patient_id}")
        patient_peak = float(group.image_peak_activation.max())
        if patient_peak > 0:
            selected = group.sort_values(
                ["image_peak_activation", "image_mass_activation", "selection_hash"],
                ascending=[False, False, True], kind="stable",
            ).iloc[0]
        else:
            selected = group.sort_values("selection_hash", kind="stable").iloc[0]
        rows.append({
            "patient_id": str(patient_id),
            "label": int(selected.label),
            "source": str(selected.source),
            "patient_peak_activation": patient_peak,
            "image_index": int(selected.image_index),
            "relative_path": str(selected.relative_path),
            "image_peak_activation": float(selected.image_peak_activation),
            "image_mass_activation": float(selected.image_mass_activation),
            "selection_hash": str(selected.selection_hash),
        })
    return pd.DataFrame(rows)


def select_activation_bins(
    patient_table: pd.DataFrame,
    anchor_id: str,
    label: int,
    count: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """按High→Low→Mid→Zero顺序选择互斥患者。"""
    label_frame = patient_table[patient_table.label.eq(int(label))].copy()
    active = label_frame[label_frame.patient_peak_activation.gt(0)].sort_values(
        ["patient_peak_activation", "selection_hash"],
        ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    active["active_rank"] = np.arange(1, len(active) + 1, dtype=np.int64)
    midpoint = (len(active) + 1) / 2
    used: set[str] = set()
    selected_rows = []
    shortfalls: dict[str, int] = {}
    for bin_name in BINS:
        if bin_name == "high":
            candidates = active.sort_values(
                ["patient_peak_activation", "selection_hash"],
                ascending=[False, True], kind="stable",
            )
        elif bin_name == "low":
            candidates = active.sort_values(
                ["patient_peak_activation", "selection_hash"],
                ascending=[True, True], kind="stable",
            )
        elif bin_name == "mid":
            candidates = active.assign(
                midpoint_distance=(active.active_rank - midpoint).abs(),
            ).sort_values(
                ["midpoint_distance", "selection_hash"],
                ascending=[True, True], kind="stable",
            )
        else:
            candidates = label_frame[label_frame.patient_peak_activation.eq(0)].sort_values(
                "selection_hash", kind="stable",
            )
        candidates = candidates[~candidates.patient_id.astype(str).isin(used)].head(int(count))
        shortfalls[bin_name] = int(count - len(candidates))
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            record = row.to_dict()
            record.update({
                "anchor_id": anchor_id,
                "canonical_seed": 42,
                "case_role": bin_name,
                "activation_bin": bin_name,
                "selection_rank": rank,
                "selection_status": "selected",
            })
            selected_rows.append(record)
            used.add(str(row.patient_id))
    return pd.DataFrame(selected_rows), shortfalls


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """计算单个query与候选pooled表示的cosine。"""
    query = np.asarray(query, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    denominator = np.linalg.norm(candidates, axis=1) * np.linalg.norm(query)
    return (candidates @ query) / denominator


def select_hard_negative(
    query: pd.Series,
    patient_table: pd.DataFrame,
    selected_low: pd.DataFrame,
    pooled_features: np.ndarray,
) -> tuple[pd.Series | None, float | None, str]:
    """按zero优先、选中Low回退规则选择一个hard negative。"""
    candidates = patient_table[
        patient_table.label.eq(int(query.label))
        & patient_table.patient_id.astype(str).ne(str(query.patient_id))
        & patient_table.patient_peak_activation.eq(0)
    ].copy()
    pool_status = "zero"
    if candidates.empty:
        candidates = selected_low[
            selected_low.patient_id.astype(str).ne(str(query.patient_id))
        ].copy()
        pool_status = "selected_low_fallback"
    if candidates.empty:
        return None, None, "insufficient_candidates"
    candidate_index = candidates.image_index.to_numpy(dtype=np.int64)
    similarities = cosine_similarity(
        pooled_features[int(query.image_index)], pooled_features[candidate_index],
    )
    candidates["hard_negative_cosine"] = similarities
    selected = candidates.sort_values(
        ["hard_negative_cosine", "selection_hash"],
        ascending=[False, True], kind="stable",
    ).iloc[0]
    return selected, float(selected.hard_negative_cosine), pool_status
