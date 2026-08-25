#!/usr/bin/env python3
"""RP-A-lite探索性B100子序列的纯函数与固定边界。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clong_rpa_bootstrap import METRIC_NAMES


LITE_REPLICATE_COUNT = 100
LITE_INDICES = tuple(range(LITE_REPLICATE_COUNT))
MISSING_INDICES = tuple(range(56, 100, 3))
EQUIVALENCE_INDEX = 53
SOURCE_BLOCK = 32
OLD_TARGET_BLOCK = 256
NEW_TARGET_BLOCK = 128
RETRY_TARGET_BLOCK = 64
RETRY_MISSING_INDICES = tuple(index for index in MISSING_INDICES if index != 56)


def load_jsonl(path: Path) -> list[dict]:
    """读取一个JSONL文件。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_record(record: dict) -> None:
    """校验Lite使用的单条bootstrap记录语义。"""
    status = record.get("status")
    if status not in {"completed", "replicate_structural_failure"}:
        raise ValueError(f"非法Lite replicate status: {status}")
    if (
        status == "replicate_structural_failure"
        and record.get("reason") != "null_stratification_infeasible"
    ):
        raise ValueError("非法Lite结构失败reason")
    folds = record.get("fold_metrics")
    if not isinstance(folds, dict) or set(folds) != {"42", "43", "44"}:
        raise ValueError("Lite replicate必须恰好包含三折")
    for fold in (42, 43, 44):
        metrics = folds[str(fold)]
        if set(metrics) != set(METRIC_NAMES):
            raise ValueError("Lite replicate每折必须恰好包含六指标")
        values = np.asarray([metrics[name] for name in METRIC_NAMES], dtype=np.float64)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Lite replicate指标必须是[0,1]内有限比例")
    if status == "replicate_structural_failure" and any(
        float(folds[str(fold)][name]) != 0.0
        for fold in (42, 43, 44)
        for name in METRIC_NAMES
    ):
        raise ValueError("Lite结构失败记录的三折六指标必须全为0")


def collect_source_records(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """从首轮worker读取0--99已完成记录并返回其他记录。"""
    all_records = [record for path in paths for record in load_jsonl(path)]
    indices = [int(record["replicate_index"]) for record in all_records]
    if len(indices) != len(set(indices)):
        raise RuntimeError("首轮bootstrap记录存在重复index")
    selected = sorted(
        (record for record in all_records if int(record["replicate_index"]) < 100),
        key=lambda row: int(row["replicate_index"]),
    )
    for record in selected:
        validate_record(record)
    expected_existing = sorted(set(LITE_INDICES) - set(MISSING_INDICES))
    observed = [int(record["replicate_index"]) for record in selected]
    if observed != expected_existing:
        raise RuntimeError("Lite首轮记录不是预期的85条固定index")
    remainder = [record for record in all_records if int(record["replicate_index"]) >= 100]
    return selected, remainder


def merge_lite_records(old_records: list[dict], new_records: list[dict]) -> list[dict]:
    """合并85条旧记录与15条新记录，强制唯一覆盖0--99。"""
    records = [*old_records, *new_records]
    for record in records:
        validate_record(record)
    indices = [int(record["replicate_index"]) for record in records]
    if len(indices) != len(set(indices)) or sorted(indices) != list(LITE_INDICES):
        raise RuntimeError("RP-A-lite必须唯一且完整覆盖0--99")
    return sorted(records, key=lambda row: int(row["replicate_index"]))


def exploratory_thresholds(records: list[dict]) -> dict[str, float]:
    """按最弱折Q0.05/lower计算B100探索性cutoff。"""
    ordered = merge_lite_records([], records)
    thresholds = {}
    for name in METRIC_NAMES:
        weakest = np.asarray([
            min(float(record["fold_metrics"][str(fold)][name]) for fold in (42, 43, 44))
            for record in ordered
        ])
        thresholds[name] = float(np.quantile(weakest, 0.05, method="lower"))
    return thresholds


def max_metric_difference(left: dict, right: dict) -> float:
    """返回两条同index记录三折六指标的最大绝对差。"""
    validate_record(left)
    validate_record(right)
    if int(left["replicate_index"]) != int(right["replicate_index"]):
        raise ValueError("等价性比较的replicate index不同")
    return max(
        abs(
            float(left["fold_metrics"][str(fold)][name])
            - float(right["fold_metrics"][str(fold)][name])
        )
        for fold in (42, 43, 44)
        for name in METRIC_NAMES
    )
