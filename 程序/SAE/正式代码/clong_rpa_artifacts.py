#!/usr/bin/env python3
"""RP-A最终产物schema与科学/实现失败边界的纯函数校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from clong_rpa_bootstrap import METRIC_NAMES


PROTOCOL_FILES = (
    "rpa_eligible_protocol_v1.json",
    "rpa_null_fdr_protocol_v1.json",
    "rpa_coverage_protocol_v1.json",
    "rpa_bootstrap_protocol_v1.json",
    "rpa_spatial_protocol_v1.json",
    "rpa_gpu_identity_protocol_v1.json",
    "rpa_overall_protocol_v1.json",
)

REQUIRED_FORMAL_ARTIFACTS = {
    "run_manifest.json", "best_directed_hypotheses.csv", "reciprocal_edges.csv",
    "development_anchors.csv", "bootstrap_records.jsonl", "bootstrap_thresholds.json",
    "formal_metrics.json", "gpu_identity.json", "formal_result.json", "SHA256SUMS.txt",
}
FORBIDDEN_LARGE_ARTIFACTS = {
    "complete_candidate_matrix", "complete_pairwise_metric_matrix",
    "complete_pairwise_pvalue_matrix",
}


def file_sha256(path: Path) -> str:
    """计算普通文件SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_bundle_payload(directory: Path) -> tuple[list[dict[str, str]], str]:
    """只由排序后的冻结协议文件名和SHA生成bundle SHA。"""
    members = []
    lines = []
    for name in sorted(PROTOCOL_FILES):
        path = directory / name
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if protocol.get("status") != "frozen_2026-08-24":
            raise RuntimeError(f"协议尚未冻结: {name}")
        sha = file_sha256(path)
        members.append({"file": name, "sha256": sha})
        lines.append(f"{name}  {sha}\n")
    bundle_sha = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return members, bundle_sha


def validate_best_hypotheses(rows: list[dict]) -> None:
    """要求每个有向source hypothesis只保留唯一best。"""
    required = {"source_seed", "source_feature_id", "target_seed", "target_feature_id", "p_value", "bh_rejected"}
    keys = []
    for row in rows:
        if not required.issubset(row):
            raise ValueError("best hypothesis缺少字段")
        keys.append((row["source_seed"], row["source_feature_id"], row["target_seed"]))
        if not 0.0 <= float(row["p_value"]) <= 1.0:
            raise ValueError("p_value必须在[0,1]")
    if len(keys) != len(set(keys)):
        raise ValueError("同一有向source hypothesis出现多个best")


def validate_edges(rows: list[dict]) -> None:
    """要求最终edge在每个seed pair内一一对应。"""
    left, right = [], []
    for row in rows:
        if not row.get("both_directions_bh_rejected") or not row.get("reciprocal"):
            raise ValueError("正式edge必须双向BH通过且互为best")
        pair = tuple(sorted((int(row["seed_a"]), int(row["seed_b"]))))
        left.append((pair, int(row["feature_a"])))
        right.append((pair, int(row["feature_b"])))
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError("正式edge违反一一性")


def validate_anchors(rows: list[dict]) -> None:
    """正式development anchor必须各含42/43/44一个Feature。"""
    anchor_ids = []
    members = []
    for row in rows:
        anchor_ids.append(row["anchor_id"])
        member = (int(row["feature_42"]), int(row["feature_43"]), int(row["feature_44"]))
        members.append(member)
        if not row.get("edge_42_43") or not row.get("edge_42_44") or not row.get("edge_43_44"):
            raise ValueError("anchor不是严格3-clique")
    if len(anchor_ids) != len(set(anchor_ids)) or len(members) != len(set(members)):
        raise ValueError("anchor ID或成员重复")


def validate_bootstrap_records(records: list[dict]) -> None:
    """要求恰好400条、index完整，且三折六指标均为有限比例。"""
    if len(records) != 400 or sorted(row.get("replicate_index") for row in records) != list(range(400)):
        raise ValueError("bootstrap records必须恰好覆盖0..399")
    for row in records:
        status = row.get("status")
        if status not in {"completed", "replicate_structural_failure"}:
            raise ValueError("非法bootstrap status")
        if status == "replicate_structural_failure" and row.get("reason") != "null_stratification_infeasible":
            raise ValueError("非法结构失败reason")
        for fold in (42, 43, 44):
            for metric in METRIC_NAMES:
                value = float(row["fold_metrics"][str(fold)][metric])
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("bootstrap正式指标必须为[0,1]有限比例")


def validate_six_metrics(metrics: dict) -> None:
    """六项正式指标必须齐全且为[0,1]有限比例。"""
    if set(metrics) != set(METRIC_NAMES):
        raise ValueError("正式指标必须恰好为冻结六项")
    values = np.asarray(list(metrics.values()), dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("正式指标必须为[0,1]有限比例")


def validate_failure_record(record: dict) -> None:
    """区分科学失败与实现失败，禁止实现失败产出科学结论。"""
    failure_type = record.get("failure_type")
    if failure_type == "scientific_failure":
        if record.get("is_formal_result") is not True or record.get("downstream_allowed") is not False:
            raise ValueError("科学失败必须是正式结果且禁止下游")
    elif failure_type == "implementation_failure":
        required = {"failure_stage", "exception_type", "message", "traceback", "git_commit", "protocol_bundle_sha256"}
        if not required.issubset(record) or "scientific_conclusion" in record:
            raise ValueError("实现失败现场字段不完整或混入科学结论")
        if record.get("is_formal_result") is not False:
            raise ValueError("实现失败不得标为正式结果")
    else:
        raise ValueError("未知failure_type")


def validate_output_file_set(file_names: set[str], status: str) -> None:
    """按运行状态校验正式文件集合，并拒绝完整候选矩阵。"""
    names = set(file_names)
    if any(any(token in name for token in FORBIDDEN_LARGE_ARTIFACTS) for name in names):
        raise ValueError("正式输出禁止保存巨大完整候选矩阵")
    if status == "completed":
        missing = REQUIRED_FORMAL_ARTIFACTS - names
        if missing:
            raise ValueError(f"成功产物不完整: {sorted(missing)}")
        if "implementation_failure.json" in names:
            raise ValueError("成功产物不得混入实现失败现场")
    elif status == "scientific_failure":
        required = {"run_manifest.json", "formal_result.json", "SHA256SUMS.txt"}
        if not required.issubset(names) or "implementation_failure.json" in names:
            raise ValueError("科学失败缺少正式结论/血缘或混入实现失败")
    elif status == "implementation_failure":
        if "implementation_failure.json" not in names or "formal_result.json" in names:
            raise ValueError("实现失败必须保留现场且不得生成正式科学结论")
    else:
        raise ValueError("未知输出状态")
