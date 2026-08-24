#!/usr/bin/env python3
"""组装RP-A development正式结果，并区分科学失败与实现不完整。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from clong_rpa_artifacts import (
    protocol_bundle_payload,
    validate_anchors,
    validate_best_hypotheses,
    validate_bootstrap_records,
    validate_edges,
    validate_failure_record,
    validate_output_file_set,
)
from clong_rpa_bootstrap import METRIC_NAMES
from clong_rpa_train_development import OUTPUT_ROOT as DEVELOPMENT_ROOT
from clong_s2c_matryoshka import file_sha256


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
IDENTITY_AUDIT = PROJECT_ROOT / "结果/SAE/RP_A_GPU_Identity_20260824/identity_audit_v2.json"
IDENTITY_SHA = "bb254f91a7726781d4a53a35eb8b74c437aba2317def1dd6292c423bec7b685e"
PROTOCOL_BUNDLE_SHA = "768da344bfd3d49ca518528bc043a4eb2ef5b1b4f76b449c42e00c7223093280"


def parse_args() -> argparse.Namespace:
    """解析debug模式；正式模式只读取已完成阶段产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    """记录执行时Git提交，不把它混入冻结protocol bundle。"""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def run_manifest(debug: bool) -> dict:
    """生成运行血缘；协议bundle与代码版本分开记录。"""
    members, bundle = protocol_bundle_payload(SCRIPT_DIR)
    if bundle != PROTOCOL_BUNDLE_SHA:
        raise RuntimeError("运行时protocol bundle SHA与冻结值不一致")
    return {
        "stage": "RP-A development-calibration",
        "debug": debug,
        "development_seeds": [42, 43, 44],
        "confirmation_seeds_started": [],
        "git_commit": git_commit(),
        "protocol_bundle_sha256": bundle,
        "protocol_members": members,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }


def write_sha_manifest(target: Path) -> None:
    """为最终目录内除SHA文件自身外的普通文件生成确定性清单。"""
    names = sorted(path.name for path in target.iterdir() if path.is_file()
                   and path.name != "SHA256SUMS.txt")
    (target / "SHA256SUMS.txt").write_text(
        "\n".join(f"{file_sha256(target / name)}  {name}" for name in names) + "\n",
        encoding="utf-8",
    )


def scientific_failure(target: Path, manifest: dict, reason: str, details: dict) -> None:
    """保存协议允许的正式科学失败，不伪造未完成阶段证据。"""
    target.mkdir(parents=True)
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    result = {
        "status": reason,
        "failure_type": "scientific_failure",
        "is_formal_result": True,
        "downstream_allowed": False,
        "details": details,
    }
    validate_failure_record(result)
    (target / "formal_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_sha_manifest(target)
    validate_output_file_set({path.name for path in target.iterdir()}, "scientific_failure")


def read_jsonl(path: Path) -> list[dict]:
    """读取bootstrap逐replicate证据。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    """根据阶段产物生成科学失败或完整development结果。"""
    args = parse_args()
    target = DEVELOPMENT_ROOT / ("final_debug" if args.debug else "formal_development_result")
    if target.exists():
        raise FileExistsError(f"最终输出已存在，禁止覆盖: {target}")
    matching = DEVELOPMENT_ROOT / "full_train_matching" / ("debug_v2" if args.debug else "formal")
    matching_result = json.loads(
        (matching / "full_train_metrics.json").read_text(encoding="utf-8")
    )
    manifest = run_manifest(bool(args.debug))
    anchor_count = int(matching_result["strict_anchor_count"])
    if anchor_count < 100:
        scientific_failure(
            target, manifest, "development_calibration_infeasible",
            {"strict_anchor_count": anchor_count, "minimum_required": 100,
             "matching_result_sha256": file_sha256(matching / "full_train_metrics.json")},
        )
        print(f"RP-A development科学失败已落盘: anchors={anchor_count}")
        return
    if args.debug:
        raise RuntimeError("debug随机小模型不应越过正式anchor门槛")

    bootstrap = DEVELOPMENT_ROOT / "bootstrap_calibration/formal"
    val = DEVELOPMENT_ROOT / "val_reproduction/formal"
    thresholds_payload = json.loads(
        (bootstrap / "bootstrap_thresholds.json").read_text(encoding="utf-8")
    )
    records = read_jsonl(bootstrap / "bootstrap_records.jsonl")
    validate_bootstrap_records(records)
    if thresholds_payload["status"] == "bootstrap_calibration_infeasible":
        scientific_failure(
            target, manifest, "bootstrap_calibration_infeasible",
            {"reason": thresholds_payload["reason"],
             "bootstrap_record_count": len(records),
             "bootstrap_result_sha256": file_sha256(
                 bootstrap / "bootstrap_thresholds.json"
             )},
        )
        print("RP-A bootstrap校准不可行，正式停止")
        return
    if thresholds_payload["status"] != "bootstrap_thresholds_frozen":
        raise RuntimeError("bootstrap状态不是正式成功或科学失败")
    val_payload = json.loads((val / "val_reproduction.json").read_text(encoding="utf-8"))
    if val_payload["status"] == "val_reproduction_failure":
        scientific_failure(
            target, manifest, "val_reproduction_failure",
            {"val_metrics": val_payload["minimum_fold_metrics"],
             "thresholds": val_payload["thresholds"],
             "val_result_sha256": file_sha256(val / "val_reproduction.json")},
        )
        print("RP-A development val复现失败，正式停止")
        return
    if val_payload["status"] != "val_reproduction_passed":
        raise RuntimeError("val复现状态不是正式成功或科学失败")

    if file_sha256(IDENTITY_AUDIT) != IDENTITY_SHA:
        raise RuntimeError("冻结GPU identity audit SHA不一致")
    hypotheses = pd.read_csv(matching / "best_directed_hypotheses.csv").to_dict("records")
    edges = pd.read_csv(matching / "reciprocal_edges.csv").to_dict("records")
    anchors = pd.read_csv(matching / "development_anchors.csv").to_dict("records")
    validate_best_hypotheses(hypotheses)
    validate_edges(edges)
    validate_anchors(anchors)

    target.mkdir(parents=True)
    copies = {
        matching / "best_directed_hypotheses.csv": target / "best_directed_hypotheses.csv",
        matching / "reciprocal_edges.csv": target / "reciprocal_edges.csv",
        matching / "development_anchors.csv": target / "development_anchors.csv",
        bootstrap / "bootstrap_records.jsonl": target / "bootstrap_records.jsonl",
        bootstrap / "bootstrap_thresholds.json": target / "bootstrap_thresholds.json",
        IDENTITY_AUDIT: target / "gpu_identity.json",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    train_minimum = {
        metric: min(float(matching_result["fold_metrics"][str(seed)][metric])
                    for seed in (42, 43, 44))
        for metric in METRIC_NAMES
    }
    formal_metrics = {
        "train_fold_metrics": matching_result["fold_metrics"],
        "train_minimum_fold_metrics": train_minimum,
        "bootstrap_thresholds": thresholds_payload["thresholds"],
        "val_fold_metrics": val_payload["fold_metrics"],
        "val_minimum_fold_metrics": val_payload["minimum_fold_metrics"],
    }
    (target / "formal_metrics.json").write_text(
        json.dumps(formal_metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    result = {
        "status": "development_calibration_passed",
        "failure_type": None,
        "is_formal_result": True,
        "downstream_allowed": True,
        "strict_anchor_count": anchor_count,
        "bootstrap_record_count": len(records),
        "val_reproduction_passed": True,
        "next_allowed_seeds": [202, 503],
        "seed911_locked": True,
    }
    (target / "formal_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_sha_manifest(target)
    validate_output_file_set({path.name for path in target.iterdir()}, "completed")
    print(f"RP-A development正式结果完成: {target}")


if __name__ == "__main__":
    main()
