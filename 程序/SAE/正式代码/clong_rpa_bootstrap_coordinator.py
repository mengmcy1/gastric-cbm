#!/usr/bin/env python3
"""归并RP-A bootstrap workers并生成冻结六门槛。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clong_rpa_artifacts import validate_bootstrap_records
from clong_rpa_bootstrap import (
    aggregate_thresholds,
    verify_full_train_self_consistency,
)
from clong_rpa_train_development import OUTPUT_ROOT as DEVELOPMENT_ROOT
from clong_s2c_matryoshka import file_sha256


WORKER_ROOT = DEVELOPMENT_ROOT / "bootstrap_workers"
OUTPUT_ROOT = DEVELOPMENT_ROOT / "bootstrap_calibration"
MATCHING_ROOT = DEVELOPMENT_ROOT / "full_train_matching" / "formal"


def parse_args() -> argparse.Namespace:
    """解析worker数量和debug记录数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-replicates", type=int, default=2)
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> list[dict]:
    """读取所有静态worker输出，拒绝缺失文件和重复index。"""
    root = WORKER_ROOT / ("debug" if args.debug else "formal")
    records = []
    for rank in range(args.worker_count):
        path = root / f"worker_{rank:02d}_of_{args.worker_count:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"缺少bootstrap worker输出: {path}")
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    expected = args.debug_replicates if args.debug else 400
    indices = [int(record["replicate_index"]) for record in records]
    if len(records) != expected or sorted(indices) != list(range(expected)):
        raise RuntimeError("bootstrap worker归并未完整唯一覆盖预期index")
    return sorted(records, key=lambda row: int(row["replicate_index"]))


def main() -> None:
    """归并records；正式生成门槛，debug只保存非正式路径证明。"""
    args = parse_args()
    if args.worker_count < 1:
        raise ValueError("worker_count必须为正整数")
    target = OUTPUT_ROOT / ("debug" if args.debug else "formal")
    if target.exists():
        raise FileExistsError(f"bootstrap coordinator输出已存在: {target}")
    records = load_records(args)
    target.mkdir(parents=True)
    records_path = target / "bootstrap_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records),
        encoding="utf-8",
    )
    if args.debug:
        result = {
            "status": "debug_merge_completed",
            "debug": True,
            "record_count": len(records),
            "formal_thresholds_generated": False,
        }
    else:
        validate_bootstrap_records(records)
        thresholds = aggregate_thresholds(records)
        full = json.loads(
            (MATCHING_ROOT / "full_train_metrics.json").read_text(encoding="utf-8")
        )
        verify_full_train_self_consistency(thresholds, full["fold_metrics"])
        result = {
            "status": "bootstrap_thresholds_frozen",
            "debug": False,
            "record_count": 400,
            "thresholds": thresholds,
            "full_train_self_consistency": True,
            "bootstrap_records_sha256": file_sha256(records_path),
            "full_train_metrics_sha256": file_sha256(
                MATCHING_ROOT / "full_train_metrics.json"
            ),
        }
    result_path = target / "bootstrap_thresholds.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (target / "SHA256SUMS.txt").write_text(
        f"{file_sha256(records_path)}  bootstrap_records.jsonl\n"
        f"{file_sha256(result_path)}  bootstrap_thresholds.json\n",
        encoding="utf-8",
    )
    print(f"bootstrap coordinator完成: {target}")


if __name__ == "__main__":
    main()
