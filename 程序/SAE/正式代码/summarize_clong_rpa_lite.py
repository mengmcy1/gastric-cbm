#!/usr/bin/env python3
"""归并RP-A-lite固定B100并生成非正式探索性cutoff。"""

from __future__ import annotations

import json
from pathlib import Path

from clong_rpa_lite_core import (
    MISSING_INDICES,
    NEW_TARGET_BLOCK,
    OLD_TARGET_BLOCK,
    RETRY_MISSING_INDICES,
    RETRY_TARGET_BLOCK,
    SOURCE_BLOCK,
    collect_source_records,
    exploratory_thresholds,
    load_jsonl,
    max_metric_difference,
    merge_lite_records,
)
from clong_rpa_train_development import OUTPUT_ROOT as SOURCE_ROOT
from clong_s2c_matryoshka import file_sha256
from train_utils import git_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Lite_Exploratory_20260825"
SOURCE_WORKERS = SOURCE_ROOT / "bootstrap_workers/formal"
SOURCE_FILES = tuple(SOURCE_WORKERS / f"worker_{rank:02d}_of_03.jsonl" for rank in range(3))
SOURCE_MATCHING = SOURCE_ROOT / "full_train_matching/formal/full_train_metrics.json"


def main() -> None:
    """验证等价性、唯一覆盖0--99，并保存独立Lite产物。"""
    target = OUTPUT_ROOT / "summary"
    if target.exists():
        raise FileExistsError(f"RP-A-lite汇总目录已存在: {target}")
    for path in (*SOURCE_FILES, SOURCE_MATCHING):
        if not path.is_file():
            raise FileNotFoundError(f"缺少Lite源产物: {path}")

    old_records, ignored_records = collect_source_records(list(SOURCE_FILES))
    old_by_index = {int(row["replicate_index"]): row for row in old_records}
    equivalence_differences = {}
    for target_block in (NEW_TARGET_BLOCK, RETRY_TARGET_BLOCK):
        path = OUTPUT_ROOT / f"equivalence/replicate53_32x{target_block}.jsonl"
        rows = load_jsonl(path)
        if len(rows) != 1:
            raise RuntimeError(f"Lite 32x{target_block}等价性探针必须恰好一条")
        difference = max_metric_difference(old_by_index[53], rows[0])
        if difference != 0.0:
            raise RuntimeError(f"32x256与32x{target_block}指标不等价: {difference}")
        equivalence_differences[str(target_block)] = difference

    initial_partial = OUTPUT_ROOT / "missing_workers/worker_00_of_03.partial.jsonl"
    initial_records = load_jsonl(initial_partial)
    if [int(row["replicate_index"]) for row in initial_records] != [56]:
        raise RuntimeError("Lite首次恢复现场必须只包含已完成replicate56")

    retry_records = []
    new_paths = []
    for rank in range(3):
        path = OUTPUT_ROOT / f"missing_retry1/worker_{rank:02d}_of_03.jsonl"
        new_paths.append(path)
        retry_records.extend(load_jsonl(path))
    new_records = [*initial_records, *retry_records]
    observed_missing = sorted(int(row["replicate_index"]) for row in new_records)
    if observed_missing != list(MISSING_INDICES):
        raise RuntimeError("Lite新记录未恰好覆盖固定缺失15条")
    if sorted(int(row["replicate_index"]) for row in retry_records) != list(RETRY_MISSING_INDICES):
        raise RuntimeError("Lite retry1未恰好覆盖剩余14条")
    records = merge_lite_records(old_records, new_records)
    thresholds = exploratory_thresholds(records)

    target.mkdir(parents=True)
    records_path = target / "exploratory_bootstrap_records_0_99.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records),
        encoding="utf-8",
    )
    matching = json.loads(SOURCE_MATCHING.read_text(encoding="utf-8"))
    result = {
        "status": "exploratory_bootstrap_screen_completed",
        "is_formal_rpa_result": False,
        "formal_rpa_protocol_modified": False,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        "confirmation_seeds_evaluated": [],
        "replicate_indices": [0, 99],
        "record_count": 100,
        "old_record_count": len(old_records),
        "new_record_count": len(new_records),
        "ignored_first_run_records_ge_100": len(ignored_records),
        "old_execution": {
            "git_commit": "c96ce66fda499f9d451e88499e1a7c303d0798e8",
            "source_block": SOURCE_BLOCK,
            "target_block": OLD_TARGET_BLOCK,
        },
        "new_execution": {
            "git": git_snapshot(),
            "source_block": SOURCE_BLOCK,
            "target_blocks": [NEW_TARGET_BLOCK, RETRY_TARGET_BLOCK],
        },
        "block_equivalence_replicate": 53,
        "block_equivalence_max_absolute_difference": equivalence_differences,
        "quantile": 0.05,
        "quantile_method": "lower",
        "exploratory_cutoffs": thresholds,
        "development_strict_anchor_count": int(matching["strict_anchor_count"]),
        "source_sha256": {
            **{path.name: file_sha256(path) for path in SOURCE_FILES},
            "full_train_metrics.json": file_sha256(SOURCE_MATCHING),
        },
        "new_worker_sha256": {
            "initial_partial_with_replicate56": file_sha256(initial_partial),
            **{path.name: file_sha256(path) for path in new_paths},
        },
    }
    result_path = target / "exploratory_bootstrap_cutoffs.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (target / "SHA256SUMS.txt").write_text(
        f"{file_sha256(records_path)}  {records_path.name}\n"
        f"{file_sha256(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    print(f"RP-A-lite汇总完成: {target}")


if __name__ == "__main__":
    main()
