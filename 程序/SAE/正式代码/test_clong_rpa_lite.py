#!/usr/bin/env python3
"""RP-A-lite固定B100、探索性cutoff与隔离边界测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from clong_rpa_bootstrap import BOOTSTRAP_COUNT, METRIC_NAMES
from clong_rpa_lite_core import (
    EQUIVALENCE_INDEX,
    LITE_INDICES,
    MISSING_INDICES,
    NEW_TARGET_BLOCK,
    OLD_TARGET_BLOCK,
    RETRY_MISSING_INDICES,
    RETRY_TARGET_BLOCK,
    collect_source_records,
    exploratory_thresholds,
    max_metric_difference,
    merge_lite_records,
)
from clong_rpa_lite_worker import assigned_indices


def record(index: int, value: float | None = None) -> dict:
    """构造三折六指标的合法测试记录。"""
    metric = float(index + 1) / 101.0 if value is None else float(value)
    return {
        "replicate_index": index,
        "status": "completed",
        "fold_metrics": {
            str(fold): {name: metric for name in METRIC_NAMES}
            for fold in (42, 43, 44)
        },
    }


class Args:
    """最小worker分工参数。"""

    def __init__(self, mode: str, rank: int, count: int):
        self.mode = mode
        self.worker_rank = rank
        self.worker_count = count


class RPALiteTests(unittest.TestCase):
    """验证Lite不改写正式B400且只使用固定子序列。"""

    def test_formal_bootstrap_count_remains_400(self) -> None:
        self.assertEqual(BOOTSTRAP_COUNT, 400)
        self.assertEqual(LITE_INDICES, tuple(range(100)))

    def test_missing_indices_are_exactly_fifteen(self) -> None:
        self.assertEqual(MISSING_INDICES, tuple(range(56, 100, 3)))
        self.assertEqual(len(MISSING_INDICES), 15)

    def test_three_workers_partition_missing_indices(self) -> None:
        groups = [assigned_indices(Args("missing", rank, 3)) for rank in range(3)]
        flattened = [index for group in groups for index in group]
        self.assertEqual(sorted(flattened), list(MISSING_INDICES))
        self.assertTrue(all(len(group) == 5 for group in groups))

    def test_retry_workers_cover_remaining_fourteen(self) -> None:
        groups = [assigned_indices(Args("retry1", rank, 3)) for rank in range(3)]
        flattened = [index for group in groups for index in group]
        self.assertEqual(sorted(flattened), list(RETRY_MISSING_INDICES))
        self.assertEqual([len(group) for group in groups], [5, 5, 4])

    def test_equivalence_mode_only_computes_replicate53(self) -> None:
        self.assertEqual(assigned_indices(Args("equivalence", 0, 1)), (EQUIVALENCE_INDEX,))
        with self.assertRaises(ValueError):
            assigned_indices(Args("equivalence", 0, 3))

    def test_source_collection_requires_exact_existing_85(self) -> None:
        existing = sorted(set(range(100)) - set(MISSING_INDICES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(
                "".join(__import__("json").dumps(record(index)) + "\n" for index in existing),
                encoding="utf-8",
            )
            selected, remainder = collect_source_records([path])
        self.assertEqual(len(selected), 85)
        self.assertEqual(remainder, [])

    def test_merge_requires_unique_complete_zero_to_99(self) -> None:
        records = [record(index) for index in range(100)]
        self.assertEqual(len(merge_lite_records(records[:85], records[85:])), 100)
        with self.assertRaises(RuntimeError):
            merge_lite_records(records[:-1], [])

    def test_lower_quantile_is_fifth_observation(self) -> None:
        records = [record(index) for index in range(100)]
        cutoffs = exploratory_thresholds(records)
        expected = float(np.quantile(np.arange(1, 101) / 101.0, 0.05, method="lower"))
        self.assertEqual(expected, 5.0 / 101.0)
        self.assertTrue(all(value == expected for value in cutoffs.values()))

    def test_metric_equivalence_is_exact(self) -> None:
        left = record(53, 0.5)
        right = record(53, 0.5)
        self.assertEqual(max_metric_difference(left, right), 0.0)
        right["fold_metrics"]["42"][METRIC_NAMES[0]] += 1e-8
        self.assertGreater(max_metric_difference(left, right), 0.0)

    def test_execution_blocks_are_explicit(self) -> None:
        self.assertEqual(OLD_TARGET_BLOCK, 256)
        self.assertEqual(NEW_TARGET_BLOCK, 128)
        self.assertEqual(RETRY_TARGET_BLOCK, 64)


if __name__ == "__main__":
    unittest.main()
