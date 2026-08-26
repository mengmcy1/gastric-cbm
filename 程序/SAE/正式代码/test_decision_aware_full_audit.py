#!/usr/bin/env python3
"""Decision-aware全量数值审计的冻结排名与方向测试。"""

import unittest

import numpy as np

from audit_rp_sae_decision_awareness import (
    deterministic_ranks,
    overlap_counts,
    support_class,
    top_indices_from_ranks,
)


class DecisionAwareFullAuditTests(unittest.TestCase):
    def test_ties_use_anchor_id_ascending(self) -> None:
        scores = np.asarray([[1.0, 2.0, 2.0, 0.0]])
        anchors = np.asarray(["a00004", "a00003", "a00002", "a00001"])
        ranks = deterministic_ranks(scores, anchors)
        self.assertEqual(ranks.tolist(), [[3, 2, 1, 4]])

    def test_top_indices_follow_deterministic_ranks(self) -> None:
        ranks = np.asarray([[3, 2, 1, 4]], dtype=np.int16)
        self.assertEqual(top_indices_from_ranks(ranks, 2).tolist(), [[2, 1]])

    def test_overlap_is_intersection_over_k_not_jaccard(self) -> None:
        raw = np.asarray([[1, 2, 3, 4]], dtype=np.int16)
        functional = np.asarray([[2, 1, 4, 3]], dtype=np.int16)
        self.assertEqual(overlap_counts(raw, functional, 1).tolist(), [0])
        self.assertEqual(overlap_counts(raw, functional, 2).tolist(), [2])

    def test_correct_label_support_uses_signed_delta_and_label(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 1])
        delta = np.asarray([-1.0, 1.0, 1.0, -1.0, 0.0])
        self.assertEqual(support_class(labels, delta).tolist(), [
            "correct_label_supporting", "opposing",
            "correct_label_supporting", "opposing", "exact_zero",
        ])


if __name__ == "__main__":
    unittest.main()
