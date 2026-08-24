#!/usr/bin/env python3
"""RP-A null/FDR纯函数回归测试。"""

from __future__ import annotations

import hashlib
import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from clong_rpa_null_fdr import (  # noqa: E402
    DirectedHypothesis,
    assign_target_strata,
    benjamini_hochberg,
    choose_best_candidate,
    edge_scores,
    exact_conditional_search_p,
    frozen_train_cdf_percentile,
    load_protocol,
    raw_direction_gates_pass,
    reciprocal_edges,
    train_midrank_percentile,
    validate_target_strata,
)


class NullFDRTests(unittest.TestCase):
    """验证strata、百分位、精确null、BH与RNN边界。"""

    def test_protocol_has_no_monte_carlo_b_null(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["null"]["type"], "exact_conditional_permutation")
        self.assertIsNone(protocol["null"]["b_null"])
        self.assertEqual(protocol["fdr"]["q"], 0.05)

    def test_protocol_eligible_lineage_matches_formal_files(self) -> None:
        protocol = load_protocol()
        paths = {
            "eligible_protocol_sha256": REPO_ROOT / "程序/SAE/正式代码/rpa_eligible_protocol_v1.json",
            "eligible_audit_summary_sha256": REPO_ROOT / (
                "结果/SAE/RP_A_Eligible校准_20260821/"
                "seed42_feature_aggregation_audit_summary.json"
            ),
            "eligible_feature_csv_sha256": REPO_ROOT / (
                "结果/SAE/RP_A_Eligible校准_20260821/"
                "seed42_feature_aggregation_audit.csv"
            ),
        }
        for key, path in paths.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(protocol[key], digest)

    def test_hierarchical_strata_are_balanced_and_deterministic(self) -> None:
        coverage = np.repeat(np.arange(8), 8).astype(float)
        frequency = np.tile(np.arange(8), 8).astype(float)
        ids = np.arange(64)
        strata = assign_target_strata(
            coverage, frequency, ids,
        )
        self.assertEqual(validate_target_strata(strata, minimum_size=4), {
            stratum: 4 for stratum in range(16)
        })
        np.testing.assert_array_equal(strata, assign_target_strata(coverage, frequency, ids))

    def test_strata_reject_duplicate_feature_ids(self) -> None:
        with self.assertRaises(ValueError):
            assign_target_strata(np.arange(16), np.arange(16), np.zeros(16, dtype=int))

    def test_small_nonempty_stratum_fails_without_merge(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_target_strata(np.array([0] * 31 + [1] * 40), minimum_size=32)

    def test_train_midrank_ties(self) -> None:
        result = train_midrank_percentile(np.array([1.0, 2.0, 2.0, 9.0]), np.ones(4, bool))
        np.testing.assert_allclose(result, np.array([0.25, 0.625, 0.625, 1.0]))

    def test_val_uses_frozen_train_cdf(self) -> None:
        result = frozen_train_cdf_percentile(
            np.array([1.0, 2.0, 2.0, 4.0]),
            np.array([0.0, 2.0, 3.0, 5.0]),
            np.ones(4, bool),
        )
        np.testing.assert_allclose(result, np.array([0.0, 0.5, 0.75, 1.0]))

    def test_edge_score_is_structural_behavior_bottleneck(self) -> None:
        behavior, score = edge_scores(
            np.array([0.9, 0.4]), np.array([0.8, 0.8]), np.array([0.7, 0.6]),
            np.array([0.6, 0.7]), np.ones(2, bool),
        )
        np.testing.assert_allclose(behavior, np.array([0.7, 0.7]))
        np.testing.assert_allclose(score, np.array([0.7, 0.4]))

    def test_best_candidate_tie_order(self) -> None:
        index = choose_best_candidate(
            np.array([0.8, 0.8, 0.8]), np.array([0.7, 0.9, 0.9]),
            np.array([9, 8, 3]), np.ones(3, bool),
        )
        self.assertEqual(index, 2)

    def test_exact_null_matches_enumerable_two_item_case(self) -> None:
        # decoder高位与behavior高位在2个候选中随机对齐，p=1/2。
        p = exact_conditional_search_p(
            np.array([1.0, 0.5]), np.array([1.0, 0.5]), np.array([0, 0]),
            np.ones(2, bool), observed_score=1.0,
        )
        self.assertAlmostEqual(p, 0.5)

    def test_exact_null_multistratum_product(self) -> None:
        # 两个独立2-item strata都无高高重合的概率为1/4，tail=3/4。
        p = exact_conditional_search_p(
            np.array([1, 0, 1, 0], float), np.array([1, 0, 1, 0], float),
            np.array([0, 0, 1, 1]), np.ones(4, bool), observed_score=1.0,
        )
        self.assertAlmostEqual(p, 0.75)

    def test_exact_null_matches_bruteforce_permutations(self) -> None:
        decoder = np.array([1.0, 0.8, 0.4, 0.2])
        behavior = np.array([0.9, 0.7, 0.3, 0.1])
        observed = 0.7
        exact = exact_conditional_search_p(
            decoder, behavior, np.zeros(4, dtype=int), np.ones(4, bool), observed,
        )
        exceed = 0
        total = 0
        for permuted in itertools.permutations(behavior.tolist()):
            total += 1
            score = np.max(np.minimum(decoder, np.asarray(permuted)))
            exceed += int(score >= observed)
        self.assertAlmostEqual(exact, exceed / total)

    def test_raw_direction_gates(self) -> None:
        self.assertTrue(raw_direction_gates_pass(0.1, 0.2, 0.3, 0.4))
        self.assertFalse(raw_direction_gates_pass(0.1, 0.0, 0.3, 0.4))
        self.assertFalse(raw_direction_gates_pass(0.1, math.nan, 0.3, 0.4))

    def test_bh_and_tie_cutoff(self) -> None:
        hypotheses = [
            DirectedHypothesis(1, 2, 0, 10, 0.001),
            DirectedHypothesis(1, 2, 1, 11, 0.02),
            DirectedHypothesis(2, 1, 10, 0, 0.001),
            DirectedHypothesis(2, 1, 11, 1, 0.9),
        ]
        rejected, cutoff = benjamini_hochberg(hypotheses, q=0.05)
        self.assertEqual(cutoff, 0.02)
        self.assertEqual(len(rejected), 3)

    def test_rnn_rejects_duplicate_source_hypotheses(self) -> None:
        hypotheses = [
            DirectedHypothesis(1, 2, 0, 10, 0.001),
            DirectedHypothesis(1, 2, 0, 11, 0.002),
        ]
        with self.assertRaises(RuntimeError):
            reciprocal_edges(hypotheses, set())

    def test_rnn_requires_both_directions_and_is_one_to_one(self) -> None:
        hypotheses = [
            DirectedHypothesis(1, 2, 0, 10, 0.001),
            DirectedHypothesis(1, 2, 1, 11, 0.001),
            DirectedHypothesis(2, 1, 10, 0, 0.001),
            DirectedHypothesis(2, 1, 11, 99, 0.001),
        ]
        rejected = {
            (h.source_seed, h.target_seed, h.source_feature_id, h.target_feature_id)
            for h in hypotheses
        }
        self.assertEqual(reciprocal_edges(hypotheses, rejected), [(0, 10)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
