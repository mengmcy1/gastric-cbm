#!/usr/bin/env python3
"""RP-A development bootstrap纯函数回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from clong_rpa_bootstrap import (  # noqa: E402
    ACTIVE_EPS,
    BOOTSTRAP_COUNT,
    METRIC_NAMES,
    aggregate_thresholds,
    activation_presence,
    eligible_membership,
    generate_bootstrap_plans,
    load_protocol,
    structural_failure_record,
    top_patient_instances,
    validate_formal_patient_table,
    verify_full_train_self_consistency,
    weighted_mean,
    weighted_spatial_patient_mean,
    weighted_sum,
    weighted_union_positive_spearman,
    worker_replicate_indices,
)


def toy_patients() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为六个冻结strata各构造2位患者。"""
    ids, labels, sources = [], [], []
    source_names = ("武大省人民", "第一届早癌大赛", "第二届早癌大赛")
    for label in (0, 1):
        for source in source_names:
            for local in range(2):
                ids.append(f"{label}-{source}-{local}")
                labels.append(label)
                sources.append(source)
    return np.asarray(ids), np.asarray(labels), np.asarray(sources)


class BootstrapTests(unittest.TestCase):
    """验证抽样、重复instance、加权聚合和门槛归并。"""

    def test_protocol_is_candidate_and_uses_active_eps(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["status"], "candidate_freeze_pending_user_confirmation")
        self.assertEqual(protocol["bootstrap"]["replicate_count"], 400)
        self.assertEqual(protocol["eligible"]["active_eps"], ACTIVE_EPS)
        values = np.array([[0.0, ACTIVE_EPS / 2], [0.0, ACTIVE_EPS * 2]])
        np.testing.assert_array_equal(
            activation_presence(values, axis=1), np.array([False, True])
        )

    def test_plans_are_deterministic_and_preserve_stratum_sizes(self) -> None:
        ids, labels, sources = toy_patients()
        first = generate_bootstrap_plans(ids, labels, sources)
        second = generate_bootstrap_plans(ids, labels, sources)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (400, 12))
        for plan in first:
            self.assertEqual(int(plan.sum()), 12)
            for label in (0, 1):
                for source in np.unique(sources):
                    mask = (labels == label) & (sources == source)
                    self.assertEqual(int(plan[mask].sum()), 2)

    def test_eligible_weighting_matches_explicit_instances_and_eps(self) -> None:
        presence = np.array([[True, False], [False, True], [True, True]])
        frequency = np.array([[0.01, 0.0], [0.0, 0.01], [0.02, 0.03]])
        weights = np.array([2, 0, 1])
        positive, active, _ = eligible_membership(
            presence, frequency, weights, np.ones(2, bool)
        )
        expanded = np.repeat(np.arange(3), weights)
        np.testing.assert_array_equal(positive, presence[expanded].sum(0))
        np.testing.assert_allclose(active, frequency[expanded].mean(0))
        scores = np.array([ACTIVE_EPS / 2, ACTIVE_EPS * 2, ACTIVE_EPS * 3])
        correlation = weighted_union_positive_spearman(scores, scores, np.ones(3, int))
        self.assertAlmostEqual(correlation, 1.0)

    def test_top25_duplicate_instances_match_explicit_sort(self) -> None:
        scores = np.array([0.9, 0.8, 0.9])
        ids = np.array(["b", "c", "a"])
        weights = np.array([2, 1, 3])
        result = top_patient_instances(scores, ids, weights, count=5)
        self.assertEqual(result, [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)])

    def test_weighted_mass_and_energy_match_explicit_copy(self) -> None:
        values = np.array([[1., 2.], [3., 5.], [7., 11.]])
        weights = np.array([2, 0, 3])
        expanded = np.repeat(values, weights, axis=0)
        np.testing.assert_allclose(weighted_sum(values, weights), expanded.sum(0))
        np.testing.assert_allclose(weighted_mean(values, weights), expanded.mean(0))

    def test_weighted_spearman_matches_explicit_copy(self) -> None:
        source = np.array([0., 1., 2., 4.])
        target = np.array([3., 0., 2., 1.])
        weights = np.array([2, 1, 3, 0])
        expanded_source = np.repeat(source, weights)
        expanded_target = np.repeat(target, weights)
        union = (expanded_source > ACTIVE_EPS) | (expanded_target > ACTIVE_EPS)
        expected = spearmanr(expanded_source[union], expanded_target[union]).statistic
        self.assertAlmostEqual(
            weighted_union_positive_spearman(source, target, weights), expected
        )

    def test_weighted_spatial_matches_explicit_copy(self) -> None:
        values = np.array([0.2, 0.7, 0.9])
        valid = np.array([True, False, True])
        weights = np.array([2, 4, 1])
        expanded = np.repeat(values, weights)
        expanded_valid = np.repeat(valid, weights)
        self.assertAlmostEqual(
            weighted_spatial_patient_mean(values, valid, weights),
            expanded[expanded_valid].mean(),
        )

    def test_structural_failure_sets_all_folds_and_metrics_to_zero(self) -> None:
        record = structural_failure_record(7, "null_stratification_infeasible")
        self.assertEqual(record["status"], "replicate_structural_failure")
        for fold in record["fold_metrics"].values():
            self.assertEqual(set(fold), set(METRIC_NAMES))
            self.assertTrue(all(value == 0.0 for value in fold.values()))
        with self.assertRaises(ValueError):
            structural_failure_record(8, "zero_anchor")

    def test_formal_patient_table_requires_frozen_counts(self) -> None:
        ids, labels, sources = toy_patients()
        with self.assertRaises(RuntimeError):
            validate_formal_patient_table(ids, labels, sources)

    def test_worker_assignment_is_complete_and_disjoint(self) -> None:
        groups = [worker_replicate_indices(rank, 4) for rank in range(4)]
        merged = np.concatenate(groups)
        np.testing.assert_array_equal(np.sort(merged), np.arange(BOOTSTRAP_COUNT))
        self.assertEqual(np.unique(merged).size, BOOTSTRAP_COUNT)

    def test_aggregator_lower_quantile_and_index_guards(self) -> None:
        records = []
        values = np.linspace(0.1, 0.9, BOOTSTRAP_COUNT)
        for index, value in enumerate(values):
            records.append({
                "replicate_index": index,
                "status": "completed",
                "fold_metrics": {
                    str(fold): {name: float(value + offset) for name in METRIC_NAMES}
                    for fold, offset in ((42, 0.0), (43, 0.01), (44, 0.02))
                },
            })
        thresholds = aggregate_thresholds(records)
        expected = float(np.quantile(values, 0.05, method="lower"))
        self.assertTrue(all(value == expected for value in thresholds.values()))
        with self.assertRaises(RuntimeError):
            aggregate_thresholds(records[:-1])
        duplicate = records.copy()
        duplicate[-1] = records[-2]
        with self.assertRaises(RuntimeError):
            aggregate_thresholds(duplicate)
        invalid_status = [dict(record) for record in records]
        invalid_status[0] = {**invalid_status[0], "status": "implementation_error"}
        with self.assertRaises(RuntimeError):
            aggregate_thresholds(invalid_status)

    def test_nonpositive_threshold_and_full_train_failure_stop(self) -> None:
        records = [structural_failure_record(index, "null_stratification_infeasible") for index in range(400)]
        with self.assertRaises(RuntimeError):
            aggregate_thresholds(records)
        thresholds = {name: 0.5 for name in METRIC_NAMES}
        folds = {
            str(fold): {name: 0.6 for name in METRIC_NAMES}
            for fold in (42, 43, 44)
        }
        verify_full_train_self_consistency(thresholds, folds)
        folds["44"][METRIC_NAMES[0]] = 0.4
        with self.assertRaises(RuntimeError):
            verify_full_train_self_consistency(thresholds, folds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
