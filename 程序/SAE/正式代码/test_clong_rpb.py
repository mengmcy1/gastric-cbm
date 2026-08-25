#!/usr/bin/env python3
"""RP-B技术规则的最小回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from clong_rpb_core import (
    bh_adjust,
    choose_representative,
    classify_sharedness,
    complete_link_families,
    diagnose_sharedness,
    label_source_bootstrap_contrasts,
)


PROTOCOL = {
    "sharedness": {
        "coverage_equivalence_margin_absolute": 0.10,
        "mass_equivalence_margin_absolute": 0.10,
        "shared_high_minimum_coverage_each_label": 0.25,
        "seed_consensus_minimum": 2,
    }
}


def seed_row(low: float, high: float, mass_low: float, mass_high: float, coverage: float) -> dict:
    """构造sharedness单seed测试行。"""
    return {
        "coverage_ci_low": low,
        "coverage_ci_high": high,
        "mass_ci_low": mass_low,
        "mass_ci_high": mass_high,
        "cancer_coverage": coverage,
        "noncancer_coverage": coverage,
    }


class RPBTests(unittest.TestCase):
    """验证五类判定、BH、complete-link与代表词典序。"""

    def test_bh_is_monotone_in_rank_order(self) -> None:
        p = np.array([0.04, 0.001, 0.02])
        q = bh_adjust(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= 0))

    def test_cancer_enriched_requires_two_seeds_and_no_opposite(self) -> None:
        cancer = seed_row(0.11, 0.20, 0.11, 0.20, 0.4)
        mixed = seed_row(-0.2, 0.2, -0.2, 0.2, 0.4)
        self.assertEqual(classify_sharedness([cancer, cancer, mixed], PROTOCOL), "cancer_enriched")
        opposite = seed_row(-0.20, -0.11, -0.20, -0.11, 0.4)
        self.assertEqual(classify_sharedness([cancer, cancer, opposite], PROTOCOL), "mixed_uncertain")

    def test_shared_high_and_low(self) -> None:
        equivalent_high = seed_row(-0.05, 0.05, -0.05, 0.05, 0.4)
        equivalent_low = seed_row(-0.05, 0.05, -0.05, 0.05, 0.1)
        self.assertEqual(classify_sharedness([equivalent_high] * 3, PROTOCOL), "shared_high")
        self.assertEqual(classify_sharedness([equivalent_low] * 3, PROTOCOL), "shared_low_rare")

    def test_mixed_reason_preserves_frozen_class(self) -> None:
        cancer = seed_row(0.11, 0.20, 0.11, 0.20, 0.4)
        noncancer = seed_row(-0.20, -0.11, -0.20, -0.11, 0.4)
        mixed = seed_row(-0.20, 0.20, -0.20, 0.20, 0.4)
        result = diagnose_sharedness([cancer, noncancer, mixed], PROTOCOL)
        self.assertEqual(result[:2], ("mixed_uncertain", "opposite_enrichment_across_seeds"))

    def test_mixed_reason_separates_coverage_and_mass(self) -> None:
        row = seed_row(-0.05, 0.05, 0.11, 0.20, 0.4)
        result = diagnose_sharedness([row, row, row], PROTOCOL)
        self.assertEqual(result[:2], ("mixed_uncertain", "coverage_equivalent_mass_not_equivalent"))

    def test_label_source_bootstrap_preserves_shapes(self) -> None:
        presence = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float64)
        mass = presence * 0.5
        labels = np.asarray([0, 0, 1, 1])
        sources = np.asarray(["a", "b", "a", "b"])
        result = label_source_bootstrap_contrasts(
            presence, mass, labels, sources, 10, np.random.default_rng(1),
        )
        self.assertEqual(result["coverage_low"].shape, (2,))
        self.assertTrue(all(np.isfinite(values).all() for values in result.values()))

    def test_complete_link_prevents_chain_merge(self) -> None:
        edges = {("a", "b"), ("b", "c")}
        self.assertEqual(complete_link_families(["a", "b", "c"], edges), [["a", "b"], ["c"]])

    def test_no_family_edge_retains_every_singleton(self) -> None:
        self.assertEqual(complete_link_families(["b", "a"], set()), [["a"], ["b"]])

    def test_representative_uses_frozen_lexicographic_order(self) -> None:
        rows = [
            {"anchor_id": "a2", "stability_worst_edge_score": 0.9, "overall_patient_coverage": 0.8, "energy_percentile": 0.7},
            {"anchor_id": "a1", "stability_worst_edge_score": 0.9, "overall_patient_coverage": 0.8, "energy_percentile": 0.7},
            {"anchor_id": "a3", "stability_worst_edge_score": 0.8, "overall_patient_coverage": 1.0, "energy_percentile": 1.0},
        ]
        self.assertEqual(choose_representative(rows), "a1")


if __name__ == "__main__":
    unittest.main()
