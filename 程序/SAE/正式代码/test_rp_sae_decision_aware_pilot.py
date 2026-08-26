#!/usr/bin/env python3
"""Decision-aware pilot核心公式测试。"""

import unittest

import numpy as np

from build_rp_sae_decision_aware_pilot import (
    bbox_metrics,
    fixed_attention_local_ablation_map,
    grid_edge_mass_ratio,
    select_features,
)


class DecisionAwarePilotTests(unittest.TestCase):
    def test_fixed_attention_map_sum_matches_margin_formula(self) -> None:
        h = np.arange(1, 50, dtype=float).reshape(7, 7)
        attention = np.full((7, 7), 1 / 49)
        direction = -0.25
        result = fixed_attention_local_ablation_map(h, attention, direction)
        self.assertAlmostEqual(float(result.sum()), -float((attention * h).sum()) * direction)

    def test_grid_edge_mass_ratio(self) -> None:
        values = np.zeros((7, 7), dtype=float)
        values[0, 0] = 1
        values[3, 3] = 1
        self.assertEqual(grid_edge_mass_ratio(values), 0.5)

    def test_bbox_signed_and_absolute_metrics_are_separate(self) -> None:
        raw = np.ones((7, 7), dtype=float)
        gated = np.ones((7, 7), dtype=float)
        direct = np.zeros((7, 7), dtype=float)
        direct[0, 0] = 2
        direct[0, 1] = -1
        result = bbox_metrics(raw, gated, direct, np.asarray([0, 0, 2 / 7, 1 / 7]))
        self.assertAlmostEqual(result["direct_bbox_signed_sum"], 1.0)
        self.assertAlmostEqual(result["direct_bbox_absolute_mass_fraction"], 1.0)

    def test_selection_uses_distinct_raw_and_actual_effect_rankings(self) -> None:
        raw = np.asarray([9, 8, 7, 6, 5, 4, 3], dtype=float)
        delta = np.asarray([0, 0, 0, 0, 0, 0, 10], dtype=float)
        anchors = __import__("pandas").DataFrame(
            {"anchor_id": [f"a{i:05d}" for i in range(7)]}
        )
        raw_top, intervention_top, union = select_features(raw, delta, anchors)
        self.assertNotIn(6, raw_top)
        self.assertEqual(int(intervention_top[0]), 6)
        self.assertEqual(len(union), 7)


if __name__ == "__main__":
    unittest.main()
