#!/usr/bin/env python3
"""RP-C1数学、聚合和分轨回归测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from clong_rpc_core import (
    aggregate_active_by_patient,
    aggregate_image_matrix_by_patient,
    assign_rpc2_tracks,
    intervention_block,
    label_seed_metrics,
)


class RPCTests(unittest.TestCase):
    """覆盖no-op、符号、患者平衡和固定分轨。"""

    def test_alpha_one_is_exact_noop(self) -> None:
        torch.manual_seed(1)
        result = intervention_block(
            torch.randn(2, 49, 4), torch.rand(2, 49, 3), torch.randn(3, 4),
            torch.randn(4), torch.tensor(0.2), torch.randn(4), torch.tensor(-0.1), 1.0,
        )
        self.assertLessEqual(float(result["delta_margin"].abs().max()), 1e-6)
        self.assertLessEqual(float(result["delta_probability"].abs().max()), 1e-6)
        self.assertTrue(torch.allclose(result["attention_cosine"], torch.ones_like(result["attention_cosine"]), atol=1e-6))
        self.assertLessEqual(float(result["attention_l1"].max()), 1e-6)

    def test_zero_hidden_has_zero_effect(self) -> None:
        result = intervention_block(
            torch.randn(1, 49, 2), torch.zeros(1, 49, 1), torch.randn(1, 2),
            torch.randn(2), torch.tensor(0.0), torch.randn(2), torch.tensor(0.0), 0.0,
        )
        self.assertLessEqual(float(result["delta_margin"].abs().max()), 1e-6)
        self.assertFalse(bool(result["active_image"].any()))

    def test_block_formula_matches_direct_feature_intervention(self) -> None:
        torch.manual_seed(3)
        spatial = torch.randn(2, 49, 4)
        hidden = torch.rand(2, 49, 2)
        decoder = torch.randn(2, 4)
        attention_weight = torch.randn(4)
        attention_bias = torch.tensor(0.1)
        margin_weight = torch.randn(4)
        margin_bias = torch.tensor(-0.2)
        result = intervention_block(
            spatial, hidden, decoder, attention_weight, attention_bias,
            margin_weight, margin_bias, 0.0,
        )
        direct = []
        for feature in range(2):
            changed = spatial - hidden[:, :, feature, None] * decoder[feature]
            attention = torch.softmax(changed @ attention_weight + attention_bias, dim=1)
            margin = (attention * (changed @ margin_weight)).sum(1) + margin_bias
            direct.append(margin)
        direct_margin = torch.stack(direct, dim=1)
        expected_delta = direct_margin - result["original_margin"][:, None]
        self.assertTrue(torch.allclose(result["delta_margin"], expected_delta, atol=1e-6, rtol=1e-6))

    def test_patient_aggregation_equal_weights_patients(self) -> None:
        values = np.asarray([[1.0], [3.0], [10.0]], dtype=np.float32)
        output = aggregate_image_matrix_by_patient(
            values, np.asarray(["a", "a", "b"]), np.asarray(["a", "b"]),
        )
        np.testing.assert_allclose(output[:, 0], [2.0, 10.0])

    def test_active_patient_uses_any_image(self) -> None:
        active = np.asarray([[False, True], [True, False], [False, False]])
        output = aggregate_active_by_patient(
            active, np.asarray(["a", "a", "b"]), np.asarray(["a", "b"]),
        )
        np.testing.assert_array_equal(output, [[True, True], [False, False]])

    def test_label_support_sign(self) -> None:
        delta = np.asarray([[-1.0], [1.0]], dtype=np.float32)
        rows = label_seed_metrics(
            delta, np.zeros_like(delta), np.ones_like(delta), np.zeros_like(delta),
            np.zeros_like(delta), np.ones_like(delta, dtype=bool), np.asarray([1, 0]),
            np.asarray([0.8, 0.2]), 0.5,
        )
        self.assertEqual(rows[0]["cancer_mean_label_support"], 1.0)
        self.assertEqual(rows[0]["noncancer_mean_label_support"], 1.0)
        self.assertEqual(rows[0]["class_separation_delta_D"], 2.0)

    def test_rpc2_tracks_have_frozen_sizes(self) -> None:
        frame = pd.DataFrame({
            "anchor_id": [f"a{i:05d}" for i in range(1150)],
            "overall_abs_effect": np.arange(1150, dtype=float),
            "class_separation_effect": np.arange(1150, dtype=float)[::-1],
            "source_risk": [i < 198 for i in range(1150)],
            "sharedness_source_sensitivity_changed": [i < 10 for i in range(1150)],
            "sharedness_v1": ["cancer_enriched" if i == 100 else "mixed_uncertain" for i in range(1150)],
        })
        result = assign_rpc2_tracks(frame)
        self.assertEqual(int(result.rpc2_track_overall.sum()), 58)
        self.assertEqual(int(result.rpc2_track_separation.sum()), 58)
        self.assertEqual(int(result.rpc2_track_source.sum()), 50)
        self.assertEqual(int(result.rpc2_track_sensitivity.sum()), 10)
        self.assertEqual(int(result.rpc2_track_cancer_enriched.sum()), 1)
        self.assertEqual(int(result.low_effect_control.sum()), 20)


if __name__ == "__main__":
    unittest.main()
