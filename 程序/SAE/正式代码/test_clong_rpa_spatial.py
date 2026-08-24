#!/usr/bin/env python3
"""RP-A spatial候选协议的最小语义回归测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from clong_rpa_spatial import apply_minimum_support, image_active, spatial_pair_matrix


PROTOCOL = Path(__file__).with_name("rpa_spatial_protocol_v1.json")


class SpatialTests(unittest.TestCase):
    """验证presence、NA/0、患者平衡和支持度语义。"""

    def test_protocol_frozen_scope(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["status"], "frozen_2026-08-24")
        self.assertEqual(protocol["minimum_support"]["train_patients"], 25)
        self.assertEqual(protocol["minimum_support"]["val_patients"], 6)
        self.assertFalse(protocol["numeric_candidate"]["tf32"])
        self.assertTrue(
            protocol["numerical_audit"]["accepted_result"]["finite_or_na_semantics_match"]
        )

    def test_presence_uses_max_not_l2(self) -> None:
        values = torch.full((1, 49, 1), 0.5e-8, dtype=torch.float32)
        self.assertGreater(float(torch.linalg.vector_norm(values[:, :, 0])), 1e-8)
        self.assertFalse(bool(image_active(values)[0, 0]))

    def test_image_semantics_and_patient_balance(self) -> None:
        source = torch.zeros((3, 49, 1), dtype=torch.float32)
        target = torch.zeros_like(source)
        source[0, 0, 0] = 1.0
        target[0, 0, 0] = 1.0
        source[1, 0, 0] = 1.0
        target[2, 1, 0] = 1.0
        values, diagnostic = spatial_pair_matrix(
            source, target, torch.tensor([0, 0, 1]), 2, torch.float64
        )
        self.assertAlmostEqual(float(values[0, 0]), 0.25)
        self.assertEqual(int(diagnostic["n_evaluable_patient_instances"][0, 0]), 2)
        self.assertEqual(float(diagnostic["n_union_active_image_instances"][0, 0]), 3.0)
        self.assertAlmostEqual(float(diagnostic["fraction_one_side_zero"][0, 0]), 2 / 3)
        self.assertAlmostEqual(
            float(diagnostic["fraction_one_side_zero"][0, 0]
                  + diagnostic["fraction_both_active"][0, 0]), 1.0
        )

    def test_both_inactive_is_na(self) -> None:
        source = torch.zeros((2, 49, 1), dtype=torch.float32)
        values, diagnostic = spatial_pair_matrix(
            source, source.clone(), torch.tensor([0, 1]), 2, torch.float64
        )
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertEqual(int(diagnostic["n_evaluable_patient_instances"][0, 0]), 0)

    def test_zero_with_support_is_valid(self) -> None:
        source = torch.zeros((2, 49, 1), dtype=torch.float32)
        target = torch.zeros_like(source)
        source[:, 0, 0] = 1.0
        target[:, 1, 0] = 1.0
        values, diagnostic = spatial_pair_matrix(
            source, target, torch.tensor([0, 1]), 2, torch.float64
        )
        filtered, valid = apply_minimum_support(
            values, diagnostic["n_evaluable_patient_instances"], 2
        )
        self.assertTrue(bool(valid[0, 0]))
        self.assertEqual(float(filtered[0, 0]), 0.0)

    def test_below_support_is_na(self) -> None:
        values = np.array([[0.7, 0.0]])
        filtered, valid = apply_minimum_support(values, np.array([[5, 6]]), 6)
        self.assertTrue(np.isnan(filtered[0, 0]))
        self.assertFalse(bool(valid[0, 0]))
        self.assertEqual(float(filtered[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
