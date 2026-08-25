#!/usr/bin/env python3
"""RP-C2冻结统计与fixed-attention数学测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from clong_rpc2_core import (
    aligned_dose_spearman,
    direction_code,
    fixed_attention_delta_margin,
    intermediate_curve,
    matched_reference_summary,
)
from run_clong_rpc2_preflight_probe import compare_exact_array


class RPC2Tests(unittest.TestCase):
    """覆盖seed顺序、三中间剂量、tail fraction和诊断路径。"""

    def test_direction_uses_three_seed_median(self) -> None:
        expected, count, pattern = direction_code(np.asarray([-2.0, -1.0, 3.0]))
        self.assertEqual((expected, count, pattern), ("negative", 2, "negative;negative;positive"))

    def test_zero_direction(self) -> None:
        expected, count, pattern = direction_code(np.asarray([-1.0, 0.0, 2.0]))
        self.assertEqual((expected, count, pattern), ("zero", 1, "negative;zero;positive"))

    def test_intermediate_curve_excludes_endpoints(self) -> None:
        values = np.asarray([100.0, 1.0, 2.0, 3.0, 200.0])
        self.assertEqual(intermediate_curve(values), 2.0)

    def test_aligned_spearman(self) -> None:
        values = np.asarray([0.0, -1.0, -2.0, -3.0, -4.0])
        self.assertAlmostEqual(aligned_dose_spearman(values, "negative"), 1.0)
        self.assertIsNone(aligned_dose_spearman(values, "zero"))

    def test_matched_reference_ties_and_tail(self) -> None:
        result = matched_reference_summary(2.0, np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(result["matched_midrank_percentile"], 0.5)
        self.assertEqual(result["matched_plus_one_tail_fraction"], 0.75)
        self.assertEqual(result["effect_ratio_to_control_median"], 1.0)

    def test_zero_control_median_has_no_ratio(self) -> None:
        result = matched_reference_summary(1.0, np.asarray([0.0, 0.0, 2.0]))
        self.assertIsNone(result["effect_ratio_to_control_median"])
        self.assertEqual(result["effect_ratio_status"], "control_median_zero")

    def test_quantiles_use_lower(self) -> None:
        result = matched_reference_summary(10.0, np.arange(1.0, 22.0))
        self.assertEqual(result["control_q90_lower"], 19.0)
        self.assertEqual(result["control_q95_lower"], 20.0)

    def test_fixed_attention_matches_direct_formula(self) -> None:
        attention = torch.tensor([[0.25, 0.75]])
        hidden = torch.tensor([[[2.0], [4.0]]])
        decoder = torch.tensor([[3.0, 1.0]])
        margin_weight = torch.tensor([2.0, -1.0])
        result = fixed_attention_delta_margin(
            attention, hidden, decoder, margin_weight, alpha=0.5,
        )
        # d dot w = 5; attention-weighted h = 3.5; scale = -0.5.
        self.assertTrue(torch.allclose(result, torch.tensor([[-8.75]])))

    def test_alpha_zero_exact_array_subset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.arange(12, dtype=np.float32).reshape(3, 4)
            candidate = reference[:, [1, 3]].copy()
            np.save(root / "reference.npy", reference)
            np.save(root / "candidate.npy", candidate)
            compare_exact_array(
                root / "candidate.npy", root / "reference.npy", np.asarray([1, 3]),
            )

    def test_alpha_zero_rejects_one_bit_difference(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.asarray([[1.0]], dtype=np.float32)
            candidate = np.nextafter(reference, np.float32(2.0))
            np.save(root / "reference.npy", reference)
            np.save(root / "candidate.npy", candidate)
            with self.assertRaises(RuntimeError):
                compare_exact_array(root / "candidate.npy", root / "reference.npy", None)


if __name__ == "__main__":
    unittest.main()
