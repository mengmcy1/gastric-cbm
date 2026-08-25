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
from run_clong_rpc2_formal import (
    CONTROL_PATH,
    STUDY_PATH,
    build_anchor_master,
    prepare_output_root,
    unique_feature_universe,
)


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

    def test_unique_universe_deduplicates_shared_controls(self) -> None:
        study = np.asarray([10, 20])
        study_frame = __import__("pandas").DataFrame({"feature_42": study})
        controls = __import__("pandas").DataFrame({
            "sae_seed": [42, 42, 42],
            "control_feature_id": [30, 30, 10],
        })
        result = unique_feature_universe(study_frame, controls, 42)
        np.testing.assert_array_equal(result, np.asarray([10, 20, 30]))

    def test_frozen_mapping_counts_and_roles(self) -> None:
        pd = __import__("pandas")
        study = pd.read_csv(STUDY_PATH, dtype={"anchor_id": str})
        controls = pd.read_csv(CONTROL_PATH, dtype={"anchor_id": str})
        self.assertEqual(len(study), 149)
        self.assertEqual(len(controls), 44424)
        self.assertEqual(int(study.primary_candidate.sum()), 122)
        self.assertEqual(int(study.low_effect_control.sum()), 20)
        self.assertEqual(int(study.post_rpc1_secondary_exploratory.sum()), 8)
        overlap = study.loc[
            study.primary_candidate & study.low_effect_control, "anchor_id"
        ].tolist()
        self.assertEqual(overlap, ["a00987"])
        distribution = controls.groupby(["anchor_id", "sae_seed"]).size().value_counts().to_dict()
        self.assertEqual(distribution, {100: 442, 78: 1, 66: 1, 32: 1, 27: 1, 21: 1})
        for seed in (42, 43, 44):
            universe = set(unique_feature_universe(study, controls, seed))
            self.assertTrue(set(study[f"feature_{seed}"].astype(int)).issubset(universe))

    def test_anchor_summary_does_not_combine_tail_fraction(self) -> None:
        pd = __import__("pandas")
        rows = []
        for anchor_index in range(149):
            for seed, percentile in zip((42, 43, 44), (0.7, 0.8, 0.9)):
                rows.append({
                    "anchor_id": f"a{anchor_index:05d}", "seed": seed,
                    "target_feature_id": seed + anchor_index,
                    "primary_candidate": True, "low_effect_control": False,
                    "post_rpc1_secondary_exploratory": False,
                    "functional_pattern": "mixed_functional_pattern",
                    "intermediate_curve_overall_effect": 0.1,
                    "matched_midrank_percentile": percentile,
                    "matched_plus_one_tail_fraction": 0.2,
                    "effect_ratio_to_control_median": 1.5, "n_control": 100,
                })
        master = build_anchor_master(pd.DataFrame(rows))
        first = master.loc[master.anchor_id.eq("a00000")].iloc[0]
        self.assertEqual(first["median_matched_midrank_percentile"], 0.8)
        self.assertEqual(first["minimum_matched_midrank_percentile"], 0.7)
        self.assertFalse(any("combined" in column for column in master.columns))

    def test_formal_runner_has_no_rematching_implementation(self) -> None:
        source = Path(__import__("run_clong_rpc2_formal").__file__).read_text(encoding="utf-8")
        self.assertNotIn("select_nearest_controls", source)
        self.assertNotIn("clong_rpc2_matching_core", source)
        self.assertNotIn("conditional_empirical_p_value", source)

    def test_existing_output_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "formal"
            path.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_output_root(path)


if __name__ == "__main__":
    unittest.main()
