#!/usr/bin/env python3
"""RP-A seed43/44训练入口和开发图结构的回归测试。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from clong_rpa_development_core import (
    all_pseudo_fold_metrics,
    strict_three_cliques,
)
from clong_rpa_train_development import (
    FORMAL_SEEDS,
    REPRESENTATION_K,
    experiment_name,
    frozen_gamma,
    validate_args,
)
from clong_rpa_prepare_seed import load_sae
from clong_s2c_core import MatryoshkaSparseAutoencoder


class DevelopmentTrainingTests(unittest.TestCase):
    """验证seed角色、冻结gamma与正式设备边界。"""

    def test_only_seed43_and_44_are_development(self) -> None:
        self.assertEqual(FORMAL_SEEDS, (43, 44))
        self.assertEqual(REPRESENTATION_K, 1024)
        for seed in FORMAL_SEEDS:
            validate_args(argparse.Namespace(seed=seed, device="cuda", debug=False,
                                             debug_epochs=3, debug_patients_per_class=2))
            self.assertIn(str(seed), experiment_name(seed))

    def test_confirmation_or_existing_seed_is_rejected(self) -> None:
        for seed in (42, 202, 503, 911):
            with self.assertRaises(ValueError):
                validate_args(argparse.Namespace(seed=seed, device="cuda", debug=False,
                                                 debug_epochs=3, debug_patients_per_class=2))

    def test_debug_allows_three_development_seeds_only(self) -> None:
        for seed in (42, 43, 44):
            validate_args(argparse.Namespace(seed=seed, device="cpu", debug=True,
                                             debug_epochs=1, debug_patients_per_class=2))
        with self.assertRaises(ValueError):
            validate_args(argparse.Namespace(seed=202, device="cpu", debug=True,
                                             debug_epochs=1, debug_patients_per_class=2))

    def test_formal_cpu_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_args(argparse.Namespace(seed=43, device="cpu", debug=False,
                                             debug_epochs=3, debug_patients_per_class=2))

    def test_debug_rejects_too_few_patients_for_strata(self) -> None:
        with self.assertRaises(ValueError):
            validate_args(argparse.Namespace(seed=43, device="cpu", debug=True,
                                             debug_epochs=1, debug_patients_per_class=1))

    def test_gamma_is_inherited_from_seed42(self) -> None:
        gamma, record = frozen_gamma()
        self.assertAlmostEqual(gamma, 0.5095280077324069, places=15)
        self.assertEqual(record["seed"], 42)

    def test_debug_checkpoint_restores_standard_encoder_keys(self) -> None:
        source = MatryoshkaSparseAutoencoder(
            input_dim=8, hidden_dim=16, feature_center=torch.zeros(8),
            k_list=(2, 4),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"sae_state_dict": source.state_dict(), "k_list": [2, 4]}, path)
            model = load_sae(path, torch.device("cpu"))
            self.assertEqual(model.hidden_dim, 16)
            self.assertEqual(model.input_dim, 8)
            self.assertEqual(max(model.k_list), 4)


class DevelopmentGraphTests(unittest.TestCase):
    """验证严格三角形和逐reference seed覆盖率。"""

    def setUp(self) -> None:
        self.edges = {
            (42, 43): [(0, 1), (2, 3)],
            (42, 44): [(0, 4), (2, 5)],
            (43, 44): [(1, 4)],
        }
        self.eligible = {
            42: np.array([0, 2]), 43: np.array([1, 3]), 44: np.array([4, 5]),
        }
        self.mass = {
            42: np.array([2., 0., 8., 0., 0., 0.]),
            43: np.array([3., 2., 0., 8., 0., 0.]),
            44: np.array([0., 0., 0., 0., 4., 6.]),
        }
        self.energy = {seed: values * 2 for seed, values in self.mass.items()}

    def test_only_complete_triangle_becomes_anchor(self) -> None:
        anchors = strict_three_cliques(self.edges)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(
            (anchors[0]["feature_42"], anchors[0]["feature_43"], anchors[0]["feature_44"]),
            (0, 1, 4),
        )

    def test_three_folds_emit_exact_six_metrics(self) -> None:
        metrics = all_pseudo_fold_metrics(
            self.edges, self.eligible, self.mass, self.energy
        )
        self.assertEqual(set(metrics), {"42", "43", "44"})
        self.assertTrue(all(len(values) == 6 for values in metrics.values()))
        self.assertAlmostEqual(metrics["44"]["R_anchor_recall"], 0.5)
        self.assertAlmostEqual(metrics["44"]["R_confirm_coverage"], 0.5)

    def test_reference_mass_ratios_are_seedwise_then_averaged(self) -> None:
        metrics = all_pseudo_fold_metrics(
            self.edges, self.eligible, self.mass, self.energy
        )
        expected = 0.5 * (2 / 10 + 2 / 10)
        self.assertAlmostEqual(metrics["44"]["R_activation_reference"], expected)

    def test_non_one_to_one_pair_is_rejected(self) -> None:
        broken = dict(self.edges)
        broken[(42, 43)] = [(0, 1), (2, 1)]
        with self.assertRaises(ValueError):
            strict_three_cliques(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
