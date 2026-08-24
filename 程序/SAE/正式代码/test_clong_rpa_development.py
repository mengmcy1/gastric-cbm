#!/usr/bin/env python3
"""RP-A seed43/44训练入口和开发图结构的回归测试。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpa_development_core import (
    all_pseudo_fold_metrics,
    pseudo_confirmation_metrics,
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
from clong_rpa_match_development import cross_spatial_matrix
from clong_rpa_bootstrap_worker import (
    bootstrap_plans,
    expanded_image_instances,
    expanded_patient_instances,
)
from clong_rpa_validate_development import (
    FORMAL_VAL_SPATIAL_SUPPORT,
    FORMAL_VAL_TOP_COUNT,
    minimum_fold_metrics,
    pair_edges_from_csv,
)
from clong_rpa_provenance import CODE_FILES, build_snapshot, validate_snapshot
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

    def test_val_uses_frozen_train_reference_edges(self) -> None:
        train_references = dict(self.edges)
        train_references[(43, 44)] = [(1, 4), (3, 5)]
        val_reproduction = dict(self.edges)
        val_reproduction[(42, 43)] = [(0, 1)]
        val_reproduction[(42, 44)] = [(0, 4)]
        val_reproduction[(43, 44)] = []
        metrics = pseudo_confirmation_metrics(
            42, val_reproduction, self.eligible, self.mass, self.energy,
            reference_pair_edges=train_references,
        )
        self.assertAlmostEqual(metrics["R_anchor_recall"], 0.5)


class BootstrapWorkerTests(unittest.TestCase):
    """验证患者有放回抽样在患者与图像层的实例语义。"""

    def setUp(self) -> None:
        self.patients = pd.DataFrame({
            "patient_id": ["p1", "p2"],
            "label": [0, 1],
            "source": ["a", "b"],
            "image_count": [2, 1],
        })
        self.images = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2"],
            "relative_path": ["a.jpg", "b.jpg", "c.jpg"],
        })

    def test_duplicate_patient_instances_are_distinct(self) -> None:
        indices, expanded = expanded_patient_instances(
            self.patients, np.asarray([2, 0], dtype=np.int32)
        )
        np.testing.assert_array_equal(indices, [0, 0])
        self.assertEqual(expanded.patient_id.tolist(), ["p1#occ0", "p1#occ1"])

    def test_each_occurrence_carries_all_patient_images(self) -> None:
        indices, expanded = expanded_image_instances(
            self.images, self.patients, np.asarray([2, 0], dtype=np.int32)
        )
        np.testing.assert_array_equal(indices, [0, 1, 0, 1])
        self.assertEqual(
            expanded.patient_id.tolist(),
            ["p1#occ0", "p1#occ0", "p1#occ1", "p1#occ1"],
        )

    def test_debug_bootstrap_plan_is_deterministic(self) -> None:
        patients = pd.concat([self.patients, self.patients], ignore_index=True)
        args = argparse.Namespace(debug=True, debug_replicates=2)
        plans = bootstrap_plans(patients, args)
        np.testing.assert_array_equal(plans[0], [1, 1, 1, 1])
        np.testing.assert_array_equal(plans[1], [2, 0, 1, 1])


class ValidationTests(unittest.TestCase):
    """验证val固定人数、最弱折归约和合法空edge输入。"""

    def test_val_uses_frozen_six_patient_rules(self) -> None:
        self.assertEqual(FORMAL_VAL_TOP_COUNT, 6)
        self.assertEqual(FORMAL_VAL_SPATIAL_SUPPORT, 6)

    def test_minimum_fold_metrics_uses_weakest_fold(self) -> None:
        folds = {
            str(seed): {
                name: 0.4 + 0.1 * index
                for name in ("R_anchor_recall", "R_confirm_coverage",
                             "R_activation_reference", "R_activation_confirmation",
                             "R_energy_reference", "R_energy_confirmation")
            }
            for index, seed in enumerate((42, 43, 44))
        }
        self.assertTrue(all(value == 0.4 for value in minimum_fold_metrics(folds).values()))

    def test_empty_train_edge_file_is_legal_zero_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.csv"
            path.write_text("\n", encoding="utf-8")
            self.assertEqual(
                pair_edges_from_csv(path),
                {(42, 43): [], (42, 44): [], (43, 44): []},
            )

    def test_spatial_target_blocking_is_numerically_invariant(self) -> None:
        generator = torch.Generator().manual_seed(42)
        source = torch.rand((3, 4, 5), generator=generator)
        target = torch.rand((3, 4, 6), generator=generator)
        source = source / torch.linalg.vector_norm(source, dim=1, keepdim=True)
        target = target / torch.linalg.vector_norm(target, dim=1, keepdim=True)
        source_active = torch.ones((3, 5), dtype=torch.bool)
        target_active = torch.ones((3, 6), dtype=torch.bool)
        patient_index = torch.tensor([0, 0, 1])
        arguments = (
            source, source_active, target, target_active, patient_index, 2,
            np.asarray([0, 2, 4]), np.asarray([0, 1, 3, 5]), 1,
        )
        blocked = cross_spatial_matrix(*arguments, target_block=1)
        single = cross_spatial_matrix(*arguments, target_block=4)
        np.testing.assert_allclose(blocked, single, rtol=0, atol=0)

    def test_code_snapshot_covers_runner_and_validates_current_tree(self) -> None:
        snapshot = build_snapshot()
        self.assertEqual(set(snapshot["code_file_sha256"]), set(CODE_FILES))
        for name in (
            "run_clong_rpa_development.sh", "clong_rpa_finalize_development.py",
            "clong_s2c_core.py", "clong_s2c_matryoshka.py",
            "clong_sae_discovery.py", "clong_s2b_discovery.py",
            "clong_s2b_core.py", "train_utils.py",
        ):
            self.assertIn(name, snapshot["code_file_sha256"])
        validate_snapshot(snapshot)

    def test_changed_code_sha_is_rejected(self) -> None:
        snapshot = build_snapshot()
        name = CODE_FILES[0]
        snapshot["code_file_sha256"][name] = "0" * 64
        with self.assertRaises(RuntimeError):
            validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
