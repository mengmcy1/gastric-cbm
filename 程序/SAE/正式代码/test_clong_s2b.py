#!/usr/bin/env python3
"""C-long S2b 核心协议的轻量回归测试。"""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from clong_s2b_core import (
    StructuredSparseAutoencoder,
    attention_drift_metrics,
    attention_from_features,
    patch_position_weights,
    pooled_from_features,
    solve_threshold_from_chunks,
    spatial_alignment_metrics,
)
from clong_s2b_discovery import formal_name, normalized_features, validate_args
from summarize_clong_s2b import (
    CLONG_CHECKPOINT_SHA256,
    FORMAL_BUDGET,
    MANIFEST_SHA256,
    common_gates,
    load_outcome,
    patch_gates,
    patch_selection_key,
)
from build_mage_teacher_roi_manifest import file_sha256


class S2bCoreTests(unittest.TestCase):
    def make_sae(self, mode: str, k: int = 2) -> StructuredSparseAutoencoder:
        torch.manual_seed(1)
        return StructuredSparseAutoencoder(4, 6, torch.zeros(4), mode, k)

    def test_topk_budget_is_per_vector(self) -> None:
        model = self.make_sae("topk")
        dense = torch.arange(1, 73, dtype=torch.float32).reshape(2, 6, 6)
        sparse = model.sparsify(dense)
        self.assertTrue(torch.equal((sparse > 0).sum(-1), torch.full((2, 6), 2)))

    def test_batchtopk_budget_is_global(self) -> None:
        model = self.make_sae("batch_topk")
        dense = torch.arange(1, 73, dtype=torch.float32).reshape(2, 6, 6)
        sparse = model.sparsify(dense)
        self.assertEqual(int((sparse > 0).sum()), 24)
        self.assertNotEqual(len(torch.unique((sparse > 0).sum(-1))), 1)

    def test_threshold_solver_exact_without_ties(self) -> None:
        chunks = [np.array([1, 7, 3, 9], np.float32), np.array([2, 8, 4, 6], np.float32)]
        result = solve_threshold_from_chunks(lambda: iter(chunks), vector_count=4, target_k=1)
        self.assertEqual(result.threshold, 6.0)
        self.assertEqual(result.actual_total, 4)
        self.assertEqual(result.actual_mean_l0, 1.0)

    def test_threshold_solver_rejects_tie_deviation(self) -> None:
        chunks = [np.array([1, 2, 2, 2], np.float32)]
        with self.assertRaisesRegex(RuntimeError, "\u5e76\u5217"):
            solve_threshold_from_chunks(lambda: iter(chunks), vector_count=2, target_k=1)

    def test_threshold_solver_rejects_insufficient_positive(self) -> None:
        chunks = [np.array([0, 0, 1], np.float32)]
        with self.assertRaisesRegex(RuntimeError, "\u6b63\u6fc0\u6d3b\u603b\u6570"):
            solve_threshold_from_chunks(lambda: iter(chunks), vector_count=2, target_k=1)

    def test_position_weight_mean_is_one(self) -> None:
        attention = torch.softmax(torch.randn(3, 49), 1)
        weights = patch_position_weights(attention)
        self.assertTrue(torch.allclose(weights.mean(1), torch.ones(3), atol=1e-6))

    def test_attention_and_pool_shapes(self) -> None:
        features = torch.randn(2, 49, 4)
        head = nn.Conv2d(4, 1, 1)
        attention = attention_from_features(features, head)
        pooled = pooled_from_features(features, attention)
        self.assertEqual(tuple(attention.shape), (2, 49))
        self.assertEqual(tuple(pooled.shape), (2, 4))
        self.assertTrue(torch.allclose(attention.sum(1), torch.ones(2)))

    def test_reconstructed_attention_is_recomputed(self) -> None:
        head = nn.Conv2d(4, 1, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(1)
        first = torch.zeros(1, 49, 4)
        second = first.clone()
        second[:, 10] = 5
        a = attention_from_features(first, head)
        b = attention_from_features(second, head)
        self.assertFalse(torch.allclose(a, b))
        self.assertEqual(int(b.argmax(1)), 10)

    def test_spatial_metrics_match_simple_box(self) -> None:
        attention = np.zeros((1, 49), dtype=np.float32)
        attention[0, 0] = 1
        labels = np.array([1])
        boxes = np.array([[0, 0, 1 / 7, 1 / 7]], dtype=np.float32)
        metrics = spatial_alignment_metrics(attention, labels, boxes)
        self.assertAlmostEqual(metrics["mean_aib"], 1.0, places=6)
        self.assertAlmostEqual(metrics["pga"], 1.0, places=6)

    def test_attention_drift_identity(self) -> None:
        values = np.full((2, 49), 1 / 49, dtype=np.float32)
        metrics = attention_drift_metrics(values, values)
        self.assertAlmostEqual(metrics["mean_kl_original_to_reconstructed"], 0.0)
        self.assertAlmostEqual(metrics["mean_cosine"], 1.0)

    def test_normalization_round_trip(self) -> None:
        values = np.random.default_rng(3).normal(size=(5, 8)).astype(np.float32)
        direction, mean, norm = normalized_features(values)
        restored = direction * norm[:, None] + mean[:, None]
        self.assertTrue(np.allclose(restored, values, atol=1e-6))

    def test_formal_name_and_budget_lock(self) -> None:
        args = SimpleNamespace(
            arm="B", seed=42, experiment=formal_name("B", 42), debug=False,
            device="cuda", postprocess_existing=False,
            learning_rate=1e-4, epochs=1000, patience=50,
            warmup_fraction=0.05, image_batch_size=32,
        )
        validate_args(args)
        args.epochs = 999
        with self.assertRaisesRegex(ValueError, "epochs"):
            validate_args(args)

    def test_postprocess_existing_only_allows_batchtopk_arms(self) -> None:
        args = SimpleNamespace(
            arm="C", seed=42, experiment=formal_name("C", 42), debug=False,
            device="cuda", postprocess_existing=True,
            learning_rate=1e-4, epochs=1000, patience=50,
            warmup_fraction=0.05, image_batch_size=32,
        )
        with self.assertRaisesRegex(ValueError, "B/D"):
            validate_args(args)

    def test_protocol_failure_is_a_valid_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / formal_name("B", 42)
            run.mkdir()
            checkpoint = run / "sae_best.pth"
            checkpoint.write_bytes(b"checkpoint")
            (run / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")
            failure = {
                "stage": "S2b", "arm": "B", "seed": 42,
                "experiment": formal_name("B", 42), "debug": False,
                "failure_stage": "freeze_batchtopk_train_threshold",
                "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
                "manifest_sha256": MANIFEST_SHA256,
                "checkpoint_sha256": file_sha256(checkpoint),
                "training": dict(FORMAL_BUDGET),
                "test_evaluated": False, "internal_test_evaluated": False,
                "external_evaluated": False,
            }
            (run / "protocol_failure.json").write_text(
                json.dumps(failure), encoding="utf-8"
            )
            outcome = load_outcome(root, "B", 42)
            self.assertEqual(outcome["status"], "protocol_failure")

    def test_debug_output_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                arm="C", seed=42, experiment="smoke", debug=True,
                debug_patients_per_class=2, output_root=Path(directory),
            )
            validate_args(args)
            self.assertIn("debug", str(args.output_root))

    @staticmethod
    def eligible_config(arm: str = "C") -> dict:
        return {
            "arm": arm,
            "evaluation": {
                "fidelity": {
                    "patient_auc_drop": 0.005,
                    "patient_prediction_agreement_at_locked_threshold": 0.97,
                    "mean_cosine": 0.93,
                    "recovered_cross_entropy": 0.98,
                },
                "dead_feature_rate": 0.02,
                "duplicate": {"duplicate_rate": 0.03},
                "sparsity": {
                    "normalized_aib_drop": 0.02, "pga_drop": 0.03,
                    "mean_l0_per_position": 128.0,
                    "mean_unique_features_per_image": 500.0,
                },
            },
        }

    def test_common_and_patch_gates(self) -> None:
        config = self.eligible_config()
        self.assertTrue(all(common_gates(config).values()))
        self.assertTrue(all(patch_gates(config).values()))
        config["evaluation"]["sparsity"]["pga_drop"] = 0.051
        self.assertFalse(patch_gates(config)["pga_drop_le_0_05"])

    def test_patch_selection_prefers_lower_auc_drop(self) -> None:
        c = self.eligible_config("C")
        d = self.eligible_config("D")
        d["evaluation"]["fidelity"]["patient_auc_drop"] = 0.001
        self.assertLess(patch_selection_key(d), patch_selection_key(c))

    def test_patch_selection_final_tie_prefers_c(self) -> None:
        c = self.eligible_config("C")
        d = self.eligible_config("D")
        self.assertLess(patch_selection_key(c), patch_selection_key(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
