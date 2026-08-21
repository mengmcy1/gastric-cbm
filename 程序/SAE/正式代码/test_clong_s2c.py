#!/usr/bin/env python3
"""C-long S2c Matryoshka patch SAE 单元测试。

覆盖冻结协议的关键实现约束：单排序嵌套、不补零、五层并集死亡口径、
gamma_pool校准公式与防线、最小合格K选择、正式参数锁死和debug隔离。
运行：

    /home/mcy/miniconda3/envs/gastric-cbm/bin/python 程序/SAE/正式代码/test_clong_s2c.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from clong_s2c_core import (  # noqa: E402
    K_LIST,
    MatryoshkaSparseAutoencoder,
    SingleKView,
    fvu_from_sums,
    gamma_from_medians,
    select_min_passing_k,
    union_nondead_mask,
)
from clong_s2c_matryoshka import (  # noqa: E402
    CLONG_CHECKPOINT_SHA256,
    FORMAL_BUDGET,
    GAMMA_PROTOCOL,
    MANIFEST_SHA256,
    OUTPUT_ROOT,
    calibrate_gamma,
    formal_name,
    frozen_variance_reference,
    joint_layer_losses,
    joint_total,
    l0_profile,
    load_gamma,
    replication_contract,
    value_strata,
    validate_args,
    variance_diagnostics,
)
import clong_s2c_matryoshka as s2c_module  # noqa: E402


def tiny_sae(input_dim=8, hidden_dim=16, k_list=(2, 4, 8), seed=0):
    torch.manual_seed(seed)
    return MatryoshkaSparseAutoencoder(
        input_dim, hidden_dim, torch.zeros(input_dim), k_list
    )


def tiny_model(input_dim=8):
    return SimpleNamespace(attention_head=torch.nn.Conv2d(input_dim, 1, 1))


class TestNestedTopK(unittest.TestCase):
    def test_nested_subset_property(self):
        sae = tiny_sae()
        hidden = sae.encode(torch.randn(5, 49, 8))
        for small, large in zip(sae.k_list, sae.k_list[1:]):
            self.assertTrue(bool(((hidden[small] > 0) <= (hidden[large] > 0)).all()))

    def test_single_sort_matches_independent_topk(self):
        sae = tiny_sae()
        features = torch.randn(4, 49, 8)
        dense = sae.preactivation(features)
        hidden = sae.encode(features)
        for k in sae.k_list:
            values, indices = dense.reshape(-1, 16).topk(k, dim=1)
            expected = torch.zeros_like(dense.reshape(-1, 16)).scatter_(1, indices, values)
            self.assertTrue(torch.allclose(hidden[k].reshape(-1, 16), expected))

    def test_no_zero_fill_when_positive_short(self):
        sae = tiny_sae()
        with torch.no_grad():
            sae.encoder.weight.zero_()
            sae.encoder.bias.fill_(-1.0)
        hidden = sae.encode(torch.randn(3, 49, 8))
        for k in sae.k_list:
            self.assertEqual(int(hidden[k].gt(0).sum()), 0)

    def test_real_l0_below_k_without_fake_activation(self):
        sae = tiny_sae()
        with torch.no_grad():
            sae.encoder.weight.zero_()
            sae.encoder.bias.fill_(-1.0)
            sae.encoder.bias[:5].fill_(1.0)  # 恰好5个正预激活
        hidden = sae.encode(torch.zeros(2, 49, 8))
        for k in sae.k_list:
            self.assertEqual(int(hidden[k].gt(0).sum(-1).max()), min(5, k))

    def test_decoder_rows_unit_norm(self):
        sae = tiny_sae()
        norms = sae.decoder_weight.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

    def test_encode_rejects_unknown_k(self):
        sae = tiny_sae()
        with self.assertRaises(ValueError):
            sae.encode(torch.randn(2, 8), k=3)


class TestUnionAndSelection(unittest.TestCase):
    def test_union_equals_widest_layer_when_nested(self):
        counts = {2: np.array([0, 1, 0, 5]), 4: np.array([0, 3, 0, 9]),
                  8: np.array([2, 3, 0, 9])}
        mask = union_nondead_mask(counts)
        self.assertEqual(mask.tolist(), [True, True, False, True])

    def test_union_rejects_broken_nesting(self):
        counts = {2: np.array([7, 0]), 4: np.array([0, 1])}
        with self.assertRaises(RuntimeError):
            union_nondead_mask(counts)

    def test_select_min_passing_k(self):
        gates = {
            64: {"a": True, "b": False},
            128: {"a": True, "b": True},
            256: {"a": True, "b": True},
        }
        self.assertEqual(select_min_passing_k(gates), 128)
        self.assertIsNone(select_min_passing_k({64: {"a": False}}))


class TestGammaCalibration(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(gamma_from_medians(0.4, 0.08), 0.25 * 0.4 / 0.08)

    def test_denominator_guard(self):
        with self.assertRaises(RuntimeError):
            gamma_from_medians(0.4, 1e-9)
        with self.assertRaises(RuntimeError):
            gamma_from_medians(float("nan"), 0.1)

    def test_formal_requires_74_batches(self):
        sae = tiny_sae()
        metadata = pd.DataFrame({
            "patient_id": [f"p{i}" for i in range(8)],
            "label": [0, 1] * 4,
        })
        spatial = np.random.rand(8, 49, 8).astype(np.float32)
        pooled = np.random.rand(8, 8).astype(np.float32)
        attention = np.full((8, 49), 1 / 49, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(debug=False, output_root=Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "74批"):
                calibrate_gamma(
                    tiny_model(), sae, spatial, pooled, attention, metadata,
                    args, torch.device("cpu"),
                )

    def test_debug_calibration_writes_json(self):
        sae = tiny_sae()
        metadata = pd.DataFrame({
            "patient_id": [f"p{i}" for i in range(8)],
            "label": [0, 1] * 4,
        })
        rng = np.random.default_rng(0)
        spatial = rng.random((8, 49, 8), dtype=np.float32)
        pooled = rng.random((8, 8), dtype=np.float32)
        attention = np.full((8, 49), 1 / 49, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(debug=True, output_root=Path(tmp))
            result = calibrate_gamma(
                tiny_model(), sae, spatial, pooled, attention, metadata,
                args, torch.device("cpu"),
            )
            self.assertEqual(result["k_list"], [2, 4, 8])
            self.assertGreater(result["gamma_pool"], 0)
            self.assertTrue((Path(tmp) / "gamma_pool_calibration_seed42.json").is_file())

    def test_load_gamma_recomputes_formula_and_checks_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, cache = root / "output", root / "cache"
            output.mkdir(); cache.mkdir()
            cache_config = {"files": {}}
            (cache / "cache_config.json").write_text(
                json.dumps(cache_config), encoding="utf-8",
            )
            record = {
                "protocol": GAMMA_PROTOCOL, "seed": 42, "batch_size": 32,
                "batch_count": 1, "batch_sha256": ["a" * 64],
                "k_list": [2, 4, 8], "s2c_initialization_sha256": "expected",
                "median_joint_patch": 0.4, "median_joint_pool": 0.1,
                "gamma_pool": 1.0, "manifest_sha256": MANIFEST_SHA256,
                "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
                "cache_files": {},
            }
            record["cache_config_sha256"] = s2c_module.file_sha256(
                cache / "cache_config.json"
            )
            gamma_path = output / "gamma_pool_calibration_seed42.json"
            gamma_path.write_text(json.dumps(record), encoding="utf-8")
            original_cache_root = s2c_module.cache_root
            s2c_module.cache_root = lambda: cache
            try:
                args = SimpleNamespace(debug=True, output_root=output)
                gamma, _config = load_gamma(args, (2, 4, 8), "expected")
                self.assertEqual(gamma, 1.0)
                with self.assertRaisesRegex(ValueError, "initialization"):
                    load_gamma(args, (2, 4, 8), "wrong")
                record["gamma_pool"] = 0.5
                gamma_path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "gamma_formula"):
                    load_gamma(args, (2, 4, 8), "expected")
            finally:
                s2c_module.cache_root = original_cache_root


class TestVarianceDiagnostics(unittest.TestCase):
    def test_fvu_and_negative_explained_variance(self):
        exact = fvu_from_sums(2.0, 4.0)
        self.assertAlmostEqual(exact["fvu"], 0.5)
        self.assertAlmostEqual(exact["explained_variance"], 0.5)
        worse = fvu_from_sums(6.0, 4.0)
        self.assertAlmostEqual(worse["explained_variance"], -0.5)

    def test_fvu_guards(self):
        with self.assertRaises(RuntimeError):
            fvu_from_sums(1.0, 0.0)
        with self.assertRaises(RuntimeError):
            fvu_from_sums(-1.0, 2.0)

    def test_frozen_reference_and_identity_reconstruction(self):
        train_spatial = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        train_pooled = train_spatial.mean(1)
        reference = frozen_variance_reference(train_spatial, train_pooled)
        np.testing.assert_allclose(
            reference["mu_patch_train"], train_spatial.mean((0, 1)),
        )

        class IdentityView:
            @staticmethod
            def encode(value):
                return value

            @staticmethod
            def decode(value):
                return value

        result = variance_diagnostics(
            IdentityView(), train_spatial[:2], train_pooled[:2], train_pooled[:2],
            reference, 1, torch.device("cpu"),
        )
        self.assertEqual(result["patch_fvu"]["value"], 0.0)
        self.assertEqual(result["pooled_fvu"]["value"], 0.0)
        self.assertTrue(result["patch_fvu"]["diagnostic_only"])

    def test_value_strata_includes_frozen_groups(self):
        metadata = pd.DataFrame({
            "label": [0, 1, 1], "source": ["a", "a", "b"],
            "size_group": ["small", "large", "large"],
            "pip_present_original": ["no", "yes", "no"],
            "bbox_area_fraction": [np.nan, 0.1, 0.4],
        })
        result = value_strata(metadata, np.array([0.8, 0.9, 1.0]), np.array([0.2, 0.3]))
        self.assertIn("size_group", result)
        self.assertIn("pip_present_original", result)
        self.assertEqual(result["lesion_size"]["small"]["images"], 1)
        self.assertEqual(result["lesion_size"]["large"]["images"], 1)


class TestLossAndEvaluationHelpers(unittest.TestCase):
    def test_joint_loss_terms_structure(self):
        sae = tiny_sae()
        features = torch.randn(4, 49, 8)
        original_pool = torch.randn(4, 8)
        original_attention = torch.full((4, 49), 1 / 49)
        totals, terms = joint_layer_losses(
            sae, features, original_pool, original_attention, tiny_model(),
            torch.randn(8), 1.0, 0.5,
        )
        self.assertEqual(sorted(totals), [2, 4, 8])
        for k, layer in terms.items():
            expected = layer["patch"] + 0.5 * layer["pool"] + 0.1 * layer["margin"]
            self.assertTrue(torch.allclose(totals[k], expected))
        self.assertTrue(torch.allclose(
            joint_total(totals), torch.stack(list(totals.values())).mean()
        ))

    def test_l0_profile_reports_shortfall(self):
        sae = tiny_sae()
        with torch.no_grad():
            sae.encoder.bias.fill_(-0.5)  # 制造大量非正预激活
        view = SingleKView(sae, 8)
        spatial = np.random.rand(6, 49, 8).astype(np.float32)
        profile = l0_profile(view, spatial, 4, torch.device("cpu"))
        self.assertIn("mean_l0_per_position", profile)
        self.assertGreater(profile["shortfall_ratio_l0_below_k"], 0)
        with self.assertRaises(ValueError):
            SingleKView(sae, 3)


class TestFormalGuards(unittest.TestCase):
    def test_formal_budget_locked(self):
        base = dict(debug=False, experiment="s2c_clong_seed42", device="cuda",
                    seed=42, output_root=OUTPUT_ROOT, **FORMAL_BUDGET)
        validate_args(SimpleNamespace(**base))
        for field in FORMAL_BUDGET:
            broken = dict(base)
            broken[field] = FORMAL_BUDGET[field] * 2
            with self.assertRaises(ValueError):
                validate_args(SimpleNamespace(**broken))

    def test_formal_name_and_device_locked(self):
        base = dict(debug=False, experiment="s2c_clong_seed42", device="cuda",
                    seed=42, output_root=OUTPUT_ROOT, **FORMAL_BUDGET)
        self.assertEqual(formal_name(202), "s2c_clong_seed202")
        with self.assertRaises(ValueError):
            validate_args(SimpleNamespace(**{**base, "experiment": "wrong"}))
        with self.assertRaises(ValueError):
            validate_args(SimpleNamespace(**{**base, "device": "cpu"}))
        with self.assertRaises(ValueError):
            validate_args(SimpleNamespace(**{**base, "output_root": Path("/tmp/unused")}))

    def test_debug_isolated_from_formal(self):
        args = SimpleNamespace(
            debug=True, experiment="s2c_clong_seed42", seed=42,
            debug_patients_per_class=2, output_root=Path("/tmp/unused"),
        )
        with self.assertRaises(ValueError):
            validate_args(args)
        args = SimpleNamespace(
            debug=True, experiment=None, seed=42,
            debug_patients_per_class=2, output_root=Path("/tmp/unused"),
        )
        validate_args(args)
        self.assertIn("debug", str(args.output_root))

    def test_replication_requires_seed42_frozen_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed42_dir = root / "s2c_clong_seed42"
            seed42_dir.mkdir(parents=True)
            checkpoint = seed42_dir / "sae_best.pth"
            checkpoint.write_bytes(b"checkpoint")
            gamma = root / "gamma_pool_calibration_seed42.json"
            gamma.write_text("{}", encoding="utf-8")
            config = {
                "stage": "S2c", "seed": 42, "status": "s2c_k_selected",
                "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
                "manifest_sha256": MANIFEST_SHA256,
                "checkpoint_sha256": s2c_module.file_sha256(checkpoint),
                "gamma_calibration_json_sha256": s2c_module.file_sha256(gamma),
                "evaluation": {"selected_k": 128},
                "test_evaluated": False, "internal_test_evaluated": False,
                "external_evaluated": False,
            }
            (seed42_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8",
            )
            original_output_root = s2c_module.OUTPUT_ROOT
            s2c_module.OUTPUT_ROOT = root
            try:
                target, _config = replication_contract(
                    SimpleNamespace(debug=False, seed=202)
                )
                self.assertEqual(target, 128)
                config["status"] = "no_product_stop_s2c"
                (seed42_dir / "config.json").write_text(
                    json.dumps(config), encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "status"):
                    replication_contract(SimpleNamespace(debug=False, seed=202))
            finally:
                s2c_module.OUTPUT_ROOT = original_output_root


if __name__ == "__main__":
    unittest.main(verbosity=2)
