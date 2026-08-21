#!/usr/bin/env python3
"""RP-A eligible 聚合与 split-half 校准的回归测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from clong_rpa_eligible import (  # noqa: E402
    build_feature_audit,
    evaluate_candidate,
    lower_quantile,
    select_top_q,
    sha_text,
    spearman_from_vectors,
    split_patients,
    validate_protocol,
)


class EligibleCalibrationTests(unittest.TestCase):
    """验证拟冻结协议的确定性、边界和统计口径。"""

    def setUp(self) -> None:
        self.protocol = json.loads((SCRIPT_DIR / "rpa_eligible_protocol_v1.json").read_text())

    def test_protocol_constants(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(self.protocol["split_count"], 400)
        self.assertEqual(self.protocol["quantile_method"], "lower")

    def test_sha_domain_separation(self) -> None:
        self.assertNotEqual(sha_text("patient_order", 1), sha_text("stratum_offset", 1))

    def test_top_q_selects_largest_safe_candidate(self) -> None:
        self.assertEqual(select_top_q(np.array([0.14, 0.25]), [0.01, 0.02, 0.05, 0.10]), 0.10)

    def test_top_q_rejects_all_unsafe_candidates(self) -> None:
        with self.assertRaises(RuntimeError):
            select_top_q(np.array([0.005]), [0.01, 0.02])

    def test_lower_quantile_is_twentieth_of_400(self) -> None:
        values = list(range(1, 401))
        self.assertEqual(lower_quantile(values, 0.05), 20.0)

    def test_split_is_deterministic_and_balanced_per_stratum(self) -> None:
        patients = pd.DataFrame({
            "patient_id": [f"p{i}" for i in range(15)],
            "label": [0] * 7 + [1] * 8,
            "source": ["a"] * 3 + ["b"] * 4 + ["a"] * 5 + ["b"] * 3,
            "image_count": [1] * 15,
        })
        a1, b1, audit1 = split_patients(patients, "protocol", 7, self.protocol)
        a2, b2, audit2 = split_patients(patients, "protocol", 7, self.protocol)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
        self.assertEqual(audit1, audit2)
        self.assertTrue(all(abs(row["n_a"] - row["n_b"]) <= 1 for row in audit1))

    def test_split_offset_can_send_singletons_to_either_half(self) -> None:
        patients = pd.DataFrame({"patient_id": ["p"], "label": [1], "source": ["x"], "image_count": [1]})
        assignments = set()
        for seed in range(20):
            a, _, _ = split_patients(patients, "protocol", seed, self.protocol)
            assignments.add(0 if len(a) else 1)
        self.assertEqual(assignments, {0, 1})

    def test_spearman_ties_and_constant_guard(self) -> None:
        self.assertAlmostEqual(spearman_from_vectors(np.array([1, 2, 2]), np.array([2, 3, 3])), 1.0)
        self.assertIsNone(spearman_from_vectors(np.ones(3), np.arange(3)))

    def test_candidate_uses_fixed_nondead_universe(self) -> None:
        presence = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=bool)
        frequency = presence.astype(float) * 0.1
        result = evaluate_candidate(
            presence, frequency, np.array([0, 1]), np.array([2, 3]),
            np.array([True, True, False]), 0.01, 0.1,
        )
        self.assertTrue(result["computable"])
        self.assertEqual(result["min_eligible_count"], 2)

    def test_feature_audit_contains_frozen_schema_and_q_lineage(self) -> None:
        patients = pd.DataFrame({
            "patient_id": ["p0", "p1", "p2", "p3"],
            "label": [0, 0, 1, 1],
            "source": ["a", "a", "b", "b"],
            "image_count": [1, 2, 1, 2],
        })
        ranking = np.array([[1, 0], [2, 1], [3, 0], [4, 2]], dtype=np.float32)
        matrices = {
            "presence": ranking > 0,
            "ranking": ranking,
            "mass": ranking / 2,
            "active_frequency": (ranking > 0).astype(np.float32) / 49,
        }
        table, summary = build_feature_audit(patients, matrices, self.protocol)
        for column in (
            "ranking_q75", "ranking_q95", "ranking_q99", "mass_q99",
            "activation_mass_total", "top_q10_nonzero_count",
        ):
            self.assertIn(column, table.columns)
        self.assertEqual(summary["selected_top_q"], 0.10)
        self.assertEqual(summary["p_min_train"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
