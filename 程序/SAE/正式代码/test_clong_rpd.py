#!/usr/bin/env python3
"""RP-D v1病例选择规则回归测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from clong_rpd_core import (
    blind_hash,
    build_patient_table,
    select_activation_bins,
    select_hard_negative,
    stable_hash,
)
from build_clong_rpd_manifests import dry_run_boundary_flags


class RPDCaseSelectionTests(unittest.TestCase):
    def test_dry_run_boundary_flags(self) -> None:
        flags = dry_run_boundary_flags()
        self.assertTrue(flags["train_only"])
        self.assertFalse(flags["images_read"])
        self.assertFalse(flags["assets_rendered"])
        self.assertFalse(flags["scientific_pass_fail_generated"])

    def test_hash_is_deterministic_and_domain_separated(self) -> None:
        first = stable_hash("a1", "p1", "x.jpg")
        self.assertEqual(first, stable_hash("a1", "p1", "x.jpg"))
        self.assertNotEqual(first, blind_hash("a1", "p1"))

    def test_patient_representative_uses_peak_mass_hash(self) -> None:
        frame = pd.DataFrame({
            "relative_path": ["a", "b", "c"],
            "patient_id": ["p", "p", "q"],
            "label": [1, 1, 0],
            "source": ["s", "s", "s"],
        })
        result = build_patient_table(frame, np.array([2.0, 2.0, 0.0]), np.array([0.2, 0.4, 0.0]), "a1")
        self.assertEqual(result.loc[result.patient_id.eq("p"), "relative_path"].item(), "b")
        self.assertEqual(result.loc[result.patient_id.eq("q"), "patient_peak_activation"].item(), 0.0)

    def test_bins_are_patient_disjoint_and_ordered(self) -> None:
        frame = pd.DataFrame({
            "patient_id": [f"p{i}" for i in range(10)],
            "label": [1] * 10,
            "source": ["s"] * 10,
            "patient_peak_activation": [10, 9, 8, 7, 6, 5, 4, 3, 0, 0],
            "image_index": range(10),
            "relative_path": [f"{i}.jpg" for i in range(10)],
            "image_peak_activation": [10, 9, 8, 7, 6, 5, 4, 3, 0, 0],
            "image_mass_activation": [1.0] * 10,
            "selection_hash": [f"{i:02d}" for i in range(10)],
        })
        selected, shortfalls = select_activation_bins(frame, "a1", 1, 2)
        self.assertEqual(selected.patient_id.nunique(), len(selected))
        self.assertEqual(selected[selected.case_role.eq("high")].patient_id.tolist(), ["p0", "p1"])
        self.assertEqual(selected[selected.case_role.eq("low")].patient_id.tolist(), ["p7", "p6"])
        self.assertEqual(shortfalls, {"high": 0, "low": 0, "mid": 0, "zero": 0})

    def test_bin_shortfall_does_not_replace(self) -> None:
        frame = pd.DataFrame({
            "patient_id": ["p1", "p2"], "label": [0, 0], "source": ["s", "s"],
            "patient_peak_activation": [1.0, 0.0], "image_index": [0, 1],
            "relative_path": ["a", "b"], "image_peak_activation": [1.0, 0.0],
            "image_mass_activation": [0.1, 0.0], "selection_hash": ["a", "b"],
        })
        selected, shortfalls = select_activation_bins(frame, "a1", 0, 3)
        self.assertEqual(len(selected), 2)
        self.assertGreater(sum(shortfalls.values()), 0)

    def test_hard_negative_prefers_zero_pool(self) -> None:
        patients = pd.DataFrame({
            "patient_id": ["q", "z", "l"], "label": [1, 1, 1], "source": ["s"] * 3,
            "patient_peak_activation": [2.0, 0.0, 0.1], "image_index": [0, 1, 2],
            "relative_path": ["q", "z", "l"], "image_peak_activation": [2.0, 0.0, 0.1],
            "image_mass_activation": [1.0, 0.0, 0.1], "selection_hash": ["q", "z", "l"],
        })
        pooled = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]])
        chosen, _, status = select_hard_negative(patients.iloc[0], patients, patients.iloc[[2]], pooled)
        self.assertEqual(chosen.patient_id, "z")
        self.assertEqual(status, "zero")

    def test_hard_negative_uses_selected_low_only_as_fallback(self) -> None:
        patients = pd.DataFrame({
            "patient_id": ["q", "l"], "label": [1, 1], "source": ["s", "s"],
            "patient_peak_activation": [2.0, 0.1], "image_index": [0, 1],
            "relative_path": ["q", "l"], "image_peak_activation": [2.0, 0.1],
            "image_mass_activation": [1.0, 0.1], "selection_hash": ["q", "l"],
        })
        chosen, _, status = select_hard_negative(
            patients.iloc[0], patients, patients.iloc[[1]], np.eye(2),
        )
        self.assertEqual(chosen.patient_id, "l")
        self.assertEqual(status, "selected_low_fallback")


if __name__ == "__main__":
    unittest.main()
