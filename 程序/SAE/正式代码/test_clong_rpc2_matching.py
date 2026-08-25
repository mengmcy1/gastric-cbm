#!/usr/bin/env python3
"""RP-C2 outcome-blind matching geometry 单元测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from clong_rpc2_matching_core import (
    audit_target_geometry,
    empirical_midrank,
    patient_equal_covariate,
    select_nearest_controls,
    smallest_global_feasible_caliper,
)


class RPC2MatchingTests(unittest.TestCase):
    """覆盖midrank、患者等权、L∞、排除集合和全局caliper。"""

    def test_midrank_ties(self) -> None:
        result = empirical_midrank(np.asarray([1.0, 1.0, 3.0, 4.0]))
        np.testing.assert_allclose(result, [0.25, 0.25, 0.625, 0.875])

    def test_patient_equal_covariate(self) -> None:
        values = np.asarray([[1.0, 10.0, 2.0], [3.0, 20.0, 6.0]])
        result = patient_equal_covariate(values, np.asarray([0, 2]))
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_linf_and_study_exclusion(self) -> None:
        ids = np.arange(105, dtype=np.int64)
        covariates = np.column_stack([ids / 1000, ids / 2000, ids / 3000])
        result = audit_target_geometry(0, ids, covariates, {0, 1, 2, 3, 4})
        self.assertEqual(result["eligible_control_universe_n"], 100)
        self.assertEqual(result["nearest_1_distance"], 0.005)
        self.assertEqual(result["nearest_100_distance"], 0.104)
        self.assertEqual(result["pool_n_c0025"], 21)

    def test_smallest_global_caliper(self) -> None:
        frame = pd.DataFrame({
            "pool_n_c0025": [80, 90],
            "pool_n_c0050": [100, 101],
            "pool_n_c0075": [120, 130],
            "pool_n_c0100": [150, 160],
            "pool_n_c0150": [200, 210],
        })
        self.assertEqual(smallest_global_feasible_caliper(frame), 0.05)

    def test_no_global_caliper(self) -> None:
        frame = pd.DataFrame({
            "pool_n_c0025": [99], "pool_n_c0050": [99],
            "pool_n_c0075": [99], "pool_n_c0100": [99],
            "pool_n_c0150": [99],
        })
        self.assertIsNone(smallest_global_feasible_caliper(frame))

    @staticmethod
    def selection_fixture(n: int = 180) -> tuple[np.ndarray, np.ndarray]:
        ids = np.arange(n, dtype=np.int64)
        values = np.linspace(0.0, 0.179, n)
        return ids, np.column_stack([values, values, values])

    def test_pool_150_selects_nearest_100(self) -> None:
        ids, covariates = self.selection_fixture()
        selected, status = select_nearest_controls(0, ids, covariates, {0})
        self.assertEqual(status, "matched")
        self.assertEqual(len(selected), 100)
        np.testing.assert_array_equal(selected.control_feature_id, np.arange(1, 101))

    def test_pool_37_selects_all(self) -> None:
        ids = np.arange(38, dtype=np.int64)
        values = np.linspace(0.0, 0.148, 38)
        covariates = np.column_stack([values, values, values])
        selected, status = select_nearest_controls(0, ids, covariates, {0})
        self.assertEqual(status, "matched")
        self.assertEqual(len(selected), 37)

    def test_pool_19_is_insufficient(self) -> None:
        ids = np.arange(20, dtype=np.int64)
        values = np.linspace(0.0, 0.14, 20)
        covariates = np.column_stack([values, values, values])
        selected, status = select_nearest_controls(0, ids, covariates, {0})
        self.assertEqual(status, "matching_support_insufficient")
        self.assertEqual(len(selected), 0)

    def test_distance_tie_uses_feature_id(self) -> None:
        ids = np.asarray([10, 7, 3, 5, 1, 8, 6, 4, 2, 9, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        covariates = np.zeros((len(ids), 3), dtype=np.float64)
        selected, status = select_nearest_controls(10, ids, covariates, {10})
        self.assertEqual(status, "matched")
        np.testing.assert_array_equal(
            selected.control_feature_id, np.asarray([*range(10), *range(11, 21)]),
        )

    def test_midrank_rational_tie_ignores_float_roundoff(self) -> None:
        ids = np.arange(21, dtype=np.int64)
        covariates = np.tile(np.asarray([4 / 42, 10 / 42, 10 / 42]), (21, 1))
        covariates[0] = [10 / 42, 10 / 42, 10 / 42]
        covariates[1] = [4 / 42, 10 / 42, 10 / 42]
        covariates[2] = [10 / 42, 16 / 42, 10 / 42]
        selected, status = select_nearest_controls(0, ids, covariates, {0})
        self.assertEqual(status, "matched")
        self.assertEqual(selected.iloc[0].control_feature_id, 1)
        self.assertEqual(selected.iloc[1].control_feature_id, 2)
        self.assertEqual(selected.iloc[0].linf_distance_units, 6)
        self.assertEqual(selected.iloc[1].linf_distance_units, 6)

    def test_study_objects_never_selected_and_reuse_allowed(self) -> None:
        ids, covariates = self.selection_fixture()
        excluded = {0, 1, 2, 3}
        first, _ = select_nearest_controls(0, ids, covariates, excluded)
        second, _ = select_nearest_controls(1, ids, covariates, excluded)
        self.assertFalse(first.control_feature_id.isin(excluded).any())
        self.assertFalse(second.control_feature_id.isin(excluded).any())
        self.assertGreater(len(set(first.control_feature_id) & set(second.control_feature_id)), 0)


if __name__ == "__main__":
    unittest.main()
