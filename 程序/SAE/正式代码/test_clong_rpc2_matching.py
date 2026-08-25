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


if __name__ == "__main__":
    unittest.main()
