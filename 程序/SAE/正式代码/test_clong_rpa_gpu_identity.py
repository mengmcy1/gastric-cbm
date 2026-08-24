#!/usr/bin/env python3
"""RP-A GPU no-op identity核心的CPU语义测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from clong_rpa_gpu_identity import (
    ErrorAccumulator,
    derive_tolerances,
    identity_passes,
)


PROTOCOL = Path(__file__).with_name("rpa_gpu_identity_protocol_v1.json")


class IdentityTests(unittest.TestCase):
    """验证误差汇总、容差公式和逐元素gate。"""

    def test_protocol_frozen_and_fixed_input(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["status"], "frozen_2026-08-24")
        self.assertEqual(protocol["fixed_input"]["batch_size"], 32)
        self.assertEqual(protocol["fixed_input"]["representation_k"], 1024)
        self.assertFalse(protocol["numeric_path"]["tf32"])
        self.assertEqual(set(protocol["frozen_tolerances"]), {
            "patch", "attention", "pooled", "logits", "probability"
        })
        self.assertTrue(protocol["accepted_audit"]["all_five_levels_passed"])

    def test_error_accumulator(self) -> None:
        accumulator = ErrorAccumulator()
        accumulator.update(torch.tensor([1.0, 0.0]), torch.tensor([1.25, 0.5]))
        summary = accumulator.summary()
        self.assertEqual(summary["max_abs_error"], 0.5)
        self.assertEqual(summary["mean_abs_error"], 0.375)
        self.assertEqual(summary["max_abs_reference"], 1.0)

    def test_tolerance_formula(self) -> None:
        summaries = {
            level: {"max_abs_error": 1e-7, "max_abs_reference": 2.0}
            for level in ("patch", "attention", "pooled", "logits", "probability")
        }
        tolerances = derive_tolerances(summaries)
        expected_rtol = 32 * np.finfo(np.float32).eps
        self.assertEqual(tolerances["patch"]["rtol"], expected_rtol)
        self.assertEqual(tolerances["patch"]["atol"], expected_rtol * 2.0)

    def test_identity_gate_boundary(self) -> None:
        reference = torch.tensor([0.0, 2.0])
        allowed = torch.tensor([1e-4, 2.0002])
        rejected = torch.tensor([1.01e-4, 2.0002])
        self.assertTrue(identity_passes(reference, allowed, atol=1e-4, rtol=1e-4))
        self.assertFalse(identity_passes(reference, rejected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
