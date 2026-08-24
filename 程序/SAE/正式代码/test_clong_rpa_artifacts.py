#!/usr/bin/env python3
"""RP-A整体bundle、schema和失败路径测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from clong_rpa_artifacts import (
    protocol_bundle_payload,
    validate_anchors,
    validate_best_hypotheses,
    validate_bootstrap_records,
    validate_edges,
    validate_failure_record,
    validate_output_file_set,
    validate_six_metrics,
)
from clong_rpa_bootstrap import METRIC_NAMES


ROOT = Path(__file__).resolve().parent


def bootstrap_record(index: int) -> dict:
    return {
        "replicate_index": index,
        "status": "completed",
        "fold_metrics": {
            str(fold): {metric: 0.5 for metric in METRIC_NAMES}
            for fold in (42, 43, 44)
        },
    }


class ArtifactTests(unittest.TestCase):
    """覆盖协议bundle、最小证据和两类失败。"""

    def test_protocol_bundle_contains_only_frozen_protocols(self) -> None:
        members, bundle_sha = protocol_bundle_payload(ROOT)
        self.assertEqual(len(members), 7)
        self.assertEqual(len(bundle_sha), 64)
        self.assertTrue(all(row["file"].endswith(".json") for row in members))

    def test_bundle_manifest_matches_live_protocols(self) -> None:
        manifest = json.loads(
            (ROOT / "rpa_protocol_bundle_v1.json").read_text(encoding="utf-8")
        )
        members, bundle_sha = protocol_bundle_payload(ROOT)
        self.assertEqual(manifest["members"], members)
        self.assertEqual(manifest["protocol_bundle_sha256"], bundle_sha)
        self.assertIn("git_commit", manifest["excludes"])

    def test_unique_best_hypotheses(self) -> None:
        row = {"source_seed": 42, "source_feature_id": 1, "target_seed": 43,
               "target_feature_id": 2, "p_value": 0.01, "bh_rejected": True}
        validate_best_hypotheses([row])
        with self.assertRaises(ValueError):
            validate_best_hypotheses([row, row])

    def test_edges_are_reciprocal_and_one_to_one(self) -> None:
        row = {"seed_a": 42, "feature_a": 1, "seed_b": 43, "feature_b": 2,
               "both_directions_bh_rejected": True, "reciprocal": True}
        validate_edges([row])
        with self.assertRaises(ValueError):
            validate_edges([row, {**row, "feature_b": 3}])

    def test_anchors_are_strict_three_cliques(self) -> None:
        row = {"anchor_id": "a1", "feature_42": 1, "feature_43": 2, "feature_44": 3,
               "edge_42_43": True, "edge_42_44": True, "edge_43_44": True}
        validate_anchors([row])
        with self.assertRaises(ValueError):
            validate_anchors([{**row, "edge_42_44": False}])

    def test_bootstrap_requires_all_400_records(self) -> None:
        records = [bootstrap_record(index) for index in range(400)]
        validate_bootstrap_records(records)
        with self.assertRaises(ValueError):
            validate_bootstrap_records(records[:-1])

    def test_six_metrics_are_exact_and_bounded(self) -> None:
        validate_six_metrics({metric: 0.5 for metric in METRIC_NAMES})
        with self.assertRaises(ValueError):
            validate_six_metrics({metric: 0.5 for metric in METRIC_NAMES[:-1]})

    def test_scientific_failure_is_formal_result(self) -> None:
        validate_failure_record({
            "failure_type": "scientific_failure",
            "is_formal_result": True,
            "downstream_allowed": False,
            "reason": "development_calibration_infeasible",
        })

    def test_implementation_failure_cannot_claim_science(self) -> None:
        record = {
            "failure_type": "implementation_failure", "is_formal_result": False,
            "failure_stage": "matching", "exception_type": "RuntimeError",
            "message": "x", "traceback": "trace", "git_commit": "abc",
            "protocol_bundle_sha256": "0" * 64,
        }
        validate_failure_record(record)
        with self.assertRaises(ValueError):
            validate_failure_record({**record, "scientific_conclusion": "failed"})

    def test_completed_output_is_complete_and_rejects_large_matrix(self) -> None:
        files = {
            "run_manifest.json", "best_directed_hypotheses.csv", "reciprocal_edges.csv",
            "development_anchors.csv", "bootstrap_records.jsonl", "bootstrap_thresholds.json",
            "formal_metrics.json", "gpu_identity.json", "formal_result.json", "SHA256SUMS.txt",
        }
        validate_output_file_set(files, "completed")
        with self.assertRaises(ValueError):
            validate_output_file_set(files | {"complete_candidate_matrix.npy"}, "completed")

    def test_failure_output_sets_are_disjoint(self) -> None:
        validate_output_file_set(
            {"run_manifest.json", "formal_result.json", "SHA256SUMS.txt"},
            "scientific_failure",
        )
        validate_output_file_set({"implementation_failure.json"}, "implementation_failure")
        with self.assertRaises(ValueError):
            validate_output_file_set(
                {"implementation_failure.json", "formal_result.json"},
                "implementation_failure",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
