#!/usr/bin/env python3
"""Summarize three fixed-640 Y3 replicates for Balanced or Full data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_y3_yolo26 import pair_m1_geometry, paired_patient_bootstrap
from train_y2_yolo26 import DEFAULT_OUTPUT_ROOT as Y2_ROOT
from train_y3_yolo26 import ALLOWED_SEEDS, DEFAULT_OUTPUT_ROOT, expected_run_name


def parse_args() -> argparse.Namespace:
    """Parse the dataset role whose three seeds should be summarized."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("balanced", "full"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def balanced_seed42_reference() -> dict:
    """Convert frozen Y2-B seed42 into Y3 schema using same-seed M1 pairing."""
    run_dir = Y2_ROOT / "y2b_yolo26s_640_seed42"
    config = json.loads((run_dir / "y2_geometry_config.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "y2_val_top1_predictions.csv", float_precision="round_trip")
    m1_columns = {
        "localization_confidence",
        "valid_box",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "m1_detected",
        "m1_iou",
        "m1_center_hit",
    }
    predictions = predictions.drop(columns=[c for c in m1_columns if c in predictions.columns])
    predictions, m1_metrics = pair_m1_geometry(predictions, "balanced", 42)
    geometry = config["geometry"]
    differences = {
        "sensitivity": float(geometry["sensitivity"] - m1_metrics["sensitivity"]),
        "iou_ge_0p5": float(geometry["iou_ge_0p5"] - m1_metrics["iou_ge_0p5"]),
        "mean_iou": float(geometry["mean_iou"] - m1_metrics["mean_iou"]),
    }
    return {
        "stage": "Y3-B",
        "role": "balanced",
        "seed": 42,
        "source_stage": "reused_Y2-B_640",
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "m1_paired_baseline": m1_metrics,
        "standard_validation": config["standard_validation"],
        "geometry": geometry,
        "safety": {
            "passed_sensitivity_engineering_gate": differences["sensitivity"] >= -0.02,
            "paired_differences_yolo_minus_m1": differences,
            "bootstrap_percentile_95_ci": paired_patient_bootstrap(predictions, 42),
        },
        "replicate_success": bool(
            differences["sensitivity"] >= -0.02
            and differences["iou_ge_0p5"] > 0
            and differences["mean_iou"] > 0
        ),
    }


def load_result(role: str, seed: int, output_root: Path) -> dict:
    """Load one formal result, reusing Y2-B only for Balanced seed42."""
    if role == "balanced" and seed == 42:
        return balanced_seed42_reference()
    path = output_root / expected_run_name(role, seed, False) / "y3_geometry_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Y3 {role} seed{seed} result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result["role"] != role
        or int(result["seed"]) != seed
        or result["debug"]
        or result["internal_test_read"]
        or result["external_read"]
        or result["full_original_test_read"]
    ):
        raise ValueError(f"Y3 {role} seed{seed} role or data boundary mismatch")
    return result


def main() -> None:
    """Create the pre-registered three-seed table and stage-level conclusion."""
    args = parse_args()
    output_root = args.output_root.resolve()
    output_dir = output_root / f"自动汇总_{args.role}"
    if output_dir.exists():
        raise FileExistsError(f"Y3 summary already exists: {output_dir}")
    results = [load_result(args.role, seed, output_root) for seed in ALLOWED_SEEDS]
    rows = []
    for result in results:
        geometry = result["geometry"]
        m1 = result["m1_paired_baseline"]
        differences = result["safety"]["paired_differences_yolo_minus_m1"]
        rows.append(
            {
                "seed": int(result["seed"]),
                "replicate_success": bool(result["replicate_success"]),
                "mAP50_95": result["standard_validation"]["mAP50_95"],
                "sensitivity": geometry["sensitivity"],
                "m1_sensitivity": m1["sensitivity"],
                "sensitivity_minus_m1": differences["sensitivity"],
                "iou_ge_0p5": geometry["iou_ge_0p5"],
                "m1_iou_ge_0p5": m1["iou_ge_0p5"],
                "iou_ge_0p5_minus_m1": differences["iou_ge_0p5"],
                "mean_iou": geometry["mean_iou"],
                "m1_mean_iou": m1["mean_iou"],
                "mean_iou_minus_m1": differences["mean_iou"],
                "center_hit_rate": geometry["center_hit_rate"],
                "noncancer_trigger_rate": geometry["noncancer_positive_trigger_rate"],
            }
        )
    table = pd.DataFrame(rows)
    metric_columns = [
        "mAP50_95",
        "sensitivity",
        "iou_ge_0p5",
        "mean_iou",
        "center_hit_rate",
        "noncancer_trigger_rate",
    ]
    aggregate = {
        metric: {
            "mean": float(table[metric].mean()),
            "sample_sd": float(table[metric].std(ddof=1)),
            "min": float(table[metric].min()),
            "max": float(table[metric].max()),
        }
        for metric in metric_columns
    }
    success_count = int(table["replicate_success"].sum())
    conclusion = "stable_success_3_of_3" if success_count == 3 else f"failed_{success_count}_of_3"
    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(output_dir / "y3_seed_comparison.csv", index=False)
    summary = {
        "stage": "Y3-B" if args.role == "balanced" else "Y3-F",
        "role": args.role,
        "seeds": list(ALLOWED_SEEDS),
        "success_definition": {
            "sensitivity_minus_same_seed_m1_minimum": -0.02,
            "iou_ge_0p5_minus_same_seed_m1": "strictly positive",
            "mean_iou_minus_same_seed_m1": "strictly positive",
            "stage_success": "all three seeds pass",
        },
        "success_count": success_count,
        "conclusion": conclusion,
        "aggregate": aggregate,
        "internal_test_read": False,
        "external_read": False,
        "full_original_test_read": False,
        "rows": rows,
    }
    (output_dir / "y3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"Y3 {args.role}: {conclusion}; successful seeds={success_count}/3")


if __name__ == "__main__":
    main()
