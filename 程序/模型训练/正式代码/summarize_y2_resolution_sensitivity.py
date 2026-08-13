#!/usr/bin/env python3
"""Summarize Y2-S convergence changes without replacing primary Y2-B."""

from __future__ import annotations

import json

import pandas as pd

from train_y2_resolution_sensitivity import DEFAULT_OUTPUT_ROOT, expected_run_name


RESOLUTIONS = (640, 960)
SEED = 42


def load_result(imgsz: int) -> dict:
    """Load one complete formal Y2-S result and reject debug products."""
    path = DEFAULT_OUTPUT_ROOT / expected_run_name(imgsz, SEED, False) / "y2s_geometry_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing formal Y2-S result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["debug"] or result["internal_test_read"] or result["external_read"]:
        raise ValueError(f"Y2-S {imgsz} result role mismatch")
    if result["primary_y2_replacement_allowed"]:
        raise ValueError("Y2-S must never directly replace primary Y2-B")
    return result


def main() -> None:
    """Compare original and tuned geometry, then test whether 960 catches 640."""
    output_dir = DEFAULT_OUTPUT_ROOT / "自动汇总"
    if output_dir.exists():
        raise FileExistsError(f"Y2-S summary directory already exists: {output_dir}")

    rows = []
    results = {imgsz: load_result(imgsz) for imgsz in RESOLUTIONS}
    for imgsz, result in results.items():
        original = result["original_y2_geometry"]
        tuned = result["geometry"]
        delta = result["delta_from_original_y2"]
        rows.append(
            {
                "imgsz": imgsz,
                "eligible": bool(result["comparison_eligible"]),
                "original_sensitivity": original["sensitivity"],
                "tuned_sensitivity": tuned["sensitivity"],
                "delta_sensitivity": delta["sensitivity"],
                "original_iou_ge_0p5": original["iou_ge_0p5"],
                "tuned_iou_ge_0p5": tuned["iou_ge_0p5"],
                "delta_iou_ge_0p5": delta["iou_ge_0p5"],
                "original_mean_iou": original["mean_iou"],
                "tuned_mean_iou": tuned["mean_iou"],
                "delta_mean_iou": delta["mean_iou"],
                "original_noncancer_trigger": original["noncancer_positive_trigger_rate"],
                "tuned_noncancer_trigger": tuned["noncancer_positive_trigger_rate"],
                "delta_noncancer_trigger": delta["noncancer_positive_trigger_rate"],
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["eligible"]].copy()
    if len(eligible) == len(RESOLUTIONS):
        ranked = eligible.sort_values(
            ["tuned_iou_ge_0p5", "tuned_mean_iou", "tuned_noncancer_trigger", "imgsz"],
            ascending=[False, False, True, True],
        )
        tuned_leader = int(ranked.iloc[0]["imgsz"])
        catch_up = tuned_leader == 960
        decision = "960_caught_up_trigger_multiseed_review" if catch_up else "640_remains_leader"
    else:
        tuned_leader = None
        catch_up = False
        decision = "incomplete_or_failed_safety_gate"

    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(output_dir / "y2s_convergence_comparison.csv", index=False)
    summary = {
        "stage": "Y2-S",
        "role": "supplementary_no_mosaic_convergence_sensitivity",
        "primary_y2_replacement_allowed": False,
        "ranking_hierarchy": [
            "paired M1 sensitivity engineering safety gate",
            "tuned IoU>=0.50 rate descending",
            "tuned mean IoU descending",
            "tuned noncancer positive-trigger rate ascending",
            "input resolution/computational cost ascending",
        ],
        "decision": decision,
        "tuned_leader": tuned_leader,
        "960_caught_up": catch_up,
        "internal_test_read": False,
        "external_read": False,
        "rows": rows,
    }
    (output_dir / "y2s_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"Y2-S decision: {decision}; tuned leader={tuned_leader}")


if __name__ == "__main__":
    main()
