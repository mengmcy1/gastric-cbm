#!/usr/bin/env python3
"""Select the single Y2-B resolution using the pre-registered hierarchy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from train_y2_yolo26 import DEFAULT_OUTPUT_ROOT, expected_run_name


RESOLUTIONS = (640, 960)
SEED = 42


def load_result(imgsz: int) -> dict:
    """Load one complete formal geometry result and reject debug products."""
    run_dir = DEFAULT_OUTPUT_ROOT / expected_run_name(imgsz, SEED, False)
    path = run_dir / "y2_geometry_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少Y2-B {imgsz}正式几何结果: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["debug"] or result["internal_test_read"] or result["external_read"]:
        raise ValueError(f"Y2-B {imgsz}结果角色或数据边界异常")
    return result


def main() -> None:
    """Apply safety first, then IoU50, mean IoU, false positives and cost."""
    output_dir = DEFAULT_OUTPUT_ROOT / "自动汇总"
    if output_dir.exists():
        raise FileExistsError(f"Y2汇总目录已存在，拒绝覆盖: {output_dir}")
    results = {imgsz: load_result(imgsz) for imgsz in RESOLUTIONS}
    rows = []
    for imgsz, result in results.items():
        geometry = result["geometry"]
        safety = result["safety"]
        rows.append(
            {
                "imgsz": imgsz,
                "eligible": bool(result["selection_eligible"]),
                "sensitivity": geometry["sensitivity"],
                "sensitivity_minus_m1": safety["point_difference_yolo_minus_m1"],
                "sensitivity_ci_low": safety["bootstrap_percentile_95_ci"][0],
                "sensitivity_ci_high": safety["bootstrap_percentile_95_ci"][1],
                "iou_ge_0p5": geometry["iou_ge_0p5"],
                "mean_iou": geometry["mean_iou"],
                "median_iou": geometry["median_iou"],
                "center_hit_rate": geometry["center_hit_rate"],
                "noncancer_positive_trigger_rate": geometry["noncancer_positive_trigger_rate"],
                "noncancer_boxes_per_image": geometry["noncancer_boxes_per_image"],
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["eligible"]].copy()
    if eligible.empty:
        decision = "failed_no_resolution_passed_sensitivity_safety_gate"
        selected = None
    else:
        ranked = eligible.sort_values(
            ["iou_ge_0p5", "mean_iou", "noncancer_positive_trigger_rate", "imgsz"],
            ascending=[False, False, True, True],
        )
        selected = int(ranked.iloc[0]["imgsz"])
        decision = "selected_by_preregistered_hierarchy"

    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(output_dir / "y2_resolution_comparison.csv", index=False)
    summary = {
        "stage": "Y2-B",
        "seed": SEED,
        "selection_hierarchy": [
            "sensitivity engineering safety gate",
            "IoU>=0.50 rate descending",
            "mean IoU descending",
            "noncancer positive-trigger rate ascending",
            "input resolution/computational cost ascending",
        ],
        "decision": decision,
        "selected_imgsz": selected,
        "internal_test_read": False,
        "external_read": False,
        "rows": rows,
    }
    (output_dir / "y2_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    if selected is None:
        raise RuntimeError("Y2-B失败: 640和960均未通过癌检出安全门槛")
    print(f"Y2-B入选分辨率: {selected}")


if __name__ == "__main__":
    main()
