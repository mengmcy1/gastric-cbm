from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stage01_common import prepare_output_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总单样本 Stage 1 三档正式 A/B 自动结果。")
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angles", type=int, nargs="+", default=[5, 15, 30])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir)
    rows = []
    details = []
    for angle in args.angles:
        run = args.sample_root / f"ang{angle}"
        integrity = read_json(run / "integrity.json")
        if integrity["status"] != "passed":
            raise RuntimeError(f"完整性未通过：{run}")
        a = read_csv(run / "render_a" / "per_frame_metrics.csv")
        b = read_csv(run / "render_b" / "per_frame_metrics.csv")
        ab = read_json(run / "summary" / "ab_summary.json")
        angle_detail = {"angle_total_deg": angle, "sides": {}}
        for side, index in (("left", 0), ("right", -1)):
            depth = read_json(run / f"{side}_depth" / "depth_run.json")
            supplement = read_json(run / f"{side}_supplement" / "supplement_stats.json")
            hole_a = as_float(a[index], "alpha_hole_lt_095")
            hole_b = as_float(b[index], "alpha_hole_lt_095")
            reduction = (hole_a - hole_b) / hole_a if hole_a > 0 else None
            row = {
                "angle_total_deg": angle,
                "side": side,
                "frames": len(a),
                "p95_px_a": as_float(a[index], "projection_p95_px"),
                "p95_width_fraction_a": as_float(a[index], "projection_p95_width_fraction"),
                "full_frame_hole_rate_a": hole_a,
                "full_frame_hole_rate_b": hole_b,
                "relative_hole_reduction": reduction,
                "depth_gate_status": depth["quality_gate"]["status"],
                "depth_accepted_pixels": depth["alignment"]["accepted_pixels"],
                "depth_rejected_pixels": depth["alignment"]["rejected_pixels"],
                "supplement_gaussian_count": supplement["supplement_gaussian_count"],
                "endpoint_visibility_weight": as_float(
                    b[index], f"{side}_supplement_visibility_weight"
                ),
            }
            rows.append(row)
            angle_detail["sides"][side] = {
                **row,
                "depth_components": depth["alignment"]["components"],
            }
        angle_detail["center_frame_rgb_difference_0_255"] = ab[
            "center_frame_rgb_difference_0_255"
        ]
        angle_detail["visibility"] = ab["supplement_visibility"]
        details.append(angle_detail)

    with (output_dir / "stage1_matrix_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    thirty = [r for r in rows if r["angle_total_deg"] == 30]
    alpha_gate = bool(
        len(thirty) == 2
        and all(
            r["relative_hole_reduction"] is not None
            and r["relative_hole_reduction"] >= 0.30
            for r in thirty
        )
    )
    write_json(
        output_dir / "stage1_matrix_summary.json",
        {
            "schema_version": "1.0-stage1-formal-matrix-summary",
            "sample_root": str(args.sample_root.resolve()),
            "rows": rows,
            "details": details,
            "automatic_candidate_gate": {
                "criterion": "30-degree full-frame endpoint Alpha<0.95 hole rate reduction >=30% on both endpoints",
                "passed": alpha_gate,
                "note": "Alpha gate alone cannot approve visual quality.",
            },
            "manual_review_status": "pending_human_full_video_playback",
        },
    )
    write_json(
        output_dir / "manual_review_template.json",
        {
            "schema_version": "1.0-stage1-manual-review",
            "review_status": "pending",
            "evaluator": None,
            "scale": "severity: 0 none, 1 mild, 2 obvious, 3 severe; overall_pass: true/false",
            "videos": [
                {
                    "angle_total_deg": angle,
                    "comparison": "B versus A, full playback",
                    "disoc_severity": None,
                    "cover_severity": None,
                    "depth_severity": None,
                    "stretch_smear_severity": None,
                    "floating_patch_severity": None,
                    "flicker_jump_severity": None,
                    "center_contamination_severity": None,
                    "opposite_view_contamination_severity": None,
                    "first_artifact_frame": None,
                    "overall_pass": None,
                    "notes": None,
                }
                for angle in args.angles
            ],
        },
    )


if __name__ == "__main__":
    main()
