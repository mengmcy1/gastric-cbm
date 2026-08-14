#!/usr/bin/env python3
"""Summarize coverage loss along an InfiniSplat real-pose interpolation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def first_crossing(rows: list[dict[str, float]], key: str, threshold: float) -> dict | None:
    for row in rows:
        if row[key] >= threshold:
            return {
                "threshold": threshold,
                "frame": int(row["frame"]),
                "progress": row["progress"],
                "baseline_m": row["baseline_m"],
                "relative_rotation_deg": row["relative_rotation_deg"],
                "value": row[key],
            }
    return None


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    csv_path = run_dir / "per_frame_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]

    hard_key = "hard_hole_alpha_lt_0_01"
    soft_key = "soft_hole_alpha_lt_0_5"
    report = {
        "schema_version": "1.0-infinisplat-pose-interpolation-analysis",
        "source_csv": csv_path.name,
        "frame_count": len(rows),
        "hard_hole_first_crossings": [
            first_crossing(rows, hard_key, threshold)
            for threshold in (0.05, 0.10, 0.25, 0.40)
        ],
        "soft_hole_first_crossings": [
            first_crossing(rows, soft_key, threshold)
            for threshold in (0.05, 0.10, 0.25, 0.40)
        ],
        "hard_hole_monotonic_non_decreasing": all(
            current[hard_key] >= previous[hard_key]
            for previous, current in zip(rows, rows[1:])
        ),
        "soft_hole_monotonic_non_decreasing": all(
            current[soft_key] >= previous[soft_key]
            for previous, current in zip(rows, rows[1:])
        ),
        "endpoint": rows[-1],
    }
    (run_dir / "trajectory_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, left = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    progress = [row["progress"] * 100.0 for row in rows]
    left.plot(progress, [row[hard_key] * 100.0 for row in rows], label="Hard holes (alpha < 0.01)", linewidth=2.4)
    left.plot(progress, [row[soft_key] * 100.0 for row in rows], label="Soft holes (alpha < 0.5)", linewidth=2.4)
    left.set_xlabel("Source-to-target trajectory progress (%)")
    left.set_ylabel("Image area (%)")
    left.set_xlim(0, 100)
    left.set_ylim(0, 55)
    left.grid(True, alpha=0.25)
    left.legend(loc="upper left")
    right = left.twiny()
    right.set_xlim(0, rows[-1]["baseline_m"])
    right.set_xlabel("Interpolated metric baseline (m)")
    left.set_title("InfiniSplat coverage loss on a real ETH3D camera path")
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory_coverage_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
