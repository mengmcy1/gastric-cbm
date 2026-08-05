"""Merge independent-validation automatic, projection, and manual results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ISSUE_FIELDS = (
    "depth_error_severity",
    "disocclusion_hole_severity",
    "coverage_edge_failure_severity",
    "thin_structure_failure_severity",
    "reflection_deformation_severity",
    "temporal_flicker_severity",
    "floating_gaussian_severity",
    "stretching_severity",
    "paper_feel_severity",
    "occlusion_error_severity",
)
MANUAL_REQUIRED_FIELDS = (
    "review_status",
    "overall_quality_score",
    "overall_pass",
    *ISSUE_FIELDS,
)
ARTIFACT_FRAME_FIELDS = (
    "first_artifact_frame_left",
    "first_artifact_frame_right",
    "worst_frame",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Independent-validation run directory containing the source JSON files.",
    )
    parser.add_argument(
        "--output-stem",
        default="validation_results",
        help="Output filename stem inside --run-root (default: validation_results).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def index_rows(
    rows: list[dict[str, Any]], source_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("validation_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{source_name}: missing validation_id")
        if sample_id in indexed:
            raise ValueError(f"{source_name}: duplicate validation_id {sample_id}")
        indexed[sample_id] = row
    return indexed


def wilson_interval(successes: int, total: int, z: float = 1.9599639845) -> list[float]:
    """Return a two-sided Wilson score interval as fractions."""
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def rate_record(count: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": total,
        "rate": count / total if total else None,
        "wilson_95_ci": wilson_interval(count, total) if total else None,
    }


def validate_manual(sample_id: str, manual: dict[str, Any]) -> None:
    missing = [field for field in MANUAL_REQUIRED_FIELDS if field not in manual]
    if missing:
        raise ValueError(f"{sample_id}: manual result missing fields: {missing}")
    if manual["review_status"] != "completed":
        raise ValueError(
            f"{sample_id}: review_status is {manual['review_status']!r}, expected 'completed'"
        )
    score = manual["overall_quality_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 3:
        raise ValueError(f"{sample_id}: overall_quality_score must be in [0, 3]")
    if not isinstance(manual["overall_pass"], bool):
        raise ValueError(f"{sample_id}: overall_pass must be boolean")
    for field in ISSUE_FIELDS:
        severity = manual[field]
        if isinstance(severity, bool) or not isinstance(severity, int) or not 0 <= severity <= 3:
            raise ValueError(f"{sample_id}: {field} must be an integer in [0, 3]")


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = len(rows)
    pass_count = sum(bool(row["overall_pass"]) for row in rows)
    scores = [float(row["overall_quality_score"]) for row in rows]
    score_counts = Counter(str(int(score)) if score.is_integer() else str(score) for score in scores)

    issue_statistics: dict[str, Any] = {}
    for field in ISSUE_FIELDS:
        severities = [int(row[field]) for row in rows]
        occurrence_count = sum(value > 0 for value in severities)
        obvious_count = sum(value >= 2 for value in severities)
        issue_statistics[field] = {
            "severity_distribution": {
                str(level): severities.count(level) for level in range(4)
            },
            "occurrence_severity_ge_1": rate_record(occurrence_count, reviewed),
            "obvious_or_worse_severity_ge_2": rate_record(obvious_count, reviewed),
        }

    any_issue_count = sum(
        any(int(row[field]) > 0 for field in ISSUE_FIELDS) for row in rows
    )
    any_obvious_issue_count = sum(
        any(int(row[field]) >= 2 for field in ISSUE_FIELDS) for row in rows
    )
    return {
        "sample_count": reviewed,
        "reviewed_count": reviewed,
        "pass_count": pass_count,
        "failure_count": reviewed - pass_count,
        "visual_success": rate_record(pass_count, reviewed),
        "overall_quality_score": {
            "median": median(scores),
            "minimum": min(scores),
            "maximum": max(scores),
            "distribution": dict(sorted(score_counts.items())),
        },
        "any_issue_occurrence": rate_record(any_issue_count, reviewed),
        "any_obvious_or_worse_issue": rate_record(any_obvious_issue_count, reviewed),
        "issue_statistics": issue_statistics,
        "rate_definition": (
            "Issue occurrence means severity >= 1; obvious-or-worse means severity >= 2. "
            "All rates use completed manual reviews as the denominator."
        ),
        "confidence_interval_definition": (
            "Two-sided 95% Wilson score interval; descriptive uncertainty for this "
            "13-sample parameter-tuning-independent set, not a population guarantee."
        ),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    automatic_path = run_root / "automatic_results.json"
    projection_path = run_root / "projection_results.json"
    automatic = load_json(automatic_path)
    projection = load_json(projection_path)

    automatic_rows = index_rows(automatic["results"], "automatic_results.json")
    projection_rows = index_rows(projection["rows"], "projection_results.json")
    if set(automatic_rows) != set(projection_rows):
        raise ValueError(
            "automatic/projection validation_id mismatch: "
            f"automatic_only={sorted(set(automatic_rows) - set(projection_rows))}, "
            f"projection_only={sorted(set(projection_rows) - set(automatic_rows))}"
        )
    if automatic.get("failed") != 0 or automatic.get("completed") != len(automatic_rows):
        raise ValueError("automatic_results.json is not a complete zero-failure run")

    merged_rows: list[dict[str, Any]] = []
    manual_sources: list[str] = []
    for sample_id in sorted(automatic_rows):
        manual_path = run_root / sample_id / "render" / "video_manual.json"
        manual = load_json(manual_path)
        validate_manual(sample_id, manual)
        manual_sources.append(str(manual_path.relative_to(run_root)))

        auto_row = dict(automatic_rows[sample_id])
        auto_row.pop("manual_review_status", None)
        projection_row = dict(projection_rows[sample_id])
        projection_row.pop("validation_id", None)
        merged_rows.append(
            {
                **auto_row,
                **projection_row,
                **{field: manual.get(field, "") for field in MANUAL_REQUIRED_FIELDS},
                **{field: manual.get(field, "") for field in ARTIFACT_FRAME_FIELDS},
                "notes": manual.get("notes", ""),
            }
        )

    csv_path = run_root / f"{args.output_stem}.csv"
    json_path = run_root / f"{args.output_stem}.json"
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有验证汇总：{path}")

    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)

    result = {
        "schema_version": "1.0-validation-summary",
        "run_id": run_root.name,
        "sources": {
            "automatic_results": automatic_path.name,
            "projection_results": projection_path.name,
            "manual_results": manual_sources,
        },
        "summary": build_summary(merged_rows),
        "rows": merged_rows,
    }
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
