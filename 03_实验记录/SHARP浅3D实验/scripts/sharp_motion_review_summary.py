"""汇总 SHARP 五样本四档运动视频的人工评分与自动位移指标。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_ROOT = EXPERIMENT_ROOT / "outputs" / "03_运动实验"
SAMPLE_IDS = ("P01", "P02", "P03", "P04", "P05")
LEVELS = {
    "md000": 0.0,
    "md002": 0.02,
    "md004": 0.04,
    "md008": 0.08,
}
CONFIG_NAMES = {
    level: f"{level}_swipe60_keep100_crop00" for level in LEVELS
}
SEVERITY_FIELDS = (
    "hole_severity",
    "stretching_severity",
    "flicker_severity",
    "paper_feel_severity",
    "occlusion_error_severity",
    "reflection_deformation_severity",
)
OPTIONAL_FRAME_FIELDS = (
    "first_artifact_frame_left",
    "first_artifact_frame_right",
    "worst_frame",
)


def parse_score(value: Any, field: str, label: str) -> int | None:
    if value == "" or value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label}: {field} 不能是布尔值")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} 必须是 0～3 的整数，实际为 {value!r}") from exc
    if score not in (0, 1, 2, 3):
        raise ValueError(f"{label}: {field} 必须在 0～3，实际为 {score}")
    return score


def parse_frame(value: Any, field: str, label: str) -> int | None:
    if value == "" or value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label}: {field} 不能是布尔值")
    try:
        frame = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} 必须是 0～59 的帧号或空字符串") from exc
    if not 0 <= frame <= 59:
        raise ValueError(f"{label}: {field} 必须在 0～59，实际为 {frame}")
    return frame


def normalize_pass(value: Any, label: str) -> bool | None:
    if value == "" or value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "pass", "通过", "1"}:
            return True
        if normalized in {"false", "no", "fail", "不通过", "0"}:
            return False
    raise ValueError(f"{label}: overall_pass 应为布尔值、通过/不通过或空字符串")


def load_motion_results(path: Path) -> dict[tuple[str, float], dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, float], dict] = {}
    for row in rows:
        key = (str(row["sample_id"]), float(row["max_disparity"]))
        if key in result:
            raise ValueError(f"motion_results.json 存在重复项：{key}")
        result[key] = row
    return result


def frame_distribution(rows: list[dict], field: str) -> dict:
    values = [row[field] for row in rows if row[field] is not None]
    if not values:
        return {"count": 0, "values": [], "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "values": values,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_recommendation(rows: list[dict]) -> dict:
    by_level = {
        level: [row for row in rows if row["max_disparity"] == disparity]
        for level, disparity in LEVELS.items()
    }
    md004 = by_level["md004"]
    md002 = by_level["md002"]
    if any(row["overall_quality_score"] is None for row in md004 + md002):
        return {
            "status": "insufficient_manual_scores",
            "recommended_max_disparity": None,
            "reason": "0.02 或 0.04 档仍有整体质量分未填写，暂不能提出冻结建议。",
        }

    md004_failed = [row["sample_id"] for row in md004 if row["overall_quality_score"] < 2]
    md002_failed = [row["sample_id"] for row in md002 if row["overall_quality_score"] < 2]
    stress_failed = [sample for sample in ("P01", "P05") if sample in md004_failed]

    if not md004_failed:
        return {
            "status": "candidate",
            "recommended_max_disparity": 0.04,
            "reason": "五张样本的 0.04 档整体质量分均不低于 2。",
            "requires_user_confirmation": True,
        }
    if not md002_failed:
        qualifier = "P01 或 P05 的 0.04 档未通过" if stress_failed else "0.04 档存在未通过样本"
        return {
            "status": "candidate",
            "recommended_max_disparity": 0.02,
            "alternative": "预先定义适用条件后采用 0.02/0.04 分场景档位",
            "reason": f"{qualifier}，而五张样本的 0.02 档均达到整体质量分 2。",
            "requires_user_confirmation": True,
        }
    return {
        "status": "no_common_passing_level",
        "recommended_max_disparity": None,
        "reason": f"0.04 未通过样本={md004_failed}；0.02 未通过样本={md002_failed}。",
        "requires_user_confirmation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    parser.add_argument("--check-only", action="store_true", help="仅打印完整性检查，不写汇总文件")
    parser.add_argument("--allow-incomplete", action="store_true", help="允许评分未完成时写出阶段性汇总")
    parser.add_argument("--allow-overwrite", action="store_true", help="允许覆盖既有人工汇总 CSV/JSON")
    args = parser.parse_args()

    motion_root = args.motion_root.resolve()
    motion_results_path = motion_root / "motion_results.json"
    if not motion_results_path.is_file():
        raise FileNotFoundError(f"缺少自动汇总：{motion_results_path}")
    automatic = load_motion_results(motion_results_path)

    rows: list[dict] = []
    incomplete: list[dict] = []
    warnings: list[str] = []

    for sample_id in SAMPLE_IDS:
        for level, disparity in LEVELS.items():
            config_name = CONFIG_NAMES[level]
            review_path = motion_root / sample_id / config_name / "video_manual.json"
            label = f"{sample_id}/{config_name}"
            if not review_path.is_file():
                incomplete.append({"path": str(review_path), "missing_fields": ["file"]})
                continue
            data = json.loads(review_path.read_text(encoding="utf-8"))
            missing_fields: list[str] = []
            if data.get("schema_version") != "2.0":
                missing_fields.append("schema_version=2.0")

            overall = parse_score(data.get("overall_quality_score"), "overall_quality_score", label)
            if overall is None:
                missing_fields.append("overall_quality_score")
            severities: dict[str, int | None] = {}
            for field in SEVERITY_FIELDS:
                severities[field] = parse_score(data.get(field), field, label)
                if severities[field] is None:
                    missing_fields.append(field)

            frame_values = {
                field: parse_frame(data.get(field), field, label) for field in OPTIONAL_FRAME_FIELDS
            }
            reported_pass = normalize_pass(data.get("overall_pass"), label)
            derived_pass = overall >= 2 if overall is not None else None
            if reported_pass is not None and derived_pass is not None and reported_pass != derived_pass:
                warnings.append(
                    f"{label}: overall_pass={reported_pass} 与 overall_quality_score={overall} 推导结果不一致"
                )
            if data.get("review_status") not in ("", "in_progress", "completed"):
                warnings.append(f"{label}: review_status 建议使用空、in_progress 或 completed")

            auto = automatic.get((sample_id, disparity))
            if auto is None:
                raise ValueError(f"motion_results.json 缺少 {sample_id} / {disparity}")
            row = {
                "sample_id": sample_id,
                "config_name": config_name,
                "max_disparity": disparity,
                "review_status": data.get("review_status", ""),
                "overall_quality_score": overall,
                **severities,
                **frame_values,
                "worst_frame": frame_values["worst_frame"],
                "overall_pass_reported": reported_pass,
                "overall_pass_derived": derived_pass,
                "notes": data.get("notes", ""),
                "left_p95_px": auto["left_p95_px"],
                "left_p95_width_percent": auto["left_p95_width_percent"],
                "right_p95_px": auto["right_p95_px"],
                "right_p95_width_percent": auto["right_p95_width_percent"],
                "review_path": str(review_path.relative_to(motion_root)).replace("\\", "/"),
            }
            rows.append(row)
            if missing_fields:
                incomplete.append({"path": row["review_path"], "missing_fields": missing_fields})

    expected = len(SAMPLE_IDS) * len(LEVELS)
    if len(rows) + sum(item["missing_fields"] == ["file"] for item in incomplete) != expected:
        raise ValueError(f"期望检查 {expected} 个配置，实际结构数量异常")

    completeness = {
        "expected_reviews": expected,
        "loaded_reviews": len(rows),
        "complete_reviews": expected - len(incomplete),
        "incomplete_reviews": incomplete,
        "warnings": warnings,
    }
    print(json.dumps(completeness, ensure_ascii=False, indent=2))
    if args.check_only:
        return
    if incomplete and not args.allow_incomplete:
        raise SystemExit("人工评分尚未完成；使用 --check-only 查看缺失项，完成后再生成正式汇总。")

    level_summary = {}
    for level, disparity in LEVELS.items():
        level_rows = [row for row in rows if row["max_disparity"] == disparity]
        scores = [row["overall_quality_score"] for row in level_rows if row["overall_quality_score"] is not None]
        level_summary[level] = {
            "max_disparity": disparity,
            "scored_count": len(scores),
            "minimum_overall_quality_score": min(scores) if scores else None,
        }

    md004_rows = [row for row in rows if row["max_disparity"] == 0.04]
    issue_rates_004 = {}
    for field in SEVERITY_FIELDS:
        values = [row[field] for row in md004_rows if row[field] is not None]
        issue_rates_004[field] = {
            "scored_count": len(values),
            "any_issue_rate": sum(value >= 1 for value in values) / len(values) if values else None,
            "obvious_or_severe_rate": sum(value >= 2 for value in values) / len(values) if values else None,
        }

    stress_samples_004 = {
        row["sample_id"]: {
            "overall_quality_score": row["overall_quality_score"],
            "passed": row["overall_quality_score"] >= 2 if row["overall_quality_score"] is not None else None,
            "left_p95_px": row["left_p95_px"],
            "left_p95_width_percent": row["left_p95_width_percent"],
            "right_p95_px": row["right_p95_px"],
            "right_p95_width_percent": row["right_p95_width_percent"],
        }
        for row in md004_rows
        if row["sample_id"] in ("P01", "P05")
    }

    summary = {
        "schema_version": "1.0",
        "completeness": completeness,
        "level_summary": level_summary,
        "md004_issue_rates": issue_rates_004,
        "md004_stress_samples": stress_samples_004,
        "first_artifact_frame_left_distribution": frame_distribution(rows, "first_artifact_frame_left"),
        "first_artifact_frame_right_distribution": frame_distribution(rows, "first_artifact_frame_right"),
        "freeze_recommendation": build_recommendation(rows),
        "decision_boundary": (
            "Recommendation only. Do not freeze parameters or start crop experiments "
            "until the user confirms the recommendation."
        ),
    }

    csv_path = motion_root / "motion_manual_review_summary.csv"
    json_path = motion_root / "motion_manual_review_summary.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(f"拒绝覆盖已有人工汇总：{', '.join(path.name for path in existing)}")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
