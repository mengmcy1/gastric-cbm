"""Validate and summarize the formal SHARP pruning matrix.

The summary combines frozen experiment configuration, automatic zero-eye metrics,
direct comparisons against keep100, and user-confirmed manual video reviews.
It never changes the frozen experiment configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRUNE_ROOT = EXPERIMENT_ROOT / "outputs" / "06_剪枝实验_crop03正式"
SAMPLE_IDS = ("P01", "P02", "P03", "P04", "P05")
KEEP_PERCENTS = (100, 75, 50, 25)
SEVERITY_FIELDS = (
    "hole_severity",
    "floating_gaussian_severity",
    "depth_layering_loss_severity",
    "stretching_severity",
    "flicker_severity",
    "paper_feel_severity",
    "occlusion_error_severity",
    "reflection_deformation_severity",
)
FRAME_FIELDS = (
    "first_artifact_frame_left",
    "first_artifact_frame_right",
    "worst_frame",
)
SSIM_MAX_SIDE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prune-root", type=Path, default=DEFAULT_PRUNE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_score(value: Any, field: str, label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label}: {field} 不能是布尔值")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} 必须是 0～3 的整数") from exc
    if result not in (0, 1, 2, 3):
        raise ValueError(f"{label}: {field} 必须在 0～3，实际为 {result}")
    return result


def parse_frame(value: Any, field: str, label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label}: {field} 不能是布尔值")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} 必须是 0～59 的帧号或留空") from exc
    if not 0 <= result <= 59:
        raise ValueError(f"{label}: {field} 必须在 0～59，实际为 {result}")
    return result


def direct_metrics(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = np.asarray(Image.open(reference_path).convert("RGB"))
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"))
    if reference.shape != candidate.shape:
        raise ValueError(
            f"相对 keep100 比较尺寸不一致：{reference_path}={reference.shape}, "
            f"{candidate_path}={candidate.shape}"
        )
    difference = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(difference * difference))
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255.0**2) / mse))
    height, width = reference.shape[:2]
    scale = min(1.0, SSIM_MAX_SIDE / max(height, width))
    ssim_size = (round(width * scale), round(height * scale))
    if scale < 1.0:
        reference_ssim = np.asarray(
            Image.fromarray(reference).resize(ssim_size, Image.Resampling.BICUBIC)
        )
        candidate_ssim = np.asarray(
            Image.fromarray(candidate).resize(ssim_size, Image.Resampling.BICUBIC)
        )
    else:
        reference_ssim = reference
        candidate_ssim = candidate
    ssim = float(
        structural_similarity(
            reference_ssim, candidate_ssim, channel_axis=2, data_range=255
        )
    )
    return {
        "vs_keep100_psnr_db": None if not np.isfinite(psnr) else round(psnr, 6),
        "vs_keep100_psnr_is_infinite": bool(np.isinf(psnr)),
        "vs_keep100_ssim": round(ssim, 6),
        "vs_keep100_ssim_width": int(reference_ssim.shape[1]),
        "vs_keep100_ssim_height": int(reference_ssim.shape[0]),
    }


def build_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    recommended: int | None = None
    for keep in sorted(KEEP_PERCENTS):
        level_rows = [row for row in rows if row["keep_percent"] == keep]
        scores = [row["overall_quality_score"] for row in level_rows]
        failed = [
            row["sample_id"]
            for row in level_rows
            if row["overall_quality_score"] is None
            or row["overall_quality_score"] < 2
        ]
        evidence[f"keep{keep:03d}"] = {
            "scores": scores,
            "minimum_score": min(scores) if scores and None not in scores else None,
            "failed_samples": failed,
        }
        if recommended is None and len(level_rows) == len(SAMPLE_IDS) and not failed:
            recommended = keep
    if recommended is None:
        return {
            "status": "no_common_passing_keep_percent",
            "recommended_keep_percent": None,
            "evidence": evidence,
            "requires_user_confirmation": True,
        }
    return {
        "status": "candidate",
        "recommended_keep_percent": recommended,
        "reason": f"keep{recommended:03d} 是五样本整体质量分全部不低于 2 的最低保留率。",
        "evidence": evidence,
        "requires_user_confirmation": True,
    }


def main() -> None:
    args = parse_args()
    prune_root = args.prune_root.resolve()
    if not prune_root.is_dir():
        raise FileNotFoundError(f"缺少正式剪枝结果目录：{prune_root}")

    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    warnings: list[str] = []

    for sample_id in SAMPLE_IDS:
        keep100_frame = prune_root / sample_id / "keep100" / "frame_center_zero.png"
        if not keep100_frame.is_file():
            raise FileNotFoundError(f"缺少 keep100 零位移帧：{keep100_frame}")
        for keep in KEEP_PERCENTS:
            config_name = f"keep{keep:03d}"
            config_dir = prune_root / sample_id / config_name
            label = f"{sample_id}/{config_name}"
            required = {
                "config": config_dir / "config.json",
                "prune_stats": config_dir / "prune_stats.json",
                "static_metrics": config_dir / "static_metrics.json",
                "manual": config_dir / "video_manual.json",
                "center": config_dir / "frame_center_zero.png",
                "color_video": config_dir / "color.mp4",
                "depth_video": config_dir / "depth.mp4",
                "per_frame": config_dir / "per_frame_metrics.csv",
            }
            missing_files = [name for name, path in required.items() if not path.is_file()]
            if missing_files:
                incomplete.append({"config": label, "missing_files": missing_files})
                continue

            config = load_json(required["config"])
            prune_stats = load_json(required["prune_stats"])
            static_metrics = load_json(required["static_metrics"])
            manual = load_json(required["manual"])
            missing_fields: list[str] = []

            expected_config = {
                "keep_percent": keep,
                "crop_single_side_percent": 3,
                "max_disparity": 0.04,
                "trajectory": "swipe",
                "num_steps": 60,
                "static_metric_pose": "exact_zero_eye",
            }
            for field, expected in expected_config.items():
                if config.get(field) != expected:
                    raise ValueError(
                        f"{label}: config.{field}={config.get(field)!r}，期望 {expected!r}"
                    )
            if config.get("encoded_color_video", {}).get("frame_count") != 60:
                raise ValueError(f"{label}: 彩色视频帧数不是 60")
            if config.get("encoded_depth_video", {}).get("frame_count") != 60:
                raise ValueError(f"{label}: 深度视频帧数不是 60")
            if config.get("encoded_color_video", {}).get("resolution") != config.get(
                "render_resolution"
            ):
                raise ValueError(f"{label}: 彩色视频编码尺寸与渲染尺寸不一致")
            if config.get("encoded_depth_video", {}).get("resolution") != config.get(
                "render_resolution"
            ):
                raise ValueError(f"{label}: 深度视频编码尺寸与渲染尺寸不一致")
            if prune_stats.get("keep_percent") != keep:
                raise ValueError(f"{label}: prune_stats.keep_percent 不匹配")
            if prune_stats.get("n_kept") != config.get("n_gaussians_kept"):
                raise ValueError(f"{label}: 剪枝高斯数在 config 与 prune_stats 中不一致")

            if manual.get("schema_version") != "2.1-prune":
                missing_fields.append("schema_version=2.1-prune")
            if manual.get("review_status") != "completed":
                missing_fields.append("review_status=completed")
            overall = parse_score(
                manual.get("overall_quality_score"), "overall_quality_score", label
            )
            if overall is None:
                missing_fields.append("overall_quality_score")
            severities: dict[str, int | None] = {}
            for field in SEVERITY_FIELDS:
                severities[field] = parse_score(manual.get(field), field, label)
                if severities[field] is None:
                    missing_fields.append(field)
            frames = {
                field: parse_frame(manual.get(field), field, label) for field in FRAME_FIELDS
            }
            reported_pass = manual.get("overall_pass")
            if not isinstance(reported_pass, bool):
                missing_fields.append("overall_pass")
                reported_pass = None
            derived_pass = overall >= 2 if overall is not None else None
            if reported_pass is not None and reported_pass != derived_pass:
                warnings.append(
                    f"{label}: overall_pass={reported_pass} 与整体质量分 {overall} 不一致"
                )
            if missing_fields:
                incomplete.append({"config": label, "missing_fields": missing_fields})

            relative = direct_metrics(keep100_frame, required["center"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "config_name": config_name,
                    "keep_percent": keep,
                    "n_gaussians_original": config["n_gaussians_original"],
                    "n_gaussians_kept": config["n_gaussians_kept"],
                    "crop_single_side_percent": config["crop_single_side_percent"],
                    "max_disparity": config["max_disparity"],
                    "trajectory": config["trajectory"],
                    "num_steps": config["num_steps"],
                    "render_width": config["render_resolution"][0],
                    "render_height": config["render_resolution"][1],
                    "vs_input_psnr_db": static_metrics.get("psnr_db"),
                    "vs_input_ssim": static_metrics.get("ssim"),
                    "vs_input_lpips_alex": static_metrics.get("lpips_alex"),
                    **relative,
                    "review_status": manual.get("review_status", ""),
                    "overall_quality_score": overall,
                    **severities,
                    **frames,
                    "overall_pass_reported": reported_pass,
                    "overall_pass_derived": derived_pass,
                    "notes": manual.get("notes", ""),
                    "review_path": str(required["manual"].relative_to(prune_root)).replace(
                        "\\", "/"
                    ),
                }
            )

    expected = len(SAMPLE_IDS) * len(KEEP_PERCENTS)
    completeness = {
        "expected_configs": expected,
        "loaded_configs": len(rows),
        "complete_configs": expected - len(incomplete),
        "incomplete_configs": incomplete,
        "warnings": warnings,
    }
    print(json.dumps(completeness, ensure_ascii=False, indent=2))
    if args.check_only:
        return
    if incomplete and not args.allow_incomplete:
        raise SystemExit("人工评分或产物尚未完整；请先使用 --check-only 查看。")

    level_summary: dict[str, Any] = {}
    for keep in KEEP_PERCENTS:
        level_rows = [row for row in rows if row["keep_percent"] == keep]
        scores = [row["overall_quality_score"] for row in level_rows]
        finite_psnr = [
            row["vs_keep100_psnr_db"]
            for row in level_rows
            if row["vs_keep100_psnr_db"] is not None
        ]
        level_summary[f"keep{keep:03d}"] = {
            "sample_count": len(level_rows),
            "overall_scores": scores,
            "minimum_overall_quality_score": min(scores) if scores else None,
            "passing_samples": sum(score >= 2 for score in scores),
            "vs_keep100_psnr_db_min": min(finite_psnr) if finite_psnr else None,
            "vs_keep100_psnr_db_median": (
                statistics.median(finite_psnr) if finite_psnr else None
            ),
            "vs_keep100_ssim_min": min(
                row["vs_keep100_ssim"] for row in level_rows
            )
            if level_rows
            else None,
        }

    summary = {
        "schema_version": "1.0-prune-summary",
        "experiment_root": str(prune_root),
        "fixed_variables": {
            "max_disparity": 0.04,
            "crop_single_side_percent": 3,
            "trajectory": "swipe",
            "num_steps": 60,
            "pruning_method": "opacity_descending",
        },
        "metric_definitions": {
            "vs_input": "cropped strict zero-eye render compared with original input",
            "vs_keep100_psnr": (
                "same-sample cropped strict zero-eye render compared directly with keep100 "
                "at full resolution"
            ),
            "vs_keep100_ssim": (
                "same comparison resized with bicubic interpolation to maximum side 512 "
                "before SSIM; exact metric width and height are stored per row"
            ),
            "manual": "user-confirmed full-video absolute quality review",
        },
        "completeness": completeness,
        "level_summary": level_summary,
        "freeze_recommendation": build_recommendation(rows),
        "decision_boundary": (
            "Candidate recommendation only. Do not update experiment_defaults.json or "
            "start normalized performance testing until the user confirms the keep percent."
        ),
    }

    csv_path = prune_root / "prune_results.csv"
    json_path = prune_root / "prune_results.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(
            f"拒绝覆盖已有汇总：{', '.join(path.name for path in existing)}"
        )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
