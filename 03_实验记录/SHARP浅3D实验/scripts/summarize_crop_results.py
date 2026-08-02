"""SHARP 裁剪实验汇总脚本（正式，可复现）

读取 outputs/04_裁剪实验/ 中全部 video_manual.json 和 crop_metrics.json，
生成 crop_results.csv 并执行完整性检查。

用法：
    python scripts/summarize_crop_results.py            # 生成 CSV，报告缺失字段
    python scripts/summarize_crop_results.py --check-only  # 只检查，不覆盖 CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CROP_BASE = REPO_ROOT / "outputs" / "04_裁剪实验"
SAMPLES = ("P01", "P02", "P03", "P04", "P05")
CROP_LEVELS = ("crop00", "crop03", "crop05", "crop10")

# 必须填写的字段（null=未评估 可接受，""=空字符串 视为缺失）
REQUIRED_FIELDS = (
    "schema_version",
    "score_interpretation",
    "review_status",
    "overall_quality_score",
    "overall_pass",
    "edge_artifact_reduction_score",
    "hole_severity",
    "sharpness_loss_severity",
    "framing_loss_severity",
)

CSV_COLUMNS = (
    "sample",
    "crop_level",
    "score_interpretation",
    "overall_quality_score",
    "overall_pass",
    "edge_artifact_reduction_score",
    "hole_severity",
    "sharpness_loss_severity",
    "framing_loss_severity",
    "stretching_severity",
    "flicker_severity",
    "paper_feel_severity",
    "occlusion_error_severity",
    "reflection_deformation_severity",
    "first_artifact_frame_left",
    "first_artifact_frame_right",
    "worst_frame",
    "review_status",
    "psnr_db",
    "ssim",
    "lpips_alex",
    "notes",
)


def check_field(val) -> bool:
    """空字符串视为缺失；None/null 视为未评估（可接受）；有值则通过。"""
    return val != ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="只报告完整性，不写入 CSV")
    args = parser.parse_args()

    missing_report: list[str] = []
    rows: list[dict] = []

    for sample in SAMPLES:
        for crop in CROP_LEVELS:
            manual_p = CROP_BASE / sample / crop / "video_manual.json"
            metrics_p = CROP_BASE / sample / crop / "crop_metrics.json"

            if not manual_p.is_file():
                print(f"[ERROR] 缺少文件：{manual_p.relative_to(REPO_ROOT)}")
                return 1
            if not metrics_p.is_file():
                print(f"[ERROR] 缺少文件：{metrics_p.relative_to(REPO_ROOT)}")
                return 1

            manual = json.loads(manual_p.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_p.read_text(encoding="utf-8"))

            for fld in REQUIRED_FIELDS:
                if not check_field(manual.get(fld)):
                    missing_report.append(f"  {sample}/{crop}: {fld} = {manual.get(fld)!r}")

            psnr_val = "inf" if metrics.get("psnr_is_infinite") else metrics.get("psnr_db", "")

            row: dict = {"sample": sample, "crop_level": crop, "psnr_db": psnr_val,
                         "ssim": metrics.get("ssim", ""), "lpips_alex": metrics.get("lpips_alex", "")}
            for col in CSV_COLUMNS:
                if col not in row:
                    row[col] = manual.get(col, "")
            rows.append(row)

    # ── 完整性报告 ─────────────────────────────────────────────────────────
    if missing_report:
        print(f"\n[WARNING] 以下 {len(missing_report)} 个必填字段为空字符串（需补填）：")
        for line in missing_report:
            print(line)
    else:
        print("\n[OK] 全部 REQUIRED_FIELDS 均已填写（null 视为未评估，已接受）。")

    print(f"\n共 {len(rows)} 行，样本：{len(SAMPLES)}，档位：{len(CROP_LEVELS)}")

    # ── 写入 CSV ──────────────────────────────────────────────────────────
    if args.check_only:
        print("[INFO] --check-only 模式，不写入 CSV。")
        return 1 if missing_report else 0

    csv_path = CROP_BASE / "crop_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] 写入：{csv_path.relative_to(REPO_ROOT)}")

    return 1 if missing_report else 0


if __name__ == "__main__":
    sys.exit(main())
