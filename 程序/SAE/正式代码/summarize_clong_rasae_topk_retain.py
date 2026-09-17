#!/usr/bin/env python3
"""仅从已有Top-k逐图/患者CSV补齐恢复曲线与图像标签分层汇总。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_clong_rasae_topk_retain import summarize_curve, summarize_image_labels


def main() -> None:
    """读取已有CSV并覆盖同一结果目录中的汇总文件，不调用模型。

    Args:
        None: 结果目录由CLI提供。

    Returns:
        None: 更新``recovery_curve.csv``、``image_label_summary.csv``、summary和verification。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    images = pd.read_csv(args.result_dir / "image_results.csv")
    patients = pd.read_csv(args.result_dir / "patient_results.csv")
    curve = summarize_curve(images, patients)
    image_labels = summarize_image_labels(images)
    curve.to_csv(args.result_dir / "recovery_curve.csv", index=False)
    image_labels.to_csv(args.result_dir / "image_label_summary.csv", index=False)
    summary_path = args.result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["curve"] = curve.to_dict(orient="records")
    summary["image_label_summary"] = image_labels.to_dict(orient="records")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    verification_path = args.result_dir / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["summary_regenerated_from_existing_csv"] = True
    verification["model_rerun_for_summary"] = False
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "levels": curve.level.tolist(), "image_label_rows": len(image_labels),
        "model_rerun": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
