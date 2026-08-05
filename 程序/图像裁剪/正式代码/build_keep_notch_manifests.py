#!/usr/bin/env python3
"""Build paired Keep/Notch manifests from a frozen base and reviewed Notch set."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建同图配对的 Keep/Notch manifest，不复制未变化图片。"
    )
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--notch-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normal-after-notch-list",
        type=Path,
        default=None,
        help="Notch后按普通图片分组的特例清单。",
    )
    parser.add_argument(
        "--box-qc-review",
        type=Path,
        default=None,
        help="人工框QC复核表；所有行必须为accepted。",
    )
    parser.add_argument(
        "--base-key-column",
        default="relative_path",
        help="母清单的图片主键列。",
    )
    parser.add_argument(
        "--base-path-column",
        default="masked",
        help="母清单中正式 Keep 图像路径列，默认为 FOV mapping 的 masked。",
    )
    return parser.parse_args()


def validate_args(args):
    paths = [args.base_mapping, args.notch_mapping]
    if args.normal_after_notch_list:
        paths.append(args.normal_after_notch_list)
    if args.box_qc_review:
        paths.append(args.box_qc_review)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"输入不存在：{path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def index_unique(rows, key, source_name):
    indexed = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            raise ValueError(f"{source_name} 存在空 {key}")
        if value in indexed:
            raise ValueError(f"{source_name} 存在重复 {key}：{value}")
        indexed[value] = row
    return indexed


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_shape(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图像无法解码：{path}")
    return image.shape[1], image.shape[0]


def patient_key(relative_path):
    parent = Path(relative_path).parent
    return parent.as_posix() if str(parent) != "." else "."


def geometry_id(relative_path, keep_sha256, width, height):
    payload = f"{relative_path}\n{keep_sha256}\n{width}x{height}\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def branch_row(pair, branch):
    prefix = branch.lower()
    return {
        "relative_path": pair["relative_path"],
        "patient_id": pair["patient_id"],
        "geometry_id": pair["geometry_id"],
        "pip_present": pair["pip_present"],
        "pip_present_original": pair["pip_present_original"],
        "analysis_group": pair[f"{prefix}_analysis_group"],
        "branch": branch,
        "input_path": pair[f"{prefix}_path"],
        "sha256": pair[f"{prefix}_sha256"],
        "width": pair["width"],
        "height": pair["height"],
    }


def main():
    args = parse_args()
    validate_args(args)
    base_rows = load_csv(args.base_mapping)
    notch_rows = load_csv(args.notch_mapping)
    if not base_rows:
        raise ValueError("母清单为空")
    if args.base_path_column not in base_rows[0]:
        raise ValueError(
            f"母清单缺少路径列：{args.base_path_column}"
        )

    if args.base_key_column not in base_rows[0]:
        raise ValueError(
            f"母清单缺少主键列：{args.base_key_column}"
        )
    base_by_path = index_unique(
        base_rows, args.base_key_column, "base mapping"
    )
    notch_by_path = index_unique(notch_rows, "relative_path", "notch mapping")
    unknown = sorted(set(notch_by_path) - set(base_by_path))
    if unknown:
        raise ValueError(f"Notch 包含母清单外图片：{unknown[:3]}")

    normal_after_notch = set()
    if args.normal_after_notch_list:
        decision_rows = load_csv(args.normal_after_notch_list)
        decision_by_path = index_unique(
            decision_rows, "relative_path", "normal-after-notch list"
        )
        normal_after_notch = {
            path for path, row in decision_by_path.items()
            if row["notch_analysis_group"]
            == "normal_after_external_pip_removal"
        }
        unknown_decisions = sorted(normal_after_notch - set(notch_by_path))
        if unknown_decisions:
            raise ValueError(
                "Notch后普通图片清单包含非Notch图片："
                f"{unknown_decisions[:3]}"
            )

    qc_review_count = 0
    if args.box_qc_review:
        qc_rows = load_csv(args.box_qc_review)
        index_unique(qc_rows, "relative_path", "box QC review")
        rejected = [
            row["relative_path"] for row in qc_rows
            if row.get("human_box_status") != "accepted"
        ]
        if rejected:
            raise ValueError(f"框QC存在未通过项：{rejected[:3]}")
        qc_review_count = len(qc_rows)

    pairs = []
    for relative_path in sorted(base_by_path):
        base = base_by_path[relative_path]
        notch = notch_by_path.get(relative_path)
        keep_path = Path(base[args.base_path_column])
        notch_path = Path(notch["notch_output"]) if notch else keep_path
        for path in (keep_path, notch_path):
            if not path.is_file():
                raise FileNotFoundError(f"配对图像不存在：{path}")

        keep_width, keep_height = read_shape(keep_path)
        notch_width, notch_height = read_shape(notch_path)
        dimensions_match = (
            keep_width == notch_width and keep_height == notch_height
        )
        if not dimensions_match:
            raise ValueError(f"Keep/Notch 尺寸不一致：{relative_path}")

        keep_sha256 = file_sha256(keep_path)
        notch_sha256 = file_sha256(notch_path)
        pip_present = "yes" if notch else "no"
        notch_analysis_group = (
            "normal_after_external_pip_removal"
            if relative_path in normal_after_notch
            else ("pip_notch" if notch else "normal")
        )
        pairs.append({
            "relative_path": relative_path,
            "patient_id": (
                notch["patient_id"] if notch else patient_key(relative_path)
            ),
            "geometry_id": geometry_id(
                relative_path, keep_sha256, keep_width, keep_height
            ),
            "pip_present": pip_present,
            "pip_present_original": pip_present,
            "keep_analysis_group": "pip_keep" if notch else "normal",
            "notch_analysis_group": notch_analysis_group,
            "keep_path": str(keep_path),
            "notch_path": str(notch_path),
            "keep_sha256": keep_sha256,
            "notch_sha256": notch_sha256,
            "width": keep_width,
            "height": keep_height,
            "dimensions_match": "yes",
            "same_file_reference": "no" if notch else "yes",
            "notch_cut_x": notch["cut_x"] if notch else "",
            "notch_cut_y": notch["cut_y"] if notch else "",
            "coordinate_transform_keep_to_notch": "identity",
        })

    args.output.mkdir(parents=True, exist_ok=False)
    pair_fields = list(pairs[0])
    branch_fields = [
        "relative_path", "patient_id", "geometry_id", "pip_present",
        "pip_present_original", "analysis_group", "branch", "input_path",
        "sha256", "width", "height",
    ]
    paired_path = args.output / "paired_manifest.csv"
    keep_path = args.output / "keep_manifest.csv"
    notch_path = args.output / "notch_manifest.csv"
    write_csv(paired_path, pairs, pair_fields)
    write_csv(keep_path, [branch_row(row, "keep") for row in pairs], branch_fields)
    write_csv(notch_path, [branch_row(row, "notch") for row in pairs], branch_fields)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": (
            "paired_reviewed_ready_for_m0"
            if args.box_qc_review
            else "paired_candidate_pending_final_human_acceptance"
        ),
        "script": str(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "base_mapping": str(args.base_mapping),
        "base_mapping_sha256": file_sha256(args.base_mapping),
        "notch_mapping": str(args.notch_mapping),
        "notch_mapping_sha256": file_sha256(args.notch_mapping),
        "base_path_column": args.base_path_column,
        "base_key_column": args.base_key_column,
        "normal_after_notch_list": (
            str(args.normal_after_notch_list)
            if args.normal_after_notch_list else ""
        ),
        "normal_after_notch_list_sha256": (
            file_sha256(args.normal_after_notch_list)
            if args.normal_after_notch_list else None
        ),
        "box_qc_review": (
            str(args.box_qc_review) if args.box_qc_review else ""
        ),
        "box_qc_review_sha256": (
            file_sha256(args.box_qc_review) if args.box_qc_review else None
        ),
        "box_qc_review_count": qc_review_count,
        "image_count_per_branch": len(pairs),
        "pip_image_count": len(notch_by_path),
        "shared_file_count": sum(
            row["same_file_reference"] == "yes" for row in pairs
        ),
        "dimension_mismatch_count": 0,
        "notch_analysis_group_counts": dict(Counter(
            row["notch_analysis_group"] for row in pairs
        )),
        "paired_manifest": str(paired_path),
        "keep_manifest": str(keep_path),
        "notch_manifest": str(notch_path),
    }
    with (args.output / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
