#!/usr/bin/env python3
"""Build paired Keep/Notch manifests from a frozen base and reviewed Notch set."""

import argparse
import csv
import hashlib
import json
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
        "--base-path-column",
        default="masked",
        help="母清单中正式 Keep 图像路径列，默认为 FOV mapping 的 masked。",
    )
    return parser.parse_args()


def validate_args(args):
    for path in (args.base_mapping, args.notch_mapping):
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

    base_by_path = index_unique(base_rows, "relative_path", "base mapping")
    notch_by_path = index_unique(notch_rows, "relative_path", "notch mapping")
    unknown = sorted(set(notch_by_path) - set(base_by_path))
    if unknown:
        raise ValueError(f"Notch 包含母清单外图片：{unknown[:3]}")

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
        pairs.append({
            "relative_path": relative_path,
            "patient_id": (
                notch["patient_id"] if notch else patient_key(relative_path)
            ),
            "geometry_id": geometry_id(
                relative_path, keep_sha256, keep_width, keep_height
            ),
            "pip_present": "yes" if notch else "no",
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
        "relative_path", "patient_id", "geometry_id", "pip_present", "branch",
        "input_path", "sha256", "width", "height",
    ]
    paired_path = args.output / "paired_manifest.csv"
    keep_path = args.output / "keep_manifest.csv"
    notch_path = args.output / "notch_manifest.csv"
    write_csv(paired_path, pairs, pair_fields)
    write_csv(keep_path, [branch_row(row, "keep") for row in pairs], branch_fields)
    write_csv(notch_path, [branch_row(row, "notch") for row in pairs], branch_fields)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "paired_candidate_pending_final_human_acceptance",
        "script": str(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "base_mapping": str(args.base_mapping),
        "base_mapping_sha256": file_sha256(args.base_mapping),
        "notch_mapping": str(args.notch_mapping),
        "notch_mapping_sha256": file_sha256(args.notch_mapping),
        "base_path_column": args.base_path_column,
        "image_count_per_branch": len(pairs),
        "pip_image_count": len(notch_by_path),
        "shared_file_count": sum(
            row["same_file_reference"] == "yes" for row in pairs
        ),
        "dimension_mismatch_count": 0,
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
