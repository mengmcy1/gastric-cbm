#!/usr/bin/env python3
"""从冻结Y0-F train/val视图生成CD1三通道灰度PNG数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "数据整理记录/CADe_CD1_Gray_20260825"
EXPECTED_COUNTS = {"train": 2350, "val": 497}
EXPECTED_PATIENTS = {"train": 1212, "val": 260}


def file_sha256(path: Path) -> str:
    """计算单个文件SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """解析冻结源视图和新Gray输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def prepare_gray_dataset(
    source_root: Path,
    output_root: Path,
    strict_counts: bool = True,
) -> dict:
    """生成Gray图像、复制YOLO标签并写出一一映射审计。"""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"CD1 Gray输出已存在，拒绝覆盖: {output_root}")

    mapping_path = source_root / "y0f_mapping.csv"
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig", dtype={"patient_id": str})
    selected = mapping[mapping.split.isin(("train", "val"))].copy().reset_index(drop=True)
    counts = selected.groupby("split").size().to_dict()
    patients = selected.groupby("split").patient_id.nunique().to_dict()
    if strict_counts and (counts != EXPECTED_COUNTS or patients != EXPECTED_PATIENTS):
        raise ValueError(f"Y0-F train/val规模异常: images={counts}, patients={patients}")
    if set(selected.split) != {"train", "val"}:
        raise ValueError("CD1 Gray只允许train和val两个split")

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=False)

    records = []
    for row in selected.itertuples(index=False):
        source_image = source_root / row.yolo_image_relpath
        source_label = source_root / row.yolo_label_relpath
        source_digest = file_sha256(source_image)
        if source_digest != row.sha256:
            raise ValueError(f"源图SHA与Y0-F mapping不一致: {source_image}")

        gray_relpath = Path("images") / row.split / f"{Path(row.yolo_image_relpath).stem}.png"
        gray_path = output_root / gray_relpath
        gray = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if gray is None:
            raise ValueError(f"无法解码源图: {source_image}")
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if not cv2.imwrite(str(gray_path), gray3):
            raise OSError(f"Gray PNG写入失败: {gray_path}")
        decoded = cv2.imread(str(gray_path), cv2.IMREAD_COLOR)
        if decoded is None or not (
            (decoded[:, :, 0] == decoded[:, :, 1]).all()
            and (decoded[:, :, 1] == decoded[:, :, 2]).all()
        ):
            raise ValueError(f"Gray三通道审计失败: {gray_path}")

        gray_label_relpath = Path("labels") / row.split / f"{gray_path.stem}.txt"
        gray_label = output_root / gray_label_relpath
        shutil.copy2(source_label, gray_label)
        if file_sha256(source_label) != file_sha256(gray_label):
            raise ValueError(f"YOLO标签复制后不一致: {source_label}")

        records.append({
            "source_relpath": row.image_relpath,
            "source_yolo_relpath": row.yolo_image_relpath,
            "gray_relpath": gray_relpath.as_posix(),
            "source_label_relpath": row.yolo_label_relpath,
            "gray_label_relpath": gray_label_relpath.as_posix(),
            "source_sha256": source_digest,
            "gray_sha256": file_sha256(gray_path),
            "split": row.split,
            "patient_id": row.patient_id,
            "label": int(row.label),
            "bbox_xyxy": row.bbox_xyxy,
        })

    manifest = pd.DataFrame(records)
    if len(manifest) != len(selected) or manifest.gray_relpath.nunique() != len(selected):
        raise ValueError("CD1 Gray源图与输出图不是一一映射")
    if set(manifest.split) != {"train", "val"}:
        raise ValueError("CD1 Gray输出意外包含非train/val路径")

    manifest_path = output_root / "gray_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    data_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "early_cancer_or_HGD"},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    config = {
        "stage": "CD1",
        "role": "gray_train_val_view",
        "source_root": str(source_root),
        "source_mapping": str(mapping_path),
        "source_mapping_sha256": file_sha256(mapping_path),
        "gray_manifest": str(manifest_path),
        "gray_manifest_sha256": file_sha256(manifest_path),
        "conversion": "cv2.imread(BGR)->BGR2GRAY->GRAY2BGR->PNG",
        "images": {key: int(value) for key, value in counts.items()},
        "patients": {key: int(value) for key, value in patients.items()},
        "test_exported": False,
        "external_exported": False,
        "channel_equality_verified": True,
    }
    (output_root / "gray_data_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def main() -> None:
    """执行正式Gray train/val数据生成，不读取test或external。"""
    args = parse_args()
    config = prepare_gray_dataset(args.source_root, args.output)
    print(
        f"CD1 Gray数据完成: train={config['images']['train']}, "
        f"val={config['images']['val']} -> {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
