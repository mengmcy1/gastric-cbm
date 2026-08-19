#!/usr/bin/env python3
"""从既有外部多中心Keep清单构建癌图病灶框标注交接包。

脚本只读已冻结的1941张外部Keep清单，仅复制标签为癌的542张图。
非癌图无需病灶框，不进入医学生标注包；原始数据和Keep产物均不移动。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = PROJECT_ROOT / (
    "结果/M0全量诊断_0804/外部多中心完整测试_Keep_v1/"
    "external_keep_manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / (
    "数据/标注与汇总/"
    "胃镜多中心外部集_Keep癌图标框交接_20260819"
)


def sha256_file(path: Path) -> str:
    """返回普通文件的SHA256。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_project_path(value: str) -> Path:
    """解析清单中的项目相对路径或绝对路径。"""
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_size(path: Path) -> tuple[int, int]:
    """读取图像并返回宽、高，同时完成解码检查。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码Keep图: {path}")
    height, width = image.shape[:2]
    return width, height


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """以UTF-8 BOM写入新CSV，拒绝覆盖。"""
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def exact_duplicate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """列出交接癌图中Keep SHA完全相同的图对，不自动排除。"""
    rows = []
    for digest, group in frame.groupby("keep_sha256", sort=False):
        items = list(group.itertuples(index=False))
        for left_index, left in enumerate(items):
            for right in items[left_index + 1:]:
                rows.append({
                    "left_annotation_name": left.annotation_image_name,
                    "right_annotation_name": right.annotation_image_name,
                    "left_patient_id": left.patient_id,
                    "right_patient_id": right.patient_id,
                    "same_patient": left.patient_id == right.patient_id,
                    "keep_sha256": digest,
                })
    return pd.DataFrame(rows, columns=[
        "left_annotation_name", "right_annotation_name",
        "left_patient_id", "right_patient_id", "same_patient",
        "keep_sha256",
    ])


def main() -> None:
    """校验上游清单、复制癌图并冻结标注任务血缘。"""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        SOURCE_MANIFEST, low_memory=False, dtype={"patient_id": "string"}
    )
    if len(source) != 1941 or source.patient_id.nunique() != 1329:
        raise ValueError("外部Keep清单不再是冻结的1941张/1329人")
    if source.label.value_counts().to_dict() != {0: 1399, 1: 542}:
        raise ValueError("外部Keep标签分布与冻结记录不一致")
    cancer = source.loc[source.label.eq(1)].copy().reset_index(drop=True)
    if len(cancer) != 542 or cancer.patient_id.nunique() != 121:
        raise ValueError("外部癌图应为542张/121人")
    allowed_status = {"auto_pass_candidate", "pending"}
    if not set(cancer.fov_review_status.unique()).issubset(allowed_status):
        raise ValueError("癌图包含未允许的FOV审核状态")
    pending_cancer = cancer.loc[cancer.fov_review_status.eq("pending")]
    if len(pending_cancer) != 1:
        raise ValueError("冻结外部癌图应恰有1张FOV pending提示图")

    records = []
    image_root = OUTPUT_ROOT / "无框Keep癌图"
    for index, row in enumerate(cancer.itertuples(index=False), start=1):
        source_path = resolve_project_path(row.image_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        width, height = read_size(source_path)
        annotation_name = f"E{index:04d}__{Path(row.image_name).name}"
        target = image_root / annotation_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source_path, target)
        digest = sha256_file(target)
        if digest != sha256_file(source_path):
            raise RuntimeError(f"复制后SHA不一致: {target}")
        records.append({
            "annotation_index": index,
            "annotation_image_name": annotation_name,
            "patient_id": row.patient_id,
            "label": int(row.label),
            "class_name": row.class_name,
            "source_image_name": row.image_name,
            "source_relative_path": row.relative_path,
            "source_keep_path": row.image_path,
            "source_original_path": row.original_path,
            "original_sha256": row.original_sha256,
            "keep_sha256": digest,
            "keep_width": width,
            "keep_height": height,
            "lesion_visible": "",
            "bbox_quality": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "coordinate_convention": "xyxy_left_closed_right_open",
            "clinical_note": "",
        })

    manifest = pd.DataFrame(records)
    if manifest.annotation_image_name.nunique() != 542:
        raise ValueError("交接文件名不唯一")
    manifest_path = OUTPUT_ROOT / "外部多中心癌图标框任务清单.csv"
    write_csv(manifest, manifest_path)
    duplicates = exact_duplicate_pairs(manifest)
    write_csv(duplicates, OUTPUT_ROOT / "癌图Keep完全重复候选.csv")

    readme = """# 胃镜多中心外部集癌图标框说明

## 材料范围

本目录包含外部多中心Keep裁剪后的542张癌图，覆盖121位癌患者。非癌图不需要病灶框，
因此未复制到本次标注包；它们仍保留在项目冻结外部测试清单中作为负样本。

## 标注要求

1. 每张图在Keep图坐标系内标注主要癌/HGD病灶的矩形框，尽量贴合完整病灶；
2. 不要为了增加上下文而主动扩框，模型裁图阶段会按冻结规则另行扩边；
3. 不要把器械、反光、气泡、黏液或胃镜边缘标成病灶；
4. 若病灶不可见或无法可靠判断，标记`lesion_visible=0`并填写说明，不要强行画框；
5. 不得修改文件名、图像尺寸、图像内容或目录中的任务清单；
6. 坐标采用`x1,y1,x2,y2`，左上角为原点。建议直接使用CVAT导出XML，并同时返回导出文件。

文件名前缀`E0001`等是本次交接的唯一编号，用于避免不同患者图片同名。原患者ID、原文件名、
Keep SHA和尺寸均记录在`外部多中心癌图标框任务清单.csv`中。

`E0440__2.jpg`在历史自动验收中因视野外存在次级连通区域被标记为`pending`，现已人工复核：
有效胃黏膜完整，黑色区域位于胃镜视野外，可以正常标框，不需要修改图像。

## 使用边界

这批外部数据此前已经参与过模型比较，补充病灶框主要用于描述性评价模型是否关注病灶，
不得用来重新调训练参数、选择checkpoint或修改分类阈值。
"""
    (OUTPUT_ROOT / "00_医学生标框说明.md").write_text(readme, encoding="utf-8")

    summary = {
        "stage": "external_multicenter_keep_cancer_annotation_handoff",
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_all_images": 1941,
        "source_all_patients": 1329,
        "annotation_images": int(len(manifest)),
        "annotation_patients": int(manifest.patient_id.nunique()),
        "noncancer_images_not_copied": 1399,
        "exact_duplicate_pairs": int(len(duplicates)),
        "cancer_fov_pending_images": int(len(pending_cancer)),
        "annotation_manifest_sha256": sha256_file(manifest_path),
        "images_copied_not_moved": True,
        "model_inference_performed": False,
    }
    (OUTPUT_ROOT / "00_交接冻结汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
