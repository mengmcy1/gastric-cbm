#!/usr/bin/env python3
"""冻结新内部测试集并生成 Keep 裁剪的全量人工验收材料。

脚本分两段执行：

1. ``freeze``：只读原图，冻结标签、患者、尺寸、SHA256和感知哈希，
   并与 MAGE/Y0F/既有外部集做患者和精确 SHA 重复审计。
2. ``finalize``：读取已完成的亮度四边裁剪与 FOV 遮罩 mapping，生成
   冻结 Keep 候选清单、跨队列近重复审计、逐图四联图、分页总览与
   癌图 bbox 标注待验收清单。

原图始终只读；脚本拒绝覆盖非空输出目录。近重复只生成候选，
不自动删除或排除任何测试图。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "数据/内部测试集省人民260612"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/"
    "内部测试集省人民260612_Keep预处理_v1_20260819"
)
STAGE1_NAME = "01_亮度四边裁剪"
STAGE1B_NAME = "01b_暗部裁剪安全复核"
STAGE2_NAME = "02_FOV遮罩"
FREEZE_NAME = "00_原始冻结与重复审计"
REVIEW_NAME = "03_全量人工验收_v1_1"
ANNOTATION_NAME = "04_癌图标框材料_待验收_v1_1"
KEEP_MANIFEST_NAME = "冻结Keep候选清单_v1_1.csv"
APPROVED_MANIFEST_NAME = "冻结Keep人工验收通过清单_v1_1.csv"
FORMAL_ANNOTATION_NAME = "05_癌图标框材料_正式交接_v1_1"
INTEGRATION_NAME = "06_内部测试集标注整合_v1_1"
DEFAULT_ANNOTATIONS_XML = PROJECT_ROOT / "数据/标注与汇总/annotations2.xml"

CLASS_TO_LABEL = {"非癌": 0, "早癌": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PHASH_DISTANCE_THRESHOLD = 6
CONTACT_ROWS_PER_PAGE = 4

REFERENCE_SPECS = {
    "MAGE_development": {
        "manifest": PROJECT_ROOT / (
            "数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817/"
            "independent_axis_dynamic/"
            "mage_teacher_roi_manifest_independent_axis_dynamic_v3.csv"
        ),
        "patient": "patient_id",
        "sha": "sha256",
        "image": "image_relpath",
    },
    "Y0F_full": {
        "manifest": PROJECT_ROOT / (
            "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
            "11_Y0F_YOLO26完整诊断数据_20260813/y0f_mapping.csv"
        ),
        "patient": "patient_id",
        "sha": "sha256",
        "image": "image_relpath",
    },
    "external_multicenter": {
        "manifest": PROJECT_ROOT / (
            "结果/M0全量诊断_0804/外部多中心完整测试_Keep_v1/"
            "external_keep_manifest.csv"
        ),
        "patient": "patient_id",
        "sha": "original_sha256",
        "image": "image_path",
    },
}


def parse_args() -> argparse.Namespace:
    """解析子命令与输入/输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("freeze", "finalize", "approve", "integrate")
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--annotations-xml", type=Path, default=DEFAULT_ANNOTATIONS_XML,
    )
    parser.add_argument(
        "--copy-cancer-images",
        action="store_true",
        help="finalize时将无框Keep癌图复制到待验收标注目录。",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """返回普通文件SHA256。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_project_path(value: str) -> Path:
    """将manifest中的项目相对路径或绝对路径解析为Path。"""
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_bgr(path: Path) -> np.ndarray:
    """支持中文路径的OpenCV图像读取。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图像: {path}")
    return image


def perceptual_hash(path: Path) -> str:
    """计算8x8 DCT pHash，用于生成近重复人工候选。"""
    image = read_bgr(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))[:8, :8]
    values = dct.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    number = 0
    for bit in bits:
        number = (number << 1) | int(bit)
    return f"{number:016x}"


def hamming_distance(left: str, right: str) -> int:
    """计算两个64位pHash的汉明距离。"""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def manifest_sha(path: Path) -> str:
    """读取已生成CSV的SHA，用于冻结上游。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def ensure_new_directory(path: Path) -> None:
    """创建新目录；已存在的非空目录拒绝覆盖。"""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {path}")
    path.mkdir(parents=True, exist_ok=True)


def scan_originals(input_root: Path) -> pd.DataFrame:
    """扫描原图并从目录层级冻结类别和患者。"""
    paths = sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    rows = []
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(input_root)
        if len(relative.parts) < 3 or relative.parts[0] not in CLASS_TO_LABEL:
            raise ValueError(f"不符合“类别/患者/...”结构: {relative}")
        image = read_bgr(path)
        height, width = image.shape[:2]
        rows.append({
            "image_index": index,
            "class_name": relative.parts[0],
            "label": CLASS_TO_LABEL[relative.parts[0]],
            "patient_id": relative.parts[1],
            "relative_path": relative.as_posix(),
            "original_path": str(path.resolve()),
            "original_width": width,
            "original_height": height,
            "file_bytes": path.stat().st_size,
            "original_sha256": sha256_file(path),
            "original_phash": perceptual_hash(path),
            "decode_ok": True,
        })
    if not rows:
        raise FileNotFoundError(f"未找到图像: {input_root}")
    frame = pd.DataFrame(rows)
    label_counts = frame.groupby("patient_id").label.nunique()
    if label_counts.max() != 1:
        raise ValueError("同一患者跨标签出现，拒绝冻结")
    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """以UTF-8 BOM写入CSV，拒绝覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def internal_near_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """列出新队列内pHash距离不超过阈值的图对。"""
    records = []
    rows = list(frame.itertuples(index=False))
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            distance = hamming_distance(left.original_phash, right.original_phash)
            if distance <= PHASH_DISTANCE_THRESHOLD:
                records.append({
                    "left_relative_path": left.relative_path,
                    "right_relative_path": right.relative_path,
                    "phash_distance": distance,
                    "same_patient": left.patient_id == right.patient_id,
                    "same_label": left.label == right.label,
                    "exact_sha_match": left.original_sha256 == right.original_sha256,
                })
    return pd.DataFrame(records, columns=[
        "left_relative_path", "right_relative_path", "phash_distance",
        "same_patient", "same_label", "exact_sha_match",
    ])


def reference_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对三份历史manifest执行患者叶ID与精确SHA交叉审计。"""
    patient_records = []
    sha_records = []
    new_patients = set(frame.patient_id.astype(str))
    new_sha = set(frame.original_sha256.astype(str))
    for reference_name, spec in REFERENCE_SPECS.items():
        reference = pd.read_csv(spec["manifest"], low_memory=False)
        reference_patients = reference[spec["patient"]].astype(str).map(
            lambda value: value.replace("\\", "/").rstrip("/").split("/")[-1]
        )
        for patient in sorted(new_patients & set(reference_patients)):
            patient_records.append({
                "reference": reference_name,
                "patient_id": patient,
            })
        reference_sha = set(reference[spec["sha"]].dropna().astype(str))
        for digest in sorted(new_sha & reference_sha):
            source = frame.loc[
                frame.original_sha256.eq(digest), "relative_path"
            ].iloc[0]
            sha_records.append({
                "reference": reference_name,
                "relative_path": source,
                "sha256": digest,
            })
    return (
        pd.DataFrame(patient_records, columns=["reference", "patient_id"]),
        pd.DataFrame(sha_records, columns=["reference", "relative_path", "sha256"]),
    )


def freeze_originals(args: argparse.Namespace) -> None:
    """生成原始冻结清单、患者汇总和重复审计。"""
    output = args.output / FREEZE_NAME
    ensure_new_directory(output)
    frame = scan_originals(args.input.resolve())
    patient_summary = (
        frame.groupby(["patient_id", "class_name", "label"], as_index=False)
        .agg(image_count=("relative_path", "size"))
        .sort_values(["label", "patient_id"])
    )
    patient_overlap, sha_overlap = reference_overlap(frame)
    near_duplicates = internal_near_duplicates(frame)

    manifest_path = output / "原始图像冻结清单.csv"
    write_csv(frame, manifest_path)
    write_csv(patient_summary, output / "患者标签与图像数汇总.csv")
    write_csv(patient_overlap, output / "跨队列患者ID重复.csv")
    write_csv(sha_overlap, output / "跨队列精确SHA重复.csv")
    write_csv(near_duplicates, output / "队列内近重复候选.csv")

    summary = {
        "stage": "internal_test_260612_original_freeze",
        "input_root": str(args.input.resolve()),
        "images": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "class_images": frame.groupby("class_name").size().astype(int).to_dict(),
        "class_patients": frame.groupby("class_name").patient_id.nunique().astype(int).to_dict(),
        "patient_overlap_count": int(len(patient_overlap)),
        "exact_sha_overlap_count": int(len(sha_overlap)),
        "internal_near_duplicate_pairs": int(len(near_duplicates)),
        "phash_distance_threshold": PHASH_DISTANCE_THRESHOLD,
        "manifest_sha256": sha256_file(manifest_path),
        "references": {
            name: {
                "manifest": str(spec["manifest"]),
                "manifest_sha256": manifest_sha(spec["manifest"]),
            }
            for name, spec in REFERENCE_SPECS.items()
        },
        "boundary": "近重复仅为人工候选，未删除或排除任何测试图。",
    }
    (output / "冻结与重复审计汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_mappings(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取原始冻结、stage1和stage2 mapping并核对一对一血缘。"""
    original_path = args.output / FREEZE_NAME / "原始图像冻结清单.csv"
    stage1_path = args.output / STAGE1_NAME / "mapping.csv"
    stage2_path = args.output / STAGE2_NAME / "mapping.csv"
    original = pd.read_csv(
        original_path, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    stage1 = pd.read_csv(stage1_path, encoding="utf-8-sig")
    stage2 = pd.read_csv(stage2_path, encoding="utf-8-sig")
    expected = set(original.relative_path.astype(str))
    for name, frame in (("stage1", stage1), ("stage2", stage2)):
        values = frame.relative_path.astype(str)
        if len(frame) != len(expected) or values.nunique() != len(frame):
            raise ValueError(f"{name} mapping行数或唯一性异常")
        if set(values) != expected:
            raise ValueError(f"{name} mapping与原始冻结路径不一致")
    if stage1.review_status.eq("error").any() or stage2.review_status.eq("error").any():
        raise RuntimeError("Keep预处理存在error行，拒绝生成验收包")
    return original, stage1, stage2


def build_keep_manifest(
    original: pd.DataFrame, stage1: pd.DataFrame, stage2: pd.DataFrame
) -> pd.DataFrame:
    """合并原图、裁剪坐标与FOV质控，并冻结Keep图SHA/几何ID。"""
    s1_columns = [
        "relative_path", "crop", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
        "crop_width", "crop_height", "retained_area_ratio", "crop_method",
        "crop_status", "failure_reason", "edge_refined", "progress_bar_detected",
        "review_status", "review_reasons",
    ]
    s2_columns = [
        "relative_path", "mask", "masked", "width", "height", "mask_coverage",
        "removed_pixel_ratio", "removed_nonblack_ratio", "raw_component_count",
        "secondary_component_ratio", "review_status", "review_reasons",
    ]
    merged = original.merge(
        stage1[s1_columns].rename(columns={
            "review_status": "stage1_review_status",
            "review_reasons": "stage1_review_reasons",
        }), on="relative_path", validate="one_to_one",
    ).merge(
        stage2[s2_columns].rename(columns={
            "review_status": "fov_review_status",
            "review_reasons": "fov_review_reasons",
            "width": "keep_width", "height": "keep_height",
        }), on="relative_path", validate="one_to_one",
    )
    keep_sha = []
    keep_phash = []
    geometry_ids = []
    for row in merged.itertuples(index=False):
        path = resolve_project_path(row.masked)
        digest = sha256_file(path)
        keep_sha.append(digest)
        keep_phash.append(perceptual_hash(path))
        geometry_text = (
            f"{row.relative_path}|{row.crop_x1},{row.crop_y1},"
            f"{row.crop_x2},{row.crop_y2}|{row.keep_width}x{row.keep_height}|{digest}"
        )
        geometry_ids.append(hashlib.sha256(geometry_text.encode()).hexdigest())
    merged["keep_sha256"] = keep_sha
    merged["keep_phash"] = keep_phash
    merged["geometry_id"] = geometry_ids
    merged["human_review"] = ""
    merged["human_note"] = ""
    return merged


def cross_keep_near_duplicates(keep: pd.DataFrame) -> pd.DataFrame:
    """将新Keep图与历史MAGE开发集及既有外部Keep图做pHash比较。"""
    reference_rows = []
    for reference_name in ("MAGE_development", "external_multicenter"):
        spec = REFERENCE_SPECS[reference_name]
        frame = pd.read_csv(
            spec["manifest"], low_memory=False,
            dtype={spec["patient"]: "string"},
        )
        for row in frame.itertuples(index=False):
            path = resolve_project_path(getattr(row, spec["image"]))
            if not path.is_file():
                raise FileNotFoundError(path)
            reference_rows.append((
                reference_name,
                str(getattr(row, spec["patient"])),
                str(path),
                perceptual_hash(path),
            ))
    records = []
    for row in keep.itertuples(index=False):
        for reference_name, patient_id, path, phash in reference_rows:
            distance = hamming_distance(row.keep_phash, phash)
            if distance <= PHASH_DISTANCE_THRESHOLD:
                records.append({
                    "new_relative_path": row.relative_path,
                    "new_patient_id": row.patient_id,
                    "reference": reference_name,
                    "reference_patient_id": patient_id,
                    "reference_image_path": path,
                    "phash_distance": distance,
                })
    return pd.DataFrame(records, columns=[
        "new_relative_path", "new_patient_id", "reference",
        "reference_patient_id", "reference_image_path", "phash_distance",
    ])


def keep_exact_duplicates(keep: pd.DataFrame) -> pd.DataFrame:
    """列出Keep后字节完全相同的图对，不自动排除。"""
    records = []
    for digest, group in keep.groupby("keep_sha256", sort=False):
        rows = list(group.itertuples(index=False))
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                records.append({
                    "left_relative_path": left.relative_path,
                    "right_relative_path": right.relative_path,
                    "keep_sha256": digest,
                    "same_patient": left.patient_id == right.patient_id,
                    "same_label": int(left.label) == int(right.label),
                    "original_sha_match": (
                        left.original_sha256 == right.original_sha256
                    ),
                })
    return pd.DataFrame(records, columns=[
        "left_relative_path", "right_relative_path", "keep_sha256",
        "same_patient", "same_label", "original_sha_match",
    ])


def font(size: int) -> ImageFont.FreeTypeFont:
    """返回服务器上含中文字形的Noto CJK字体。"""
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError("未找到可用中文字体")


def image_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    """将图像等比缩放并居中放入固定画布。"""
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def build_near_duplicate_review(
    keep: pd.DataFrame, near: pd.DataFrame, output: Path
) -> None:
    """为跨队列pHash候选生成逐对左右对照，供人工排除假阳性。"""
    if near.empty:
        return
    review_dir = output / "跨队列近重复逐对对照"
    review_dir.mkdir(parents=True, exist_ok=True)
    keep_lookup = keep.set_index("relative_path")
    title_font = font(20)
    detail_font = font(15)
    for index, row in enumerate(near.itertuples(index=False), start=1):
        new_path = resolve_project_path(
            keep_lookup.loc[row.new_relative_path, "masked"]
        )
        reference_path = Path(row.reference_image_path)
        canvas = Image.new("RGB", (1000, 410), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(image_panel(new_path, (490, 330)), (0, 46))
        canvas.paste(image_panel(reference_path, (490, 330)), (510, 46))
        draw.text((8, 10), "新内部集", fill="black", font=title_font)
        draw.text((518, 10), str(row.reference), fill="black", font=title_font)
        detail = (
            f"pHash距离={int(row.phash_distance)} | "
            f"new={Path(row.new_relative_path).name} | "
            f"ref={reference_path.name}"
        )
        draw.text((8, 382), detail, fill="black", font=detail_font)
        canvas.save(review_dir / f"candidate_{index:03d}.jpg", quality=92)


def render_case(row, output_path: Path) -> None:
    """渲染单张原图/四边裁剪/FOV掩膜/最终Keep四联图。"""
    panel_size = (430, 290)
    paths = [
        Path(row.original_path),
        resolve_project_path(row.crop),
        resolve_project_path(row.mask),
        resolve_project_path(row.masked),
    ]
    titles = ("原图", "四边裁剪", "FOV掩膜", "最终Keep图")
    top = 54
    bottom = 62
    canvas = Image.new(
        "RGB", (panel_size[0] * 4, top + panel_size[1] + bottom), "white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    meta_font = font(17)
    for index, (path, title) in enumerate(zip(paths, titles)):
        x = index * panel_size[0]
        canvas.paste(image_panel(path, panel_size), (x, top))
        draw.text((x + 8, 12), title, fill="black", font=title_font)
    review_reasons = [
        "" if pd.isna(value) else str(value).strip()
        for value in (row.stage1_review_reasons, row.fov_review_reasons)
    ]
    flags = "|".join(filter(None, review_reasons)) or "auto_pass_candidate"
    metadata = (
        f"{int(row.image_index):04d}  {row.class_name}  patient={row.patient_id}  "
        f"crop={row.crop_status}  coverage={float(row.mask_coverage):.3f}  {flags}"
    )
    draw.text((8, top + panel_size[1] + 14), metadata, fill="black", font=meta_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def build_review_assets(keep: pd.DataFrame, output: Path) -> pd.DataFrame:
    """生成113张逐图四联与每页4例分页总览，返回待填审核表。"""
    single_dir = output / "逐图四联"
    page_dir = output / "分页总览"
    single_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths = []
    for row in keep.sort_values("image_index").itertuples(index=False):
        safe_patient = str(row.patient_id).replace("/", "_")
        target = single_dir / (
            f"{int(row.image_index):04d}_{row.class_name}_{safe_patient}.jpg"
        )
        render_case(row, target)
        rendered_paths.append(target)

    page_width = 1720
    row_height = 406
    for page_index, start in enumerate(
        range(0, len(rendered_paths), CONTACT_ROWS_PER_PAGE), start=1
    ):
        paths = rendered_paths[start:start + CONTACT_ROWS_PER_PAGE]
        page = Image.new("RGB", (page_width, row_height * len(paths)), "white")
        for row_index, path in enumerate(paths):
            with Image.open(path) as handle:
                case = handle.convert("RGB")
            page.paste(case, (0, row_index * row_height))
        page.save(page_dir / f"page_{page_index:03d}.jpg", quality=90)

    review = keep[[
        "image_index", "class_name", "label", "patient_id", "relative_path",
        "masked", "keep_sha256", "geometry_id", "crop_status",
        "stage1_review_status", "stage1_review_reasons",
        "fov_review_status", "fov_review_reasons", "mask_coverage",
    ]].copy()
    review["human_decision"] = ""
    review["human_issue_type"] = ""
    review["human_note"] = ""
    return review


def build_annotation_staging(
    keep: pd.DataFrame, output: Path, copy_images: bool
) -> None:
    """生成52张癌图标框模板；只在显式开关下复制无框Keep图。"""
    cancer = keep.loc[keep.label.eq(1)].sort_values("image_index").copy()
    annotation = cancer[[
        "image_index", "patient_id", "relative_path", "masked", "keep_sha256",
        "geometry_id", "keep_width", "keep_height",
    ]].copy()
    annotation["lesion_visible"] = ""
    annotation["bbox_quality"] = ""
    annotation["x1"] = ""
    annotation["y1"] = ""
    annotation["x2"] = ""
    annotation["y2"] = ""
    annotation["coordinate_convention"] = "xyxy_left_closed_right_open"
    annotation["clinical_note"] = ""
    write_csv(annotation, output / "癌图病灶框标注模板.csv")
    readme = """# 癌图病灶框标注材料（待裁剪验收）

- 本目录当前仅准备标注清单；须等全113张Keep裁剪由用户验收后才能作为正式标注输入。
- 标注对象为无可视化框的高分辨率Keep图，不是224x224缩略图。
- 若图中看不到病灶，填`lesion_visible=0`，不得为满足定位损失而强行画框。
- 坐标单独写入CSV/JSON，不要把框画进模型输入图。
"""
    (output / "00_标注说明.md").write_text(readme, encoding="utf-8")
    if not copy_images:
        return
    image_root = output / "无框Keep癌图"
    for row in cancer.itertuples(index=False):
        source = resolve_project_path(row.masked)
        target = image_root / Path(row.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source, target)


def finalize_keep(args: argparse.Namespace) -> None:
    """冻结Keep候选并生成自动审计、全量四联图和标框待验收材料。"""
    review_output = args.output / REVIEW_NAME
    annotation_output = args.output / ANNOTATION_NAME
    ensure_new_directory(review_output)
    ensure_new_directory(annotation_output)
    original, stage1, stage2 = load_mappings(args)
    keep = build_keep_manifest(original, stage1, stage2)
    keep_manifest_path = args.output / KEEP_MANIFEST_NAME
    write_csv(keep, keep_manifest_path)
    near_cross = cross_keep_near_duplicates(keep)
    write_csv(near_cross, review_output / "跨队列Keep近重复候选.csv")
    exact_keep = keep_exact_duplicates(keep)
    write_csv(exact_keep, review_output / "队列内Keep完全重复候选.csv")
    build_near_duplicate_review(keep, near_cross, review_output)
    review = build_review_assets(keep, review_output)
    write_csv(review, review_output / "全量人工验收记录.csv")
    build_annotation_staging(keep, annotation_output, args.copy_cancer_images)

    summary = {
        "stage": "internal_test_260612_keep_candidate",
        "images": int(len(keep)),
        "patients": int(keep.patient_id.nunique()),
        "cancer_images": int(keep.label.eq(1).sum()),
        "cancer_patients": int(keep.loc[keep.label.eq(1), "patient_id"].nunique()),
        "stage1_status": keep.crop_status.value_counts().astype(int).to_dict(),
        "stage1_review": keep.stage1_review_status.value_counts().astype(int).to_dict(),
        "fov_review": keep.fov_review_status.value_counts().astype(int).to_dict(),
        "mask_coverage": {
            key: float(value) for key, value in
            keep.mask_coverage.astype(float).describe().to_dict().items()
        },
        "cross_keep_near_duplicate_candidates": int(len(near_cross)),
        "internal_keep_exact_duplicate_pairs": int(len(exact_keep)),
        "phash_distance_threshold": PHASH_DISTANCE_THRESHOLD,
        "keep_manifest_sha256": sha256_file(keep_manifest_path),
        "original_manifest_sha256": manifest_sha(
            args.output / FREEZE_NAME / "原始图像冻结清单.csv"
        ),
        "stage1_mapping_sha256": manifest_sha(
            args.output / STAGE1_NAME / "mapping.csv"
        ),
        "stage2_mapping_sha256": manifest_sha(
            args.output / STAGE2_NAME / "mapping.csv"
        ),
        "human_review_complete": False,
        "annotation_images_copied": bool(args.copy_cancer_images),
        "boundary": "Keep候选须经113张全量人工验收后才可转为正式标框输入。",
    }
    (review_output / "自动验收汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def approve_keep(args: argparse.Namespace) -> None:
    """冻结人工验收结论与标框交接材料，重复图留待临床决定。"""
    keep_path = args.output / KEEP_MANIFEST_NAME
    review_path = args.output / REVIEW_NAME / "全量人工验收记录.csv"
    keep = pd.read_csv(
        keep_path, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    review = pd.read_csv(
        review_path, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    if len(keep) != 113 or len(review) != 113:
        raise ValueError("人工验收冻结要求恰好113张候选图")
    if set(keep.geometry_id) != set(review.geometry_id):
        raise ValueError("人工验收表与Keep候选几何ID不一致")

    exact = keep_exact_duplicates(keep)
    if len(exact) != 1:
        raise ValueError(f"预期1对Keep完全重复图，实际{len(exact)}对")
    duplicate = exact.iloc[0]
    pair = keep.loc[keep.relative_path.isin([
        duplicate.left_relative_path, duplicate.right_relative_path,
    ])].sort_values("image_index")
    if len(pair) != 2 or pair.patient_id.nunique() != 1:
        raise ValueError("完全重复图不满足同患者二选一条件")
    approved_review = review.copy()
    approved_review["human_decision"] = "pass"
    approved_review["human_issue_type"] = ""
    approved_review["human_note"] = "2026-08-19用户全量肉眼验收通过"
    write_csv(
        approved_review,
        args.output / REVIEW_NAME / "全量人工验收记录_已确认.csv",
    )

    approved = keep.copy()
    approved["human_review"] = "pass"
    approved["human_note"] = "2026-08-19用户全量肉眼验收通过"
    approved["duplicate_decision"] = ""
    approved.loc[
        approved.relative_path.isin(pair.relative_path), "duplicate_decision"
    ] = "pending_clinician_review"
    approved_path = args.output / APPROVED_MANIFEST_NAME
    write_csv(approved, approved_path)

    handoff = args.output / FORMAL_ANNOTATION_NAME
    ensure_new_directory(handoff)
    cancer = approved.loc[approved.label.eq(1)].sort_values("image_index").copy()
    annotation = cancer[[
        "image_index", "patient_id", "relative_path", "masked",
        "keep_sha256", "geometry_id", "keep_width", "keep_height",
    ]].copy()
    annotation["copied_image_relative_path"] = annotation.relative_path.map(
        lambda value: (Path("无框Keep癌图") / Path(value)).as_posix()
    )
    annotation["lesion_visible"] = ""
    annotation["bbox_quality"] = ""
    annotation["x1"] = ""
    annotation["y1"] = ""
    annotation["x2"] = ""
    annotation["y2"] = ""
    annotation["coordinate_convention"] = "xyxy_left_closed_right_open"
    annotation["clinical_note"] = ""
    write_csv(annotation, handoff / "癌图病灶框标注模板.csv")

    image_root = handoff / "无框Keep癌图"
    for row in cancer.itertuples(index=False):
        source = resolve_project_path(row.masked)
        target = image_root / Path(row.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != row.keep_sha256:
            raise RuntimeError(f"交接复制SHA不一致: {target}")

    duplicate_review = pair[[
        "image_index", "patient_id", "relative_path", "masked",
        "keep_sha256", "original_sha256",
    ]].copy()
    duplicate_review["clinical_same_image"] = ""
    duplicate_review["keep_for_evaluation"] = ""
    duplicate_review["clinical_note"] = ""
    write_csv(duplicate_review, handoff / "重复图临床判定表.csv")
    duplicate_root = handoff / "重复图待确认"
    for row in pair.itertuples(index=False):
        source = resolve_project_path(row.masked)
        target = duplicate_root / f"{int(row.image_index):04d}_{Path(row.relative_path).name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    readme = """# 医学生病灶框标注说明

本目录含52张已通过Keep裁剪人工验收的癌/HGD图像，来自32位患者。图像均为无框版本；
患者ID、图像SHA、尺寸和坐标模板已经冻结。

1. 请在Keep图坐标系内画一个尽量贴合完整病灶的矩形框，不要人为扩大到大块正常黏膜。
2. 不要把反光、气泡、黏液、器械或胃镜边缘当作病灶边界。
3. 若病灶不可见或无法可靠确定，填写`lesion_visible=0`并说明原因，不要强行画框。
4. 坐标使用`x1,y1,x2,y2`，左上角为原点；不要把可视化框写入模型输入图。
5. 不得修改文件名、患者目录、图像尺寸或图像内容。

完成后请同时返回坐标文件和原目录结构。该材料仅用于病灶定位监督，不改变患者癌/非癌标签。

另外，`重复图待确认/`中有同一非癌患者的两张候选图。请在`重复图临床判定表.csv`中判断
它们是否属于同一临床画面，以及正式模型评估时建议保留哪一张。在该结论返回前，项目不会
擅自删除图像或冻结最终评估清单。
"""
    (handoff / "00_医学生标框说明.md").write_text(readme, encoding="utf-8")

    summary = {
        "stage": "internal_test_260612_keep_crop_approved",
        "approval_date": "2026-08-19",
        "candidate_images": int(len(keep)),
        "approved_images": int(len(approved)),
        "approved_patients": int(approved.patient_id.nunique()),
        "duplicate_candidate_pairs": 1,
        "duplicate_clinical_decision": "pending",
        "formal_evaluation_manifest_frozen": False,
        "cancer_annotation_images": int(len(cancer)),
        "cancer_annotation_patients": int(cancer.patient_id.nunique()),
        "approved_manifest_sha256": sha256_file(approved_path),
        "model_inference_performed": False,
    }
    (handoff / "交接冻结汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def integrate_annotations(args: argparse.Namespace) -> None:
    """校验CVAT XML，冻结bbox并生成去除已确认重复图的最终测试清单。"""
    approved_path = args.output / APPROVED_MANIFEST_NAME
    approved = pd.read_csv(
        approved_path, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    if len(approved) != 113 or approved.patient_id.nunique() != 78:
        raise ValueError("人工验收通过清单数量异常")
    duplicate = approved.loc[
        approved.duplicate_decision.eq("pending_clinician_review")
    ].sort_values("image_index")
    if len(duplicate) != 2 or duplicate.patient_id.nunique() != 1:
        raise ValueError("重复候选不满足同患者两张的冻结记录")
    missing = duplicate.loc[
        ~duplicate.original_path.map(lambda value: Path(value).is_file())
    ]
    if len(missing) != 1:
        raise ValueError("须恰有一张已确认重复原图被删除")
    excluded_index = int(missing.iloc[0].image_index)
    retained = duplicate.loc[~duplicate.image_index.eq(excluded_index)]
    if len(retained) != 1:
        raise ValueError("重复图保留项不唯一")

    xml_path = args.annotations_xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    images = ET.parse(xml_path).findall(".//image")
    cancer = approved.loc[approved.label.eq(1)].copy()
    cancer["image_name"] = cancer.relative_path.map(lambda value: Path(value).name)
    if cancer.image_name.nunique() != len(cancer):
        raise ValueError("癌图basename不唯一，无法绑定CVAT XML")
    lookup = cancer.set_index("image_name")
    records = []
    xml_names = []
    for image in images:
        name = image.attrib["name"]
        xml_names.append(name)
        if name not in lookup.index:
            raise ValueError(f"XML包含非冻结癌图: {name}")
        row = lookup.loc[name]
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
        if (width, height) != (int(row.keep_width), int(row.keep_height)):
            raise ValueError(f"XML尺寸与Keep图不一致: {name}")
        boxes = image.findall("box")
        if len(boxes) != 1 or boxes[0].attrib.get("label") != "癌":
            raise ValueError(f"要求每张癌图恰有一个癌框: {name}")
        box = boxes[0]
        x1, y1, x2, y2 = map(float, (
            box.attrib["xtl"], box.attrib["ytl"],
            box.attrib["xbr"], box.attrib["ybr"],
        ))
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"XML框越界或无效: {name}")
        records.append({
            "image_index": int(row.image_index),
            "patient_id": row.patient_id,
            "relative_path": row.relative_path,
            "image_name": name,
            "keep_sha256": row.keep_sha256,
            "geometry_id": row.geometry_id,
            "keep_width": width,
            "keep_height": height,
            "bbox_label": "癌",
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_area_fraction": (x2 - x1) * (y2 - y1) / (width * height),
            "coordinate_convention": "xyxy_continuous_keep_coordinates",
        })
    if len(images) != 52 or len(set(xml_names)) != 52:
        raise ValueError("XML必须恰好包含52张唯一癌图")
    if set(xml_names) != set(cancer.image_name):
        raise ValueError("XML与冻结癌图文件名集合不一致")
    bbox = pd.DataFrame(records).sort_values("image_index")

    output = args.output / INTEGRATION_NAME
    ensure_new_directory(output)
    bbox_path = output / "内部测试集癌图bbox冻结.csv"
    write_csv(bbox, bbox_path)
    shutil.copy2(xml_path, output / "annotations2_frozen.xml")

    final = approved.loc[~approved.image_index.eq(excluded_index)].copy()
    final["duplicate_decision"] = "not_duplicate_or_retained"
    bbox_columns = [
        "image_index", "bbox_label", "bbox_x1", "bbox_y1", "bbox_x2",
        "bbox_y2", "bbox_area_fraction", "coordinate_convention",
    ]
    final = final.merge(bbox[bbox_columns], on="image_index", how="left")
    final["bbox_available"] = final.bbox_label.notna()
    if len(final) != 112 or final.patient_id.nunique() != 78:
        raise ValueError("最终内部测试清单应为112张/78人")
    if int(final.bbox_available.sum()) != 52:
        raise ValueError("最终清单应有52张癌图bbox")
    if final.loc[final.label.eq(1), "bbox_available"].ne(True).any():
        raise ValueError("癌图存在bbox缺失")
    if final.loc[final.label.eq(0), "bbox_available"].ne(False).any():
        raise ValueError("非癌图不应包含bbox")
    final_path = output / "内部测试集最终评估清单.csv"
    write_csv(final, final_path)
    exclusion = missing.copy()
    exclusion["exclusion_reason"] = "clinician_confirmed_duplicate_keep_earlier_index"
    write_csv(exclusion, output / "重复图正式排除记录.csv")

    summary = {
        "stage": "internal_test_260612_annotations_integrated",
        "images": int(len(final)),
        "patients": int(final.patient_id.nunique()),
        "cancer_images": int(final.label.eq(1).sum()),
        "cancer_patients": int(final.loc[final.label.eq(1), "patient_id"].nunique()),
        "noncancer_images": int(final.label.eq(0).sum()),
        "noncancer_patients": int(final.loc[final.label.eq(0), "patient_id"].nunique()),
        "bbox_images": int(final.bbox_available.sum()),
        "excluded_duplicate_image_index": excluded_index,
        "excluded_duplicate_relative_path": missing.iloc[0].relative_path,
        "retained_duplicate_image_index": int(retained.iloc[0].image_index),
        "annotations_xml_sha256": sha256_file(xml_path),
        "bbox_manifest_sha256": sha256_file(bbox_path),
        "final_manifest_sha256": sha256_file(final_path),
        "model_inference_performed": False,
    }
    (output / "标注整合与最终清单汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """按子命令执行原图冻结或Keep候选收尾。"""
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    if args.mode == "freeze":
        freeze_originals(args)
    elif args.mode == "finalize":
        finalize_keep(args)
    elif args.mode == "approve":
        approve_keep(args)
    else:
        integrate_annotations(args)


if __name__ == "__main__":
    main()
