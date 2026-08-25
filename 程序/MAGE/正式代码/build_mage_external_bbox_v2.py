#!/usr/bin/env python3
"""整合外部多中心癌图病灶框，并生成全量人工验收三联图。

输入是原有542张Keep癌图交接清单与医学生返回的CVAT XML。脚本不删除或
改写任何原图，只将无框图记录为空间评价排除项，并输出独立的v2清单。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANNOTATIONS_XML = PROJECT_ROOT / "数据/标注-外部测试集胃镜多中心.xml"
TASK_MANIFEST = PROJECT_ROOT / (
    "数据/标注与汇总/胃镜多中心外部集_Keep癌图标框交接_20260819/"
    "外部多中心癌图标框任务清单.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃镜多中心测试集_M0Keep预处理_v1_20260805/"
    "03_外部癌图标注整合_v2_20260824"
)
EXCLUSION_REASON = "病灶不清晰，医学生未标框，排除于病灶空间评价"


def parse_args() -> argparse.Namespace:
    """解析输入与输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_XML)
    parser.add_argument("--task-manifest", type=Path, default=TASK_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_chinese_font() -> None:
    """为验收图选择服务器上可用的中文字体。"""
    candidates = ("Noto Sans CJK SC", "Droid Sans Fallback", "SimSun")
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_cvat_xml(path: Path) -> pd.DataFrame:
    """读取CVAT图像节点，每张图输出一行框坐标或无框状态。"""
    records = []
    for image in ET.parse(path).getroot().iter("image"):
        boxes = image.findall("box")
        if len(boxes) > 1:
            raise ValueError(f"CVAT图像包含多个框: {image.attrib['name']}")
        record = {
            "annotation_image_name": image.attrib["name"],
            "xml_image_id": int(image.attrib["id"]),
            "xml_width": int(image.attrib["width"]),
            "xml_height": int(image.attrib["height"]),
            "bbox_available": bool(boxes),
            "bbox_x1": pd.NA,
            "bbox_y1": pd.NA,
            "bbox_x2": pd.NA,
            "bbox_y2": pd.NA,
        }
        if boxes:
            box = boxes[0]
            if box.attrib["label"] != "癌":
                raise ValueError(
                    f"CVAT标签不是癌: {image.attrib['name']}={box.attrib['label']}"
                )
            record.update({
                "bbox_x1": float(box.attrib["xtl"]),
                "bbox_y1": float(box.attrib["ytl"]),
                "bbox_x2": float(box.attrib["xbr"]),
                "bbox_y2": float(box.attrib["ybr"]),
            })
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.annotation_image_name.duplicated().any():
        raise ValueError("CVAT XML存在重复图像名")
    return frame


def build_manifests(task_path: Path, xml_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """连接交接清单与XML，返回可评价清单和排除清单。"""
    task = pd.read_csv(task_path, encoding="utf-8-sig", dtype={"patient_id": "string"})
    xml = parse_cvat_xml(xml_path)
    if set(task.annotation_image_name) != set(xml.annotation_image_name):
        raise ValueError("交接清单与CVAT XML图像名集合不一致")
    merged = task.merge(
        xml, on="annotation_image_name", how="inner", validate="one_to_one"
    ).sort_values("annotation_index").reset_index(drop=True)
    if len(merged) != len(task):
        raise ValueError(f"标注连接行数异常: {len(merged)}/{len(task)}")
    if not merged.xml_width.eq(merged.keep_width).all() or not merged.xml_height.eq(
        merged.keep_height
    ).all():
        raise ValueError("CVAT画布尺寸与Keep图尺寸不一致")

    available = merged.bbox_available
    valid = (
        merged.loc[available, "bbox_x1"].ge(0)
        & merged.loc[available, "bbox_y1"].ge(0)
        & merged.loc[available, "bbox_x2"].le(merged.loc[available, "keep_width"])
        & merged.loc[available, "bbox_y2"].le(merged.loc[available, "keep_height"])
        & merged.loc[available, "bbox_x2"].gt(merged.loc[available, "bbox_x1"])
        & merged.loc[available, "bbox_y2"].gt(merged.loc[available, "bbox_y1"])
    )
    if not valid.all():
        raise ValueError("CVAT中存在越界或宽高非正的框")

    usable = merged.loc[available].copy()
    usable["bbox_area_fraction"] = (
        (usable.bbox_x2 - usable.bbox_x1)
        * (usable.bbox_y2 - usable.bbox_y1)
        / (usable.keep_width * usable.keep_height)
    )
    for axis, size in (("x", "keep_width"), ("y", "keep_height")):
        usable[f"bbox_{axis}1_norm"] = usable[f"bbox_{axis}1"] / usable[size]
        usable[f"bbox_{axis}2_norm"] = usable[f"bbox_{axis}2"] / usable[size]
    usable["spatial_evaluation_status"] = "included"

    excluded = merged.loc[~available].copy()
    excluded["spatial_evaluation_status"] = "excluded_unclear_lesion"
    excluded["exclusion_reason"] = EXCLUSION_REASON
    return usable, excluded


def render_review_images(frame: pd.DataFrame, output: Path) -> None:
    """为每张可用癌图生成原图、标框图和框内局部三联图。"""
    output.mkdir(parents=True, exist_ok=True)
    for order, row in enumerate(frame.itertuples(index=False), start=1):
        output_path = output / f"{order:04d}_{row.annotation_image_name}.png"
        if output_path.is_file():
            continue
        image_path = PROJECT_ROOT / row.source_keep_path
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            crop = image.crop((row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2))

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("原Keep图")
        axes[1].imshow(image)
        axes[1].add_patch(Rectangle(
            (row.bbox_x1, row.bbox_y1),
            row.bbox_x2 - row.bbox_x1,
            row.bbox_y2 - row.bbox_y1,
            fill=False, edgecolor="#20c05c", linewidth=2.5,
        ))
        axes[1].set_title("医学生病灶框")
        axes[2].imshow(crop)
        axes[2].set_title("框内局部")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{order:04d}/539  {row.annotation_image_name}  "
            f"patient={row.patient_id}  area={row.bbox_area_fraction:.3f}"
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=110, bbox_inches="tight")
        plt.close(figure)


def render_contact_sheets(review_dir: Path, output: Path) -> None:
    """将逐图三联图按3列×6行合成为分页总览。"""
    paths = sorted(review_dir.glob("*.png"))
    if len(paths) != 539:
        raise ValueError(f"逐图三联图数量异常: {len(paths)}/539")
    output.mkdir(parents=True, exist_ok=True)
    columns, rows = 3, 6
    cell_width, cell_height = 1080, 366
    header_height = 48
    page_size = columns * rows
    page_count = math.ceil(len(paths) / page_size)
    for page_index in range(page_count):
        selected = paths[page_index * page_size:(page_index + 1) * page_size]
        canvas = Image.new("RGB", (
            columns * cell_width, header_height + rows * cell_height
        ), "white")
        for index, path in enumerate(selected):
            with Image.open(path) as handle:
                triptych = handle.convert("RGB")
                triptych.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            x = (index % columns) * cell_width + (cell_width - triptych.width) // 2
            y = header_height + (index // columns) * cell_height
            canvas.paste(triptych, (x, y))
        page_path = output / f"page_{page_index + 1:03d}_of_{page_count:03d}.jpg"
        canvas.save(page_path, quality=90, optimize=True)


def main() -> None:
    """执行清单整合、自动验收和全量三联图导出。"""
    args = parse_args()
    usable, excluded = build_manifests(args.task_manifest, args.annotations)
    if len(usable) != 539 or len(excluded) != 3:
        raise ValueError(f"标注计数与医学生反馈不符: {len(usable)}+{len(excluded)}")

    args.output.mkdir(parents=True, exist_ok=True)
    usable_path = args.output / "外部多中心癌图bbox空间评价清单_v2.csv"
    excluded_path = args.output / "外部多中心癌图bbox排除清单_v2.csv"
    if not usable_path.exists():
        usable.to_csv(usable_path, index=False, encoding="utf-8-sig")
        excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    summary = {
        "source_images": int(len(usable) + len(excluded)),
        "source_patients": int(pd.concat([usable, excluded]).patient_id.nunique()),
        "spatial_evaluation_images": int(len(usable)),
        "spatial_evaluation_patients": int(usable.patient_id.nunique()),
        "excluded_images": int(len(excluded)),
        "excluded_patients_lost": int(
            pd.concat([usable, excluded]).patient_id.nunique()
            - usable.patient_id.nunique()
        ),
        "bbox_area_fraction": {
            "min": float(usable.bbox_area_fraction.min()),
            "median": float(usable.bbox_area_fraction.median()),
            "max": float(usable.bbox_area_fraction.max()),
        },
        "excluded": excluded[[
            "annotation_image_name", "patient_id", "exclusion_reason"
        ]].to_dict(orient="records"),
        "original_images_deleted": False,
        "classification_queue_changed": False,
    }
    (args.output / "外部多中心癌图bbox自动验收汇总_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    configure_chinese_font()
    review_dir = args.output / "全量标框三联验收图"
    render_review_images(usable, review_dir)
    render_contact_sheets(review_dir, args.output / "分页总览_每页18例")
    print(
        f"外部bbox v2整合完成: {len(usable)}张可评价/"
        f"{len(excluded)}张排除/{usable.patient_id.nunique()}人\n"
        f"输出: {args.output}"
    )


if __name__ == "__main__":
    main()
