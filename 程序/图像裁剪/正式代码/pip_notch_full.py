#!/usr/bin/env python3
"""为全部已确认画中画生成保持原尺寸的notch候选。

输入以 confirmed_pip_labels.csv 为处理池，以 quality_flags.csv 的自动框
为基线；若 pip_rect_overrides.csv 存在已确认人工修正，则优先采用修正
框。脚本不修改任何输入，每张图输出保持原画幅的notch图：将画中画框及
安全边距置黑，不改变图像尺寸和病灶bbox坐标；同时输出处理前后预览。

同时输出 mapping.csv、框选质控联系表和 run_manifest.json。所有结果均为
候选，框完整性复核完成前不得冻结为训练数据。
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    PIP_OVERRIDES,
    PIP_IMAGE_ROOT,
    STAGE3_QUALITY_FLAGS,
    STAGE3_PIP_LABELS,
    STAGE4_OUTPUT,
)


IMAGE_ROOT = PIP_IMAGE_ROOT
QUALITY_FLAGS = STAGE3_QUALITY_FLAGS
PIP_LABELS = STAGE3_PIP_LABELS
OVERRIDES = PIP_OVERRIDES
OUTPUT_ROOT = STAGE4_OUTPUT
DEFAULT_MARGIN_RATIO = 0.008
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_ROWS = 4
CONTACT_TILE_WIDTH = 360
PANEL_SIZE = 480


def parse_rect(text):
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 4:
        raise ValueError(f"无效矩形：{text}")
    return values


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图像读取失败：{path}")
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    suffix = path.suffix.lower() or ".jpg"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, parameters)
    if not ok:
        raise OSError(f"图像写入失败：{path}")
    encoded.tofile(str(path))


def black_ratio(image):
    return float((image.max(axis=2) <= 12).mean())


def fit_panel(image, panel_size=PANEL_SIZE):
    height, width = image.shape[:2]
    scale = min(panel_size / width, panel_size / height)
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    x = (panel_size - resized.shape[1]) // 2
    y = (panel_size - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def labelled_panel(image, title, detail):
    panel = fit_panel(image)
    header = np.full((52, PANEL_SIZE, 3), 35, dtype=np.uint8)
    cv2.putText(
        header, title[:60], (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        header, detail[:88], (7, 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([header, panel])


def parse_args():
    parser = argparse.ArgumentParser(
        description="画中画notch全量候选处理（保持原图尺寸）"
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=IMAGE_ROOT,
        help="第一阶段自然裁剪图。",
    )
    parser.add_argument("--quality-flags", type=Path, default=QUALITY_FLAGS)
    parser.add_argument("--pip-labels", type=Path, default=PIP_LABELS)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--margin-ratio", type=float, default=DEFAULT_MARGIN_RATIO
    )
    return parser.parse_args()


def validate_args(args):
    for path in [
        args.image_root,
        args.quality_flags,
        args.pip_labels,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")
    if not 0 <= args.margin_ratio <= 0.05:
        raise ValueError("--margin-ratio 必须位于 [0, 0.05]")


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(args):
    quality_rows = load_csv(args.quality_flags)
    quality_by_path = {
        row["relative_path"]: row for row in quality_rows
    }
    if len(quality_by_path) != len(quality_rows):
        raise ValueError("quality_flags.csv 存在重复 relative_path")

    label_rows = load_csv(args.pip_labels)
    if len({row["relative_path"] for row in label_rows}) != len(label_rows):
        raise ValueError("confirmed_pip_labels.csv 存在重复 relative_path")
    if any(row["manual_label"] != "pip" for row in label_rows):
        raise ValueError("处理池中存在非 pip 标签")

    override_rows = []
    if args.overrides.is_file():
        override_rows = [
            row for row in load_csv(args.overrides)
            if row["review_status"] == "confirmed"
        ]
    override_by_path = {
        row["relative_path"]: row for row in override_rows
    }
    if len(override_by_path) != len(override_rows):
        raise ValueError("人工修正表存在重复的已确认 relative_path")

    items = []
    for label in sorted(label_rows, key=lambda row: row["relative_path"]):
        relative_path = label["relative_path"]
        quality = quality_by_path.get(relative_path)
        if quality is None:
            raise ValueError(f"缺少 quality_flags 行：{relative_path}")
        if not quality["rect_xywh"]:
            raise ValueError(f"缺少自动框：{relative_path}")
        override = override_by_path.get(relative_path)
        if override is not None:
            if override["original_rect_xywh"] != quality["rect_xywh"]:
                raise ValueError(
                    f"人工修正表原框与自动框不一致：{relative_path}"
                )
            effective_rect = override["corrected_rect_xywh"]
            rect_source = "manual_override"
        else:
            effective_rect = quality["rect_xywh"]
            rect_source = "auto"
        items.append({
            "label": label,
            "quality": quality,
            "override": override,
            "effective_rect": effective_rect,
            "rect_source": rect_source,
        })
    return items


def marked_source(image, auto_rect, cut_x, cut_y):
    marked = image.copy()
    auto_x, auto_y, auto_w, auto_h = auto_rect
    cv2.rectangle(
        marked,
        (auto_x, auto_y),
        (auto_x + auto_w, auto_y + auto_h),
        (0, 255, 255),
        3,
    )
    cv2.rectangle(
        marked, (0, 0), (cut_x, cut_y), (0, 0, 255), 3
    )
    return marked


def process_item(args, item):
    relative_path = item["label"]["relative_path"]
    image_path = args.image_root / relative_path
    image = read_image(image_path)

    auto_rect = parse_rect(item["quality"]["rect_xywh"])
    effective_rect = parse_rect(item["effective_rect"])
    rect_x, rect_y, rect_w, rect_h = effective_rect
    if rect_x < 0 or rect_y < 0 or rect_w <= 0 or rect_h <= 0:
        raise ValueError(f"非法处理框：{effective_rect}")

    margin = max(
        4, int(round(min(image.shape[:2]) * args.margin_ratio))
    )
    cut_x = min(image.shape[1] - 1, rect_x + rect_w + margin)
    cut_y = min(image.shape[0] - 1, rect_y + rect_h + margin)
    notch = image.copy()
    notch[:cut_y, :cut_x] = 0
    notch_black = black_ratio(notch)
    newly_blacked = float(
        np.any(image[:cut_y, :cut_x] > 12, axis=2).sum()
        / (image.shape[0] * image.shape[1])
    )

    path = Path(relative_path)
    notch_path = args.output / "candidates" / "notch" / path
    write_image(notch_path, notch)

    source_panel = labelled_panel(
        marked_source(image, auto_rect, cut_x, cut_y),
        f"{item['label']['patient_id']} | {path.name}",
        (
            f"yellow=auto {item['quality']['rect_xywh']} "
            f"red=cut {cut_x}x{cut_y} "
            f"source={item['rect_source']}"
        ),
    )
    preview = np.hstack([
        source_panel,
        labelled_panel(
            notch,
            "notch branch",
            (
                f"retain=100.0% black={notch_black:.1%} "
                f"newblack={newly_blacked:.1%} "
                f"size={image.shape[1]}x{image.shape[0]}"
            ),
        ),
    ])
    preview_path = args.output / "previews" / path
    write_image(preview_path, preview)

    return {
        "relative_path": relative_path,
        "patient_id": item["label"]["patient_id"],
        "auto_tier": item["label"]["auto_tier"],
        "proposal_type": item["quality"]["proposal_type"],
        "pip_score": item["label"]["pip_score"],
        "source_width": image.shape[1],
        "source_height": image.shape[0],
        "auto_rect_xywh": item["quality"]["rect_xywh"],
        "effective_rect_xywh": item["effective_rect"],
        "rect_source": item["rect_source"],
        "margin_pixels": margin,
        "cut_x": cut_x,
        "cut_y": cut_y,
        "notch_output_width": image.shape[1],
        "notch_output_height": image.shape[0],
        "notch_fov_retention": 1.0,
        "notch_black_ratio": round(notch_black, 6),
        "notch_newly_blacked_pixel_ratio": round(
            newly_blacked, 6
        ),
        "box_qc_status": "pending",
        "box_qc_note": "",
        "notch_output": str(notch_path),
        "preview": str(preview_path),
        "_contact_panel": source_panel,
    }


def contact_tile(panel):
    scale = CONTACT_TILE_WIDTH / panel.shape[1]
    return cv2.resize(
        panel,
        (
            CONTACT_TILE_WIDTH,
            max(1, int(round(panel.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def write_contact_sheets(output, rows):
    per_page = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_ROWS
    sheets = []
    for start in range(0, len(rows), per_page):
        page_rows = rows[start:start + per_page]
        tiles = [contact_tile(row["_contact_panel"]) for row in page_rows]
        blank = np.full_like(tiles[0], 25)
        tiles.extend([blank] * (per_page - len(tiles)))
        grid_rows = []
        for row_index in range(CONTACT_SHEET_ROWS):
            begin = row_index * CONTACT_SHEET_COLUMNS
            grid_rows.append(np.hstack(
                tiles[begin:begin + CONTACT_SHEET_COLUMNS]
            ))
        sheet = np.vstack(grid_rows)
        page_number = start // per_page + 1
        path = output / "box_qc_contact_sheets" / (
            f"page_{page_number:02d}.jpg"
        )
        write_image(path, sheet)
        sheets.append(path)
    return sheets


def main():
    args = parse_args()
    validate_args(args)
    items = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for index, item in enumerate(items, start=1):
        row = process_item(args, item)
        rows.append(row)
        print(
            f"[{index}/{len(items)}] {row['relative_path']} "
            f"rect={row['rect_source']} cut={row['cut_x']}x{row['cut_y']}"
        )

    sheets = write_contact_sheets(args.output, rows)
    for row in rows:
        row.pop("_contact_panel")

    mapping_path = args.output / "mapping.csv"
    with mapping_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    script_path = Path(__file__)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "candidate_pending_box_qc",
        "script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "inputs": {
            "pip_labels": str(args.pip_labels),
            "pip_labels_sha256": file_sha256(args.pip_labels),
            "quality_flags": str(args.quality_flags),
            "quality_flags_sha256": file_sha256(args.quality_flags),
            "overrides": str(args.overrides) if args.overrides.is_file() else "",
            "overrides_sha256": (
                file_sha256(args.overrides) if args.overrides.is_file() else None
            ),
        },
        "image_count": len(rows),
        "patient_count": len({
            row["patient_id"] for row in rows
        }),
        "branches": ["notch"],
        "margin_ratio": args.margin_ratio,
        "rect_sources": dict(Counter(
            row["rect_source"] for row in rows
        )),
        "proposal_types": dict(Counter(
            row["proposal_type"] for row in rows
        )),
        "contact_sheet_count": len(sheets),
        "mean_notch_black_ratio": float(np.mean([
            float(row["notch_black_ratio"]) for row in rows
        ])),
        "mapping": str(mapping_path),
        "important_limitation": (
            "候选框尚待完整性复核；复核完成前不得作为冻结训练数据"
        ),
    }
    with (args.output / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
