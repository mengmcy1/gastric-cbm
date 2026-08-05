#!/usr/bin/env python3
"""画中画框完整性初筛，并输出供人眼核查的 4×4 联图。

这是保守的初步筛查，不自动修改框，也不把候选判定为错误。候选信号：

1. 人工覆盖框；
2. 框宽或框高显著偏离当前数据的主要尺寸簇；
3. 实际处理边界之外仍检测到较长且清晰的横/竖边界。

输出全量初筛指标、候选清单、4×4 联系表和运行清单。最终框结论必须由
人工核查确认，并通过 pip_rect_overrides.csv 追加，不得覆盖自动结果。
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import PIP_IMAGE_ROOT, STAGE4_OUTPUT
from pip_notch_full import fit_panel, parse_rect, read_image, write_image


MAPPING = STAGE4_OUTPUT / "mapping.csv"
IMAGE_ROOT = PIP_IMAGE_ROOT
OUTPUT_ROOT = STAGE4_OUTPUT / "box_qc_initial_screen"
SHEET_COLUMNS = 4
SHEET_ROWS = 4
TILE_WIDTH = 400
HEADER_HEIGHT = 72


def parse_args():
    parser = argparse.ArgumentParser(
        description="画中画框完整性初筛与4×4联图"
    )
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def validate_args(args):
    for path in [args.mapping, args.image_root]:
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def horizontal_signal(gray, y, width, x0=0):
    y = max(2, min(gray.shape[0] - 3, y))
    width = max(8, min(gray.shape[1] - x0, width))
    diff = np.abs(
        gray[y + 2, x0:x0 + width].astype(np.int16)
        - gray[y - 2, x0:x0 + width].astype(np.int16)
    )
    return float(np.median(diff)), float((diff > 18).mean())


def vertical_signal(gray, x, height, y0=0):
    x = max(2, min(gray.shape[1] - 3, x))
    height = max(8, min(gray.shape[0] - y0, height))
    diff = np.abs(
        gray[y0:y0 + height, x + 2].astype(np.int16)
        - gray[y0:y0 + height, x - 2].astype(np.int16)
    )
    return float(np.median(diff)), float((diff > 18).mean())


def strongest_bottom_extension(gray, cut_y, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    start = min(gray.shape[0] - 3, cut_y + 12)
    stop = min(
        gray.shape[0] - 2,
        cut_y + min(130, max(80, rect_h)),
    )
    best = (0.0, cut_y, 0.0, 0.0)
    for y in range(start, stop + 1):
        median, coherence = horizontal_signal(gray, y, rect_w, x0=rect_x)
        score = median * (0.5 + coherence)
        best = max(best, (score, y, median, coherence))
    return best


def strongest_right_extension(gray, cut_x, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    start = min(gray.shape[1] - 3, cut_x + 12)
    stop = min(
        gray.shape[1] - 2,
        cut_x + min(130, max(80, rect_w)),
    )
    best = (0.0, cut_x, 0.0, 0.0)
    for x in range(start, stop + 1):
        median, coherence = vertical_signal(gray, x, rect_h, y0=rect_y)
        score = median * (0.5 + coherence)
        best = max(best, (score, x, median, coherence))
    return best


def strongest_left_extension(gray, cut_x0, rect):
    """在置黑区左侧搜索竖边（向左扩展）。"""
    rect_x, rect_y, rect_w, rect_h = rect
    start = max(3, cut_x0 - min(130, max(80, rect_w)) - 12)
    stop = max(3, cut_x0 - 12)
    best = (0.0, cut_x0, 0.0, 0.0)
    for x in range(stop, start - 1, -1):
        median, coherence = vertical_signal(gray, x, rect_h, y0=rect_y)
        score = median * (0.5 + coherence)
        best = max(best, (score, x, median, coherence))
    return best


def strongest_top_extension(gray, cut_y0, rect):
    """在置黑区上侧搜索横边（向上扩展）。"""
    rect_x, rect_y, rect_w, rect_h = rect
    start = max(3, cut_y0 - min(130, max(80, rect_h)) - 12)
    stop = max(3, cut_y0 - 12)
    best = (0.0, cut_y0, 0.0, 0.0)
    for y in range(stop, start - 1, -1):
        median, coherence = horizontal_signal(gray, y, rect_w, x0=rect_x)
        score = median * (0.5 + coherence)
        best = max(best, (score, y, median, coherence))
    return best


def resolve_pip_corner(row, image):
    """优先取 pip_corner 字段，缺失时按框位置推断。"""
    corner = (row.get("pip_corner") or "").strip()
    if corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
        return corner
    rect_x, rect_y, rect_w, rect_h = parse_rect(
        row["effective_rect_xywh"]
    )
    height, width = image.shape[:2]
    if rect_x < width * 0.15 and rect_y < height * 0.15:
        return "top_left"
    if rect_x >= width * 0.85 and rect_y < height * 0.15:
        return "top_right"
    if rect_x < width * 0.15 and rect_y >= height * 0.85:
        return "bottom_left"
    return "bottom_right"


def screen_row(row, masked_root):
    source_input = row.get("source_input", "")
    image_path = (
        Path(source_input)
        if source_input
        else masked_root / row["relative_path"]
    )
    image = read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    rect = parse_rect(row["effective_rect_xywh"])
    rect_x, rect_y, rect_width, rect_height = rect
    cut_x = int(row["cut_x"])      # 置黑区 far 角（black_x1）
    cut_y = int(row["cut_y"])      # black_y1
    black_x0 = int(row.get("black_x0") or 0)
    black_y0 = int(row.get("black_y0") or 0)
    corner = resolve_pip_corner(row, image)

    # 按角决定检查的两条“对侧”边（画中画锚定角之外仍可能延伸的边）
    if corner in ("top_left", "bottom_left"):
        edge2 = ("right", strongest_right_extension(
            gray, cut_x, rect
        ), cut_x)
    else:
        edge2 = ("left", strongest_left_extension(
            gray, black_x0, rect
        ), black_x0)
    if corner in ("top_left", "top_right"):
        edge1 = ("bottom", strongest_bottom_extension(
            gray, cut_y, rect
        ), cut_y)
    else:
        edge1 = ("top", strongest_top_extension(
            gray, black_y0, rect
        ), black_y0)

    reasons = []
    if row["rect_source"] == "manual_override":
        reasons.append("manual_override")
    if rect_height < 225 or rect_height > 250:
        reasons.append("uncommon_height")
    if rect_width < 180 or 220 <= rect_width < 280:
        reasons.append("uncommon_width")

    def flag_edge(side_name, data, origin):
        median, coherence = data[2], data[3]
        median_threshold = 20 if side_name in ("right", "left") else 12
        coherence_threshold = 0.50 if side_name in ("right", "left") else 0.35
        if median >= median_threshold and coherence >= coherence_threshold:
            reasons.append(f"{side_name}_edge_beyond_cut")
        return round(median, 3), round(coherence, 6), data[1] - origin

    e1_median, e1_coh, e1_dist = flag_edge(edge1[0], edge1[1], edge1[2])
    e2_median, e2_coh, e2_dist = flag_edge(edge2[0], edge2[1], edge2[2])

    result = dict(row)
    result.update({
        "corner": corner,
        "edge1_side": edge1[0],
        "edge1_candidate_pos": edge1[1][1],
        "edge1_distance_beyond_cut": e1_dist,
        "edge1_median": e1_median,
        "edge1_coherence": e1_coh,
        "edge2_side": edge2[0],
        "edge2_candidate_pos": edge2[1][1],
        "edge2_distance_beyond_cut": e2_dist,
        "edge2_median": e2_median,
        "edge2_coherence": e2_coh,
        "screen_candidate": "yes" if reasons else "no",
        "screen_reasons": "|".join(reasons),
        "human_box_status": "",
        "human_corrected_rect_xywh": "",
        "human_note": "",
        "_image": image,
    })
    return result


def make_tile(row, index, total):
    image = row["_image"].copy()
    if row["auto_rect_xywh"]:
        auto_x, auto_y, auto_w, auto_h = parse_rect(
            row["auto_rect_xywh"]
        )
        cv2.rectangle(
            image,
            (auto_x, auto_y),
            (auto_x + auto_w, auto_y + auto_h),
            (0, 255, 255),
            3,
        )
    cv2.rectangle(
        image,
        (int(row.get("black_x0") or 0), int(row.get("black_y0") or 0)),
        (int(row["cut_x"]), int(row["cut_y"])),
        (0, 0, 255),
        3,
    )
    body = fit_panel(image)
    scale = TILE_WIDTH / body.shape[1]
    body = cv2.resize(
        body,
        (
            TILE_WIDTH,
            max(1, int(round(body.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    header = np.full(
        (HEADER_HEIGHT, TILE_WIDTH, 3), 35, dtype=np.uint8
    )
    filename = Path(row["relative_path"]).name
    cv2.putText(
        header,
        f"[{index:03d}/{total:03d}] {filename}"[:48],
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"yellow=auto red=cut {row['cut_x']}x{row['cut_y']}"[:56],
        (6, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        row["screen_reasons"][:56],
        (6, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, body])


def write_contact_sheets(output, candidates):
    per_page = SHEET_COLUMNS * SHEET_ROWS
    page_count = math.ceil(len(candidates) / per_page)
    paths = []
    for page in range(page_count):
        start = page * per_page
        subset = candidates[start:start + per_page]
        tiles = [
            make_tile(row, start + index + 1, len(candidates))
            for index, row in enumerate(subset)
        ]
        blank = np.full_like(tiles[0], 25)
        tiles.extend([blank] * (per_page - len(tiles)))
        grid = []
        for row_index in range(SHEET_ROWS):
            begin = row_index * SHEET_COLUMNS
            grid.append(np.hstack(
                tiles[begin:begin + SHEET_COLUMNS]
            ))
        sheet = np.vstack(grid)
        path = output / "contact_sheets_4x4" / (
            f"page_{page + 1:02d}.jpg"
        )
        write_image(path, sheet)
        paths.append(path)
    return paths


def write_rows(path, rows, fieldnames=None):
    clean_rows = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key != "_image"}
        clean_rows.append(clean)
    if fieldnames is None:
        if not clean_rows:
            raise ValueError("空表需要显式提供 fieldnames")
        fieldnames = list(clean_rows[0])
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(clean_rows)


def main():
    args = parse_args()
    validate_args(args)
    mapping_rows = load_csv(args.mapping)
    if not mapping_rows:
        raise ValueError("mapping.csv 为空")

    screened = []
    for index, row in enumerate(mapping_rows, start=1):
        result = screen_row(row, args.image_root)
        screened.append(result)
        print(
            f"[{index}/{len(mapping_rows)}] "
            f"{result['screen_candidate']} {result['relative_path']} "
            f"{result['screen_reasons']}"
        )

    candidates = [
        row for row in screened if row["screen_candidate"] == "yes"
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    all_path = args.output / "screening_all.csv"
    candidate_path = args.output / "candidate_list.csv"
    write_rows(all_path, screened)
    fieldnames = [key for key in screened[0] if key != "_image"]
    write_rows(candidate_path, candidates, fieldnames=fieldnames)
    sheets = (
        write_contact_sheets(args.output, candidates)
        if candidates
        else []
    )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": (
            "initial_screen_only_pending_human_review"
            if candidates
            else "initial_screen_no_candidates_pending_human_spot_check"
        ),
        "script": str(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "mapping": str(args.mapping),
        "mapping_sha256": file_sha256(args.mapping),
        "image_count": len(screened),
        "candidate_count": len(candidates),
        "candidate_patient_count": len({
            row["patient_id"] for row in candidates
        }),
        "reason_counts": dict(Counter(
            reason
            for row in candidates
            for reason in row["screen_reasons"].split("|")
            if reason
        )),
        "contact_sheet_layout": "4x4",
        "contact_sheet_count": len(sheets),
        "screening_all": str(all_path),
        "candidate_list": str(candidate_path),
        "important_limitation": (
            "初筛候选不等于框错误；必须由人工逐图核查后才能修正或冻结"
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
