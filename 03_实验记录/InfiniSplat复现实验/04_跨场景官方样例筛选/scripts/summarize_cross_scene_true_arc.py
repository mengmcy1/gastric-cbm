#!/usr/bin/env python3
"""Summarize and visualize the five-scene 30-degree true_arc stress test."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = REPO_ROOT / "03_实验记录/InfiniSplat复现实验/04_跨场景官方样例筛选"
RUN_ROOT = EXP_ROOT / "outputs/LARGEANGLE_30deg_true_arc_v1_linux5080_20260814"
PPT_ROOT = (
    REPO_ROOT
    / "临时_PPT_InfiniSplat问题总结_20260814"
    / "新增_跨场景候选_室内道路工业_20260814"
    / "大角度30deg_true_arc_20260814"
)

SCENES = [
    {
        "id": "scannetpp",
        "label": "ScanNetPP 室内",
        "input": REPO_ROOT / "源码/InfiniSplat/examples/data/rgb_demo/scannetpp_fe94fc30cf.JPG",
        "official_video": REPO_ROOT / "03_实验记录/InfiniSplat复现实验/01_官方RGB单图冒烟/outputs/OFFICIAL_scannetpp_scene_calibration_v1_linux5080_20260814_retry2/scannetpp_fe94fc30cf/scannetpp_fe94fc30cf.mp4",
    },
    {
        "id": "my_bedroom",
        "label": "MyBedroom 室内",
        "input": EXP_ROOT / "inputs/selected_rgb_v1/my_bedroom.JPG",
        "official_video": EXP_ROOT / "outputs/OFFICIAL_cross_scene_rgb_batch_v1_linux5080_20260814/my_bedroom/my_bedroom.mp4",
    },
    {
        "id": "waymo_147",
        "label": "Waymo147 道路",
        "input": EXP_ROOT / "inputs/selected_rgb_v1/waymo_147.png",
        "official_video": EXP_ROOT / "outputs/OFFICIAL_cross_scene_rgb_batch_v1_linux5080_20260814/waymo_147/waymo_147.mp4",
    },
    {
        "id": "waymo_9",
        "label": "Waymo9 道路",
        "input": EXP_ROOT / "inputs/selected_rgb_v1/waymo_9.png",
        "official_video": EXP_ROOT / "outputs/OFFICIAL_cross_scene_rgb_batch_v1_linux5080_20260814/waymo_9/waymo_9.mp4",
    },
    {
        "id": "eth3d_kicker",
        "label": "ETH3D Kicker 工业",
        "input": EXP_ROOT / "inputs/selected_rgb_v1/eth3d_kicker.png",
        "official_video": EXP_ROOT / "outputs/OFFICIAL_cross_scene_rgb_batch_v1_linux5080_20260814/eth3d_kicker/eth3d_kicker.mp4",
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    x = left + (right - left - image.width) // 2
    y = top + (bottom - top - image.height) // 2
    canvas.paste(image, (x, y))


def build_rgb_overview(rows: list[dict[str, object]], output_path: Path) -> None:
    cell_w, cell_h, header_h, label_h = 420, 315, 58, 58
    canvas = Image.new("RGB", (cell_w * 4, header_h + len(rows) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 28)
    small = ImageFont.truetype(font_path, 22)
    headers = ["Source 输入", "true_arc -15°", "原相机 0°", "true_arc +15°"]
    for column, title in enumerate(headers):
        draw.text((column * cell_w + 14, 12), title, font=font, fill="#111111")

    for row_index, row in enumerate(rows):
        y0 = header_h + row_index * (cell_h + label_h)
        paths = [row["input_path"], row["left_frame"], row["center_frame"], row["right_frame"]]
        for column, path_text in enumerate(paths):
            image = fit_image(REPO_ROOT / str(path_text), (cell_w - 12, cell_h - 12))
            paste_center(canvas, image, (column * cell_w, y0, (column + 1) * cell_w, y0 + cell_h))
        caption = (
            f"{row['label']}  |  硬空洞 left/center/right: "
            f"{100 * float(row['left_hard_hole']):.1f}% / "
            f"{100 * float(row['center_hard_hole']):.1f}% / "
            f"{100 * float(row['right_hard_hole']):.1f}%"
        )
        draw.text((14, y0 + cell_h + 10), caption, font=small, fill="#111111")
        draw.line((0, y0 + cell_h + label_h - 1, canvas.width, y0 + cell_h + label_h - 1), fill="#cccccc", width=1)
    canvas.save(output_path, quality=94)


def build_alpha_overview(rows: list[dict[str, object]], output_path: Path) -> None:
    cell_w, cell_h, header_h, label_h = 420, 315, 58, 52
    canvas = Image.new("RGB", (cell_w * 3, header_h + len(rows) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 28)
    small = ImageFont.truetype(font_path, 22)
    for column, title in enumerate(["Alpha -15°", "Alpha 0°", "Alpha +15°"]):
        draw.text((column * cell_w + 14, 12), title, font=font, fill="#111111")
    for row_index, row in enumerate(rows):
        y0 = header_h + row_index * (cell_h + label_h)
        for column, suffix in enumerate(["left", "center", "right"]):
            with Image.open(RUN_ROOT / str(row["id"]) / f"alpha_{suffix}_u16.png") as source:
                alpha = np.asarray(source, dtype=np.float32) / 65535.0
            # Coverage is white; hard holes are highlighted red for fast PPT inspection.
            gray = np.round(alpha * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
            rgb[alpha < 0.5] = np.array([220, 35, 35], dtype=np.uint8)
            image = Image.fromarray(rgb)
            image.thumbnail((cell_w - 12, cell_h - 12), Image.Resampling.LANCZOS)
            paste_center(canvas, image, (column * cell_w, y0, (column + 1) * cell_w, y0 + cell_h))
        draw.text((14, y0 + cell_h + 8), str(row["label"]), font=small, fill="#111111")
        draw.line((0, y0 + cell_h + label_h - 1, canvas.width, y0 + cell_h + label_h - 1), fill="#cccccc", width=1)
    canvas.save(output_path, quality=94)


def main() -> None:
    output_csv = EXP_ROOT / "outputs/large_angle_30deg_true_arc_summary.csv"
    output_json = EXP_ROOT / "outputs/large_angle_30deg_true_arc_summary.json"
    record_json = EXP_ROOT / "records/large_angle_30deg_true_arc_v1.json"
    output_rgb = EXP_ROOT / "outputs/large_angle_30deg_true_arc_five_scene_overview.jpg"
    output_alpha = EXP_ROOT / "outputs/large_angle_30deg_true_arc_alpha_overview.jpg"
    rows: list[dict[str, object]] = []

    for scene in SCENES:
        scene_dir = RUN_ROOT / str(scene["id"])
        result = json.loads((scene_dir / "result.json").read_text(encoding="utf-8"))
        metrics = load_metrics(scene_dir / "per_frame_metrics.csv")
        holes = [row["alpha_hard_hole_lt_050"] for row in metrics]
        tops = [row["top_10pct_hard_hole_lt_050"] for row in metrics]
        worst_index = int(np.argmax(holes))
        row = {
            "id": scene["id"],
            "label": scene["label"],
            "trajectory": "synthetic true_arc; not a real target pose",
            "angle_total_deg": result["angle_total_deg"],
            "num_steps": result["num_steps"],
            "focus_depth": result["focus_depth"],
            "left_hard_hole": result["endpoints"]["left"]["alpha_hard_hole_lt_050"],
            "center_hard_hole": result["endpoints"]["center"]["alpha_hard_hole_lt_050"],
            "right_hard_hole": result["endpoints"]["right"]["alpha_hard_hole_lt_050"],
            "mean_hard_hole": float(np.mean(holes)),
            "worst_hard_hole": holes[worst_index],
            "worst_frame": int(metrics[worst_index]["frame"]),
            "worst_angle_deg": metrics[worst_index]["angle_deg"],
            "mean_top_10pct_hard_hole": float(np.mean(tops)),
            "input_path": rel(Path(scene["input"])),
            "official_demo_video": rel(Path(scene["official_video"])),
            "true_arc_video": rel(scene_dir / "color.mp4"),
            "left_frame": rel(scene_dir / "frame_left.png"),
            "center_frame": rel(scene_dir / "frame_center.png"),
            "right_frame": rel(scene_dir / "frame_right.png"),
            "result_path": rel(scene_dir / "result.json"),
        }
        rows.append(row)

    fieldnames = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "1.0-infinisplat-cross-scene-true-arc-summary",
        "date": "2026-08-14",
        "status": "success",
        "protocol": "30-degree total range (-15 to +15), 61 frames, frozen RGB-only PLY, original camera at center",
        "critical_scope": "synthetic camera stress test; no target image or real target pose is claimed",
        "runtime": {
            "physical_gpu": 2,
            "selection_reason": "GPU1 had active Python compute; GPU2 was idle",
            "existing_processes_terminated": False,
        },
        "scenes": rows,
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    output_json.write_text(serialized, encoding="utf-8")
    record_json.parent.mkdir(parents=True, exist_ok=True)
    record_json.write_text(serialized, encoding="utf-8")
    build_rgb_overview(rows, output_rgb)
    build_alpha_overview(rows, output_alpha)

    PPT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_csv, PPT_ROOT / "01_30deg_true_arc_five_scene_summary.csv")
    shutil.copy2(output_json, PPT_ROOT / "02_30deg_true_arc_five_scene_summary.json")
    shutil.copy2(output_rgb, PPT_ROOT / "03_五场景_输入_负15_中心_正15度总览.jpg")
    shutil.copy2(output_alpha, PPT_ROOT / "04_五场景_Alpha硬空洞总览_红色为空洞.jpg")
    for index, (scene, row) in enumerate(zip(SCENES, rows), start=1):
        scene_ppt = PPT_ROOT / f"{index:02d}_{scene['id']}"
        scene_ppt.mkdir(parents=True, exist_ok=True)
        copies = {
            Path(scene["input"]): "01_source_input" + Path(scene["input"]).suffix,
            Path(scene["official_video"]): "02_original_official_demo_small_trajectory.mp4",
            RUN_ROOT / str(scene["id"]) / "color.mp4": "03_true_arc_30deg_total.mp4",
            RUN_ROOT / str(scene["id"]) / "frame_left.png": "04_frame_minus15deg.png",
            RUN_ROOT / str(scene["id"]) / "frame_center.png": "05_frame_center_0deg.png",
            RUN_ROOT / str(scene["id"]) / "frame_right.png": "06_frame_plus15deg.png",
            RUN_ROOT / str(scene["id"]) / "alpha_left_u16.png": "07_alpha_minus15deg_u16.png",
            RUN_ROOT / str(scene["id"]) / "alpha_center_u16.png": "08_alpha_center_u16.png",
            RUN_ROOT / str(scene["id"]) / "alpha_right_u16.png": "09_alpha_plus15deg_u16.png",
            RUN_ROOT / str(scene["id"]) / "per_frame_metrics.csv": "10_per_frame_metrics.csv",
            RUN_ROOT / str(scene["id"]) / "result.json": "11_result.json",
        }
        for source, name in copies.items():
            shutil.copy2(source, scene_ppt / name)
    print(json.dumps({"scenes": len(rows), "ppt_root": rel(PPT_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
