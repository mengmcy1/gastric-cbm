"""重新排版已有 MOCE 聚类结果，不重新提取特征或执行聚类。"""

import argparse
import math
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
RESULT_DIR = os.path.join(
    PROJECT_DIR, "结果", "MOCE聚类", "概念严格平衡_v1", "full"
)

MODEL_NAMES = ["resnet50", "efficientnet_b0"]
CLASS_NAMES = {0: "非癌", 1: "癌/高级别"}
REPRESENTATIVES = 5
CLUSTERS_PER_PAGE = 3

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
FONT_INDEX = 2  # Noto Serif CJK SC，简体中文宋体风格
TITLE_FONT = ImageFont.truetype(FONT_PATH, 34, index=FONT_INDEX)
CLUSTER_FONT = ImageFont.truetype(FONT_PATH, 28, index=FONT_INDEX)
TEXT_FONT = ImageFont.truetype(FONT_PATH, 22, index=FONT_INDEX)
SMALL_FONT = ImageFont.truetype(FONT_PATH, 18, index=FONT_INDEX)

PATCH_SIZE = (300, 250)
LABEL_WIDTH = 330
PANEL_GAP = 10
HEADER_HEIGHT = 90
ROW_HEIGHT = 325
CANVAS_WIDTH = LABEL_WIDTH + (PATCH_SIZE[0] + PANEL_GAP) * REPRESENTATIVES


def fit_patch(path):
    """保持比例，把候选区域放入放大的固定面板。"""
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, PATCH_SIZE, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", PATCH_SIZE, "black")
    panel.paste(
        image,
        ((PATCH_SIZE[0] - image.width) // 2, (PATCH_SIZE[1] - image.height) // 2),
    )
    return panel


def load_clusters(class_dir):
    """读取聚类结果，选取每簇最接近中心的代表区域。"""
    assignment = pd.read_csv(
        os.path.join(class_dir, "cluster_assignments.csv"), encoding="utf-8-sig"
    )
    importance = pd.read_csv(
        os.path.join(class_dir, "concept_importance.csv"), encoding="utf-8-sig"
    ).set_index("cluster_id")

    clusters = []
    records = []
    for cluster_id in sorted(assignment["cluster_id"].unique()):
        members = assignment[assignment["cluster_id"] == cluster_id]
        representatives = members.nsmallest(REPRESENTATIVES, "distance_to_center")
        score = importance.loc[cluster_id]
        clusters.append((int(cluster_id), members, representatives, score))

        for order, row in enumerate(representatives.itertuples(index=False), 1):
            records.append({
                "concept_number": int(cluster_id) + 1,
                "cluster_id": int(cluster_id),
                "representative_order": order,
                "importance_rank": int(score["importance_rank"]),
                "patient_id": row.patient_id,
                "image_name": row.image_name,
                "patch_file": row.patch_file,
                "mask_file": row.mask_file,
                "distance_to_center": row.distance_to_center,
            })
    return clusters, pd.DataFrame(records)


def draw_page(model_name, label, page_clusters, page, total_pages, output_path):
    """绘制一页概念簇，每页展示三个概念簇。"""
    height = HEADER_HEIGHT + ROW_HEIGHT * len(page_clusters)
    canvas = Image.new("RGB", (CANVAS_WIDTH, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{model_name}  类别{label}（{CLASS_NAMES[label]}）MOCE概念聚类"
        f"  第{page}/{total_pages}页"
    )
    draw.text((24, 20), title, fill=(20, 20, 20), font=TITLE_FONT)

    for row_index, (cluster_id, members, representatives, score) in enumerate(page_clusters):
        y = HEADER_HEIGHT + row_index * ROW_HEIGHT
        background = (250, 248, 243) if row_index % 2 == 0 else "white"
        draw.rectangle((0, y, CANVAS_WIDTH, y + ROW_HEIGHT), fill=background)
        draw.line((0, y, CANVAS_WIDTH, y), fill=(205, 195, 180), width=2)

        draw.text((20, y + 28), f"概念簇 {cluster_id + 1:02d}", fill=(25, 25, 25), font=CLUSTER_FONT)
        draw.text((20, y + 80), f"候选区域：{len(members)} 个", fill=(65, 65, 65), font=TEXT_FONT)
        draw.text((20, y + 120), f"患者：{members['patient_id'].nunique()} 位", fill=(65, 65, 65), font=TEXT_FONT)
        draw.text((20, y + 160), f"重要性排名：{int(score['importance_rank'])}", fill=(65, 65, 65), font=TEXT_FONT)
        draw.text((20, y + 200), f"S_h：{score['S_h']:.3f}", fill=(65, 65, 65), font=TEXT_FONT)

        for column, representative in enumerate(representatives.itertuples(index=False)):
            x = LABEL_WIDTH + column * (PATCH_SIZE[0] + PANEL_GAP)
            canvas.paste(fit_patch(representative.patch_file), (x, y + 10))
            caption = f"代表区域 {column + 1}"
            caption_width = draw.textlength(caption, font=SMALL_FONT)
            draw.text(
                (x + (PATCH_SIZE[0] - caption_width) / 2, y + PATCH_SIZE[1] + 18),
                caption,
                fill=(40, 40, 40),
                font=SMALL_FONT,
            )

    canvas.save(output_path, pnginfo=None)


def render_class(model_name, label):
    """为一个模型类别生成分页清晰版总览和代表区域索引。"""
    class_dir = os.path.join(RESULT_DIR, model_name, f"class_{label}")
    assignment_path = os.path.join(class_dir, "cluster_assignments.csv")
    if not os.path.isfile(assignment_path):
        print(f"跳过尚未完成的结果：{model_name}/class_{label}")
        return

    output_dir = os.path.join(class_dir, "概念聚类清晰版")
    os.makedirs(output_dir, exist_ok=True)
    clusters, representative_index = load_clusters(class_dir)
    total_pages = math.ceil(len(clusters) / CLUSTERS_PER_PAGE)

    for page_index in range(total_pages):
        start = page_index * CLUSTERS_PER_PAGE
        output_path = os.path.join(
            output_dir, f"concept_clusters_page_{page_index + 1:02d}.png"
        )
        draw_page(
            model_name,
            label,
            clusters[start:start + CLUSTERS_PER_PAGE],
            page_index + 1,
            total_pages,
            output_path,
        )

    representative_index.to_csv(
        os.path.join(output_dir, "代表区域索引.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已生成：{output_dir}")


def main():
    parser = argparse.ArgumentParser(description="重新排版已有MOCE概念聚类图")
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], default="all")
    parser.add_argument("--class-label", choices=["all", "0", "1"], default="all")
    args = parser.parse_args()

    models = MODEL_NAMES if args.model == "all" else [args.model]
    labels = [0, 1] if args.class_label == "all" else [int(args.class_label)]
    for model_name in models:
        for label in labels:
            render_class(model_name, label)


if __name__ == "__main__":
    main()
