#!/usr/bin/env python3
"""按病例展示RP-SAE Top Feature、空间位置和多Feature重合。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont, ImageOps

from clong_rpd_render_core import display_response, render_heatmap, render_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAE_ROOT = PROJECT_ROOT / "结果/SAE"
RPA_CACHE = SAE_ROOT / "RP_A_Development_20260824/analysis_cache/seed42"
SPATIAL_CACHE = SAE_ROOT / "CLong_S2b结构重构_20260820/frozen_spatial_cache"
SPATIAL_METADATA = SPATIAL_CACHE / "train_metadata.csv"
CLONG_ATTENTION = SPATIAL_CACHE / "train_attention.npy"
RPB_MASTER = SAE_ROOT / "RP_B_Technical_20260825/anchor_master/anchor_master.csv"
RPD_SELECTION = SAE_ROOT / "RP_D_Technical_Atlas_20260825/manifest_dry_run_v1_retry1"
RPD_RENDER = SAE_ROOT / "RP_D_Technical_Atlas_20260825/render_v1_retry2"
DEFAULT_OUTPUT = (
    SAE_ROOT / "RP_SAE完整阶段成果_医学生提交版_v1_20260826/11_C-long注意力与SAE对照图谱_v1"
)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
ACTIVE_EPS = 1e-8
TOP_N = 6
OVERLAP_THRESHOLD_Q99 = 0.5
COLORS = np.asarray([
    [213, 62, 79], [50, 136, 189], [244, 109, 67],
    [102, 194, 165], [171, 85, 182], [230, 171, 2],
], dtype=np.float32) / 255.0
def parse_args() -> argparse.Namespace:
    """读取输出路径和可选的小规模验收数量。"""
    parser = argparse.ArgumentParser(description="Build image-centered RP-SAE Atlas")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="0表示处理全部1770张冻结Atlas图像")
    parser.add_argument(
        "--image-indices", type=str, default="",
        help="可选的逗号分隔image_index；用于定向复核病例",
    )
    return parser.parse_args()


def load_inputs(
    limit: int, requested_image_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.memmap, np.memmap]:
    """加载冻结病例、149个Anchor元数据、Q99和seed42空间激活。"""
    required = [
        RPA_CACHE / "train_image_activations.npy",
        RPA_CACHE / "train_images.csv",
        SPATIAL_METADATA,
        CLONG_ATTENTION,
        RPB_MASTER,
        RPD_SELECTION / "rpd_anchor_manifest.csv",
        RPD_SELECTION / "rpd_case_manifest.csv",
        RPD_RENDER / "q99_scales.csv",
        FONT_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少逐图图谱输入:\n" + "\n".join(missing))

    cases = pd.read_csv(RPD_SELECTION / "rpd_case_manifest.csv", low_memory=False)
    images = cases.sort_values(["image_index", "case_id"]).drop_duplicates("image_index").copy()
    images = images[["image_index", "patient_id", "label", "source", "image_relpath"]]
    images = images.sort_values("image_index").reset_index(drop=True)
    if images.image_index.nunique() != 1770:
        raise RuntimeError(f"冻结RP-D唯一图像应为1770，实际={images.image_index.nunique()}")
    if requested_image_indices:
        requested = set(requested_image_indices)
        images = images[images.image_index.isin(requested)].copy()
        missing_indices = sorted(requested - set(images.image_index.astype(int)))
        if missing_indices:
            raise ValueError(f"指定image_index不在冻结Atlas中: {missing_indices}")
        images = images.sort_values("image_index").reset_index(drop=True)
    if limit > 0:
        images = images.head(limit).copy()

    spatial_metadata = pd.read_csv(SPATIAL_METADATA, low_memory=False)
    if len(spatial_metadata) != 2350 or "cancer_probability" not in spatial_metadata:
        raise RuntimeError("冻结空间metadata缺少完整C-long癌概率")
    image_indices = images.image_index.to_numpy(np.int64)
    expected_paths = spatial_metadata.iloc[image_indices].image_relpath.astype(str).to_numpy()
    if not np.array_equal(expected_paths, images.image_relpath.astype(str).to_numpy()):
        raise RuntimeError("冻结病例image_index与空间metadata行顺序不一致")
    images["cancer_probability"] = spatial_metadata.iloc[image_indices].cancer_probability.to_numpy()

    anchor_manifest = pd.read_csv(RPD_SELECTION / "rpd_anchor_manifest.csv").rename(
        columns={"feature_42": "feature_id"}
    )
    q99 = pd.read_csv(RPD_RENDER / "q99_scales.csv")
    q99 = q99[q99.seed.eq(42)][["anchor_id", "feature_id", "q99_scale"]]
    anchors = anchor_manifest.merge(q99, on=["anchor_id", "feature_id"], validate="one_to_one")
    if len(anchors) != 149 or anchors.q99_scale.le(0).any():
        raise RuntimeError("seed42的149个Anchor或Q99不完整")
    anchors = anchors.merge(
        pd.read_csv(RPB_MASTER)[[
            "anchor_id", "sharedness_class", "cancer_coverage", "noncancer_coverage",
            "cancer_mass", "noncancer_mass", "label_auc", "energy_percentile",
        ]],
        on="anchor_id", validate="one_to_one",
    )
    anchors = anchors.sort_values("anchor_id").reset_index(drop=True)

    activation = np.load(RPA_CACHE / "train_image_activations.npy", mmap_mode="r")
    if activation.shape != (2350, 49, 10240):
        raise RuntimeError(f"seed42空间激活shape异常: {activation.shape}")
    attention = np.load(CLONG_ATTENTION, mmap_mode="r")
    if attention.shape != (2350, 49):
        raise RuntimeError(f"C-long注意力shape异常: {attention.shape}")
    selected_attention = np.asarray(attention[image_indices], dtype=np.float64)
    if not np.isfinite(selected_attention).all():
        raise RuntimeError("C-long注意力包含NaN/Inf")
    if not np.allclose(selected_attention.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("C-long注意力逐图和不为1")
    return images, anchors, activation, attention


def attention_alignment(
    response: np.ndarray, attention: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """比较同一7x7网格上的Feature激活与C-long原始注意力。"""
    feature = np.clip(np.asarray(response, dtype=np.float64).reshape(-1), 0.0, None)
    original_attention = np.asarray(attention, dtype=np.float64).reshape(-1)
    feature_sum = float(feature.sum())
    if feature_sum <= ACTIVE_EPS:
        return None, None, None
    feature_distribution = feature / feature_sum
    attention_distribution = original_attention / original_attention.sum()
    overlap = float(np.minimum(feature_distribution, attention_distribution).sum())
    denominator = float(
        np.linalg.norm(feature_distribution) * np.linalg.norm(attention_distribution)
    )
    cosine = float(np.dot(feature_distribution, attention_distribution) / denominator)
    feature_peak = np.asarray(np.unravel_index(np.argmax(feature), (7, 7)))
    attention_peak = np.asarray(np.unravel_index(np.argmax(attention_distribution), (7, 7)))
    peak_distance = float(np.linalg.norm(feature_peak - attention_peak))
    return overlap, cosine, peak_distance


def pad_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """保持纵横比并放入固定白底格。"""
    return ImageOps.pad(image.convert("RGB"), size, color="white", method=Image.Resampling.LANCZOS)


def color_composite(
    original: Image.Image, responses: np.ndarray,
) -> Image.Image:
    """将Top-6 Q99缩放响应按固定颜色软混合到原图。"""
    width, height = original.size
    resized = []
    for response in responses:
        image = Image.fromarray(np.uint8(np.clip(response, 0, 1) * 255)).resize(
            (width, height), Image.Resampling.BILINEAR,
        )
        resized.append(np.asarray(image, dtype=np.float32) / 255.0)
    weights = np.stack(resized, axis=-1)
    weight_sum = weights.sum(axis=-1, keepdims=True)
    color = np.einsum("hwk,kc->hwc", weights, COLORS) / np.maximum(weight_sum, 1e-8)
    alpha = np.clip(weights.max(axis=-1, keepdims=True) * 0.65, 0, 0.65)
    base = np.asarray(original, dtype=np.float32) / 255.0
    mixed = base * (1 - alpha) + color * alpha
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 1) * 255))


def overlap_map(responses: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """显示Top-6中响应达到各自0.5×Q99的Feature重合数。"""
    count = (responses >= OVERLAP_THRESHOLD_Q99).sum(axis=0).astype(np.float32)
    normalized = count / TOP_N
    image = Image.fromarray(np.uint8(normalized * 255)).resize(size, Image.Resampling.NEAREST)
    colored = colormaps["magma"](np.asarray(image, dtype=np.float32) / 255.0)[..., :3]
    return Image.fromarray(np.uint8(colored * 255))


def draw_title(
    canvas: Image.Image, xy: tuple[int, int], title: str, subtitle: str,
    title_font: ImageFont.FreeTypeFont, small_font: ImageFont.FreeTypeFont,
) -> None:
    """绘制一个面板格标题。"""
    draw = ImageDraw.Draw(canvas)
    draw.text(xy, title, fill=(25, 30, 32), font=title_font)
    draw.text((xy[0], xy[1] + 25), subtitle, fill=(75, 82, 85), font=small_font)


def render_panel(
    image_row: pd.Series,
    anchors: pd.DataFrame,
    image_activation: np.ndarray,
    clong_attention: np.ndarray,
    output: Path,
) -> tuple[dict, list[dict]]:
    """生成一张病例中心总览，并返回Top与全149激活记录。"""
    image_path = PROJECT_ROOT / str(image_row.image_relpath)
    with Image.open(image_path) as source:
        original = source.convert("RGB")

    feature_ids = anchors.feature_id.to_numpy(np.int64)
    raw = np.asarray(image_activation[:, feature_ids], dtype=np.float32).T.reshape(-1, 7, 7)
    peak = raw.max(axis=(1, 2))
    mass = raw.mean(axis=(1, 2))
    q99 = anchors.q99_scale.to_numpy(np.float32)
    score = peak / q99
    order = np.lexsort((anchors.anchor_id.to_numpy(str), -score))
    selected = order[:TOP_N]
    responses = np.stack([display_response(raw[index], float(q99[index])) for index in selected])
    attention_7x7 = np.asarray(clong_attention, dtype=np.float32).reshape(7, 7)
    attention_display = attention_7x7 / float(attention_7x7.max())
    selected_alignment = [attention_alignment(raw[index], attention_7x7) for index in selected]

    tile = (420, 315)
    header = 92
    gap = 12
    label_h = 58
    canvas_w = gap * 3 + tile[0] * 2
    row_h = label_h + tile[1]
    canvas_h = header + gap * 7 + row_h * 6
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(FONT_PATH), 19)
    small_font = ImageFont.truetype(str(FONT_PATH), 13)
    tiny_font = ImageFont.truetype(str(FONT_PATH), 11)
    draw.text((gap, 10), f"图像中心RP-SAE多Feature图谱 | image_index={int(image_row.image_index)}",
              fill=(20, 25, 28), font=title_font)
    draw.text((gap, 40),
              f"C-long癌概率={float(image_row.cancer_probability):.3f} | label={int(image_row.label)} | "
              f"source={image_row.source} | active展示Anchor={int((peak > ACTIVE_EPS).sum())}/149",
              fill=(55, 62, 65), font=small_font)
    draw.text((gap, 63), "Top-6按本图峰值 / 各Anchor完整train正激活Q99排序；仅为图像内相对排序",
              fill=(75, 82, 85), font=tiny_font)

    base = pad_image(original, tile)
    top_original = base
    attention_heatmap = render_heatmap(attention_display, base.size)
    attention_overlay = render_overlay(base, attention_heatmap, attention_display, 0.58)
    composite = color_composite(base, responses)
    top_items = [
        ("原图", "同一张冻结train图像", top_original),
        ("C-long原始注意力", "模型分类时实际使用的7x7空间权重", attention_overlay),
        ("Top-6 SAE Feature多颜色叠加", "颜色对应下方Feature；重叠处发生混色", composite),
        ("Top-6响应重合数", "达到各自0.5×Q99；亮色表示重合更多", overlap_map(responses, tile)),
    ]
    for item_index, (title, subtitle, image) in enumerate(top_items):
        row = item_index // 2
        column = item_index % 2
        x = gap + column * (tile[0] + gap)
        y = header + gap + row * (row_h + gap)
        draw_title(canvas, (x, y), title, subtitle, title_font, tiny_font)
        canvas.paste(image, (x, y + label_h))

    for rank, anchor_index in enumerate(selected):
        row = rank // 2 + 2
        column = rank % 2
        x = gap + column * (tile[0] + gap)
        y = header + gap + row * (row_h + gap)
        anchor = anchors.iloc[anchor_index]
        color = tuple(int(value * 255) for value in COLORS[rank])
        draw.rectangle((x, y + 2, x + 13, y + 15), fill=color)
        overlap, _, peak_distance = selected_alignment[rank]
        draw_title(
            canvas, (x + 19, y),
            f"Top {rank + 1}: {anchor.anchor_id} / Feature {int(anchor.feature_id)}",
            f"注意重合={overlap:.3f} | 峰距={peak_distance:.1f}格 | "
            f"Anchor总体干预={float(anchor.median_intermediate_curve_overall_effect):.4f}",
            title_font, tiny_font,
        )
        response = responses[rank]
        heatmap = render_heatmap(response, base.size)
        overlay = render_overlay(base, heatmap, response, 0.58)
        canvas.paste(overlay, (x, y + label_h))

    bottom_y = header + gap + 5 * (row_h + gap)
    draw_title(
        canvas, (gap, bottom_y), "原注意力与SAE Feature怎么比较",
        "重合度0到1；越高表示空间分布越接近。峰距单位为7x7网格格数", title_font, tiny_font,
    )
    explanation = Image.new("RGB", tile, (246, 248, 248))
    explanation_draw = ImageDraw.Draw(explanation)
    explanation_lines = [
        "C-long注意力：模型总体从哪些位置汇总信息。",
        "SAE Feature：某一种内部视觉模式在哪些位置激活。",
        "二者不要求完全相同；Feature激活不等于分类依赖。",
        "请结合干预效应判断模型是否真正依赖该Feature。",
    ]
    for line_index, line in enumerate(explanation_lines):
        explanation_draw.text(
            (14, 18 + line_index * 52), line, fill=(42, 48, 50), font=small_font,
        )
    canvas.paste(explanation, (gap, bottom_y + label_h))
    summary_x = gap * 2 + tile[0]
    draw_title(canvas, (summary_x, bottom_y), "Top-6技术摘要",
               "matched percentile不是p值；source-risk只是排查提示", title_font, tiny_font)
    summary_box = Image.new("RGB", tile, (246, 248, 248))
    summary_draw = ImageDraw.Draw(summary_box)
    for rank, anchor_index in enumerate(selected):
        anchor = anchors.iloc[anchor_index]
        overlap, cosine, peak_distance = selected_alignment[rank]
        color = tuple(int(value * 255) for value in COLORS[rank])
        y_text = 12 + rank * 48
        summary_draw.rectangle((10, y_text + 3, 24, y_text + 17), fill=color)
        summary_draw.text((32, y_text), f"{rank + 1}. {anchor.anchor_id} / F{int(anchor.feature_id)}",
                          fill=(25, 30, 32), font=small_font)
        summary_draw.text(
            (32, y_text + 21),
            f"重合={overlap:.2f}  cosine={cosine:.2f}  峰距={peak_distance:.1f}  "
            f"source-risk={bool(anchor.source_risk)}",
            fill=(72, 78, 80), font=tiny_font,
        )
    canvas.paste(summary_box, (summary_x, bottom_y + label_h))

    stem = Path(str(image_row.image_relpath)).stem
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)[:60]
    filename = f"image_{int(image_row.image_index):04d}_{safe_stem}.png"
    canvas.save(output / filename, optimize=True)

    top_record = {
        "image_index": int(image_row.image_index),
        "patient_id": str(image_row.patient_id),
        "label": int(image_row.label),
        "source": str(image_row.source),
        "clong_cancer_probability": float(image_row.cancer_probability),
        "image_relpath": str(image_row.image_relpath),
        "active_display_anchor_count": int((peak > ACTIVE_EPS).sum()),
        "top_anchor_ids": ";".join(anchors.iloc[selected].anchor_id.astype(str)),
        "top_feature_ids": ";".join(anchors.iloc[selected].feature_id.astype(int).astype(str)),
        "top_relative_q99_scores": ";".join(f"{score[index]:.6g}" for index in selected),
        "top_attention_overlaps": ";".join(
            f"{selected_alignment[rank][0]:.6g}" for rank in range(TOP_N)
        ),
        "top_attention_cosines": ";".join(
            f"{selected_alignment[rank][1]:.6g}" for rank in range(TOP_N)
        ),
        "top_attention_peak_distances": ";".join(
            f"{selected_alignment[rank][2]:.6g}" for rank in range(TOP_N)
        ),
        "panel_path": f"图像中心总览/{filename}",
    }
    all_records = []
    top_rank_by_index = {
        int(anchor_index): rank + 1 for rank, anchor_index in enumerate(selected)
    }
    for index, anchor in anchors.iterrows():
        overlap, cosine, peak_distance = attention_alignment(raw[index], attention_7x7)
        all_records.append({
            "image_index": int(image_row.image_index),
            "patient_id": str(image_row.patient_id),
            "label": int(image_row.label),
            "source": str(image_row.source),
            "anchor_id": str(anchor.anchor_id),
            "feature_id": int(anchor.feature_id),
            "peak_activation": float(peak[index]),
            "mass_activation": float(mass[index]),
            "q99_scale": float(q99[index]),
            "relative_q99_score": float(score[index]),
            "top6_rank": top_rank_by_index.get(int(index)),
            "clong_attention_overlap": overlap,
            "clong_attention_cosine": cosine,
            "clong_attention_peak_distance_grid": peak_distance,
            "active": bool(peak[index] > ACTIVE_EPS),
            "sharedness_class": str(anchor.sharedness_class),
            "source_risk": bool(anchor.source_risk),
            "functional_pattern": str(anchor.functional_pattern),
            "median_intermediate_curve_overall_effect": float(
                anchor.median_intermediate_curve_overall_effect
            ),
        })
    return top_record, all_records


def write_html(summary: pd.DataFrame, path: Path) -> None:
    """生成可按病例、标签和来源筛选的逐图图册。"""
    cards = []
    for row in summary.itertuples(index=False):
        cards.append(f"""
<article class="card" data-label="{row.label}" data-source="{html.escape(str(row.source))}">
 <a href="{html.escape(row.panel_path)}"><img loading="lazy" src="{html.escape(row.panel_path)}"></a>
 <div><strong>image {row.image_index}</strong><span>label={row.label} | {html.escape(str(row.source))}</span>
 <span>Top: {html.escape(str(row.top_anchor_ids))}</span></div></article>""")
    sources = sorted(summary.source.astype(str).unique())
    source_options = "".join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in sources)
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>逐图多Feature图谱</title><style>
body{{margin:0;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#f5f7f7;color:#202428}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd4d6;padding:12px 18px;z-index:2}}
h1{{font-size:21px;margin:0 0 9px}} .controls{{display:flex;gap:9px;flex-wrap:wrap}}
input,select{{height:34px;border:1px solid #98a5a9;border-radius:4px;padding:0 8px;background:#fff}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px;padding:14px}}
.card{{background:#fff;border:1px solid #d4dcde;border-radius:6px;overflow:hidden}}
.card img{{display:block;width:100%;height:330px;object-fit:contain}} .card div{{display:grid;gap:4px;padding:9px;font-size:12px}}
.card strong{{font-size:15px}} .hidden{{display:none}}</style></head><body><header>
<h1>图像中心RP-SAE多Feature图谱：{len(summary)}张冻结Atlas图像</h1><div class="controls">
<input id="query" placeholder="图像编号或Anchor"><select id="label"><option value="">全部标签</option>
<option value="1">癌</option><option value="0">非癌</option></select><select id="source"><option value="">全部来源</option>{source_options}</select>
</div></header><main>{''.join(cards)}</main><script>
const controls=[...document.querySelectorAll('input,select')];function go(){{const q=document.querySelector('#query').value.toLowerCase(),
l=document.querySelector('#label').value,s=document.querySelector('#source').value;document.querySelectorAll('.card').forEach(c=>{{
const ok=(!q||c.textContent.toLowerCase().includes(q))&&(!l||c.dataset.label===l)&&(!s||c.dataset.source===s);
c.classList.toggle('hidden',!ok);}})}}controls.forEach(x=>x.addEventListener('input',go));</script></body></html>"""
    path.write_text(content, encoding="utf-8")


def write_readme(path: Path, image_count: int) -> None:
    """写入医生阅读边界和显示规则。"""
    path.write_text(f"""# 图像中心RP-SAE多Feature图谱

## 目的

本目录从“病例”出发，展示同一张图像同时激活了哪些已进入RP-D展示的Anchor，以及这些Anchor
分别对应图像中的哪些位置。共覆盖当前正式Atlas使用的{image_count}张不同train图像。

## 每张总览怎么看

1. 原图；
2. C-long原始注意力图；
3. Top-6多颜色叠加图；
4. Top-6响应重合数图；
5. 六个Feature各自的单独overlay，并报告与原注意力的重合度和峰值距离。

Top Feature按“本图空间峰值 / 该Anchor完整train正激活Q99”排序。这个相对分数只用于同一张图
内选择突出Feature，不能解释为Feature A在医学上比Feature B强多少。重合数使用0.5×各自Q99作为
显示阈值，仅是空间可视化诊断，不参与科学门槛或Feature筛选。

“注意重合”先把C-long注意力和单Feature非负激活分别归一化为总和1，再计算逐网格最小值之和，
范围0到1，越高表示两种空间分布越接近。“峰距”是两张7x7图最高响应网格之间的欧氏距离。
这些指标只用于回答原注意力与SAE Feature关注位置是否一致，不参与选模或Feature排序。
图中“Anchor总体干预”来自RP-C2跨病例汇总，不是当前单张图的个体干预效应。

`逐图全部149Feature激活.csv`保存每张图对全部149个展示Anchor的激活，而总览只画Top-6，
因此Top-6不表示模型只激活了六个Feature。

## 证据边界

- canonical SAE为seed42，与RP-D一致；
- 只复用正式train空间缓存，不重新训练或推理；
- 只分析149个RP-D展示Anchor，不代表模型全部10240个SAE Feature；
- Feature重合不等于医学概念相同，也不等于两个Feature可以合并；
- 热图不是病灶边界，医学含义仍需医学生或医生判断。
""", encoding="utf-8")


def main() -> None:
    """生成病例中心总览、全149激活长表和HTML图册。"""
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
    requested_image_indices = [
        int(value.strip()) for value in args.image_indices.split(",") if value.strip()
    ]
    images, anchors, activation, attention = load_inputs(args.limit, requested_image_indices)
    panel_root = output / "图像中心总览"
    panel_root.mkdir(parents=True)

    top_records: list[dict] = []
    all_records: list[dict] = []
    for number, image_row in enumerate(images.itertuples(index=False), start=1):
        top, records = render_panel(
            pd.Series(image_row._asdict()), anchors,
            activation[int(image_row.image_index)],
            attention[int(image_row.image_index)], panel_root,
        )
        top_records.append(top)
        all_records.extend(records)
        if number % 25 == 0 or number == len(images):
            print(f"rendered {number}/{len(images)}", flush=True)

    summary = pd.DataFrame(top_records)
    summary.to_csv(output / "逐图Top6_Feature汇总.csv", index=False, encoding="utf-8-sig")
    all_frame = pd.DataFrame(all_records)
    all_frame.to_csv(output / "逐图全部149Feature激活.csv", index=False, encoding="utf-8-sig")
    top6_frame = all_frame[all_frame.top6_rank.notna()].copy()
    summary_rows = []
    for group_name, frame in [("全部", top6_frame), ("非癌", top6_frame[top6_frame.label.eq(0)]),
                              ("癌", top6_frame[top6_frame.label.eq(1)])]:
        summary_rows.append({
            "分组": group_name,
            "图像数": int(frame.image_index.nunique()),
            "Top6记录数": int(len(frame)),
            "注意重合均值": float(frame.clong_attention_overlap.mean()),
            "注意重合中位数": float(frame.clong_attention_overlap.median()),
            "注意重合Q25": float(frame.clong_attention_overlap.quantile(0.25)),
            "注意重合Q75": float(frame.clong_attention_overlap.quantile(0.75)),
            "重合度不低于0.5比例": float(frame.clong_attention_overlap.ge(0.5).mean()),
            "峰距中位数_网格": float(frame.clong_attention_peak_distance_grid.median()),
            "source_risk比例": float(frame.source_risk.mean()),
        })
    pd.DataFrame(summary_rows).to_csv(
        output / "C-long注意力与Top6_SAE空间一致性汇总.csv",
        index=False, encoding="utf-8-sig",
    )
    write_html(summary, output / "逐图多Feature图册.html")
    write_readme(output / "README_图片怎么看.md", len(images))
    config = {
        "status": "image_centered_atlas_complete",
        "image_count": len(images),
        "display_anchor_count": len(anchors),
        "canonical_seed": 42,
        "top_n": TOP_N,
        "ranking": "image_peak_activation / frozen_anchor_train_positive_q99",
        "overlap_display_threshold": "0.5 * frozen_anchor_train_positive_q99",
        "clong_attention_comparison": {
            "source": str(CLONG_ATTENTION.relative_to(PROJECT_ROOT)),
            "grid": [7, 7],
            "overlap": "sum(min(L1_normalized_feature, L1_normalized_attention))",
            "peak_distance": "euclidean distance between argmax cells in 7x7 grid",
            "selection_role": "diagnostic_only",
        },
        "diagnostic_only": True,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
