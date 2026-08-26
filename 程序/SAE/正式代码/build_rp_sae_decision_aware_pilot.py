#!/usr/bin/env python3
"""生成两个冻结train病例的语义Top-6与功能Top-6对照图。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from build_rp_sae_image_centered_atlas import (
    ACTIVE_EPS,
    CLONG_ATTENTION,
    FONT_PATH,
    PROJECT_ROOT,
    RPA_CACHE,
    RPB_MASTER,
    RPD_RENDER,
    RPD_SELECTION,
    SPATIAL_METADATA,
    attention_alignment,
    pad_image,
)
from clong_rpd_render_core import display_response, render_heatmap, render_overlay
from clong_s2b_core import cell_overlap_map


SAE_ROOT = PROJECT_ROOT / "结果/SAE"
SPATIAL_CACHE = SAE_ROOT / "CLong_S2b结构重构_20260820/frozen_spatial_cache"
RPC2_EFFECT_ROOT = (
    SAE_ROOT
    / "RP_C2_Intervention_20260825/formal_retry1/unique_feature_facts/seed42/target_patient_effects"
)
CLONG_CHECKPOINT = (
    PROJECT_ROOT
    / "结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选/"
    "mg2l_armc_efficientnet_b0_seed42/mg2_armc_best_student.pth"
)
OUTPUT_ROOT = SAE_ROOT / "RP_D_Decision_Aware_Pilot_20260826_retry1"
IMAGE_INDICES = (42, 1245)
TOP_N = 6


def fixed_attention_local_ablation_map(
    activation: np.ndarray, attention: np.ndarray, decoder_direction: float,
) -> np.ndarray:
    """计算删除单Feature时固定原注意力的逐位置margin变化。"""
    h = np.asarray(activation, dtype=np.float64).reshape(7, 7)
    a = np.asarray(attention, dtype=np.float64).reshape(7, 7)
    return -a * h * float(decoder_direction)


def grid_edge_mass_ratio(values: np.ndarray) -> float | None:
    """计算7x7最外圈占非负响应总质量的比例。"""
    array = np.clip(np.asarray(values, dtype=np.float64).reshape(7, 7), 0.0, None)
    total = float(array.sum())
    if total <= ACTIVE_EPS:
        return None
    edge = np.zeros((7, 7), dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return float(array[edge].sum() / total)


def bbox_metrics(
    raw: np.ndarray, gated: np.ndarray, direct: np.ndarray, bbox: np.ndarray | None,
) -> dict[str, float | None]:
    """按冻结7x7面积相交口径计算非负图和有符号图的框内统计。"""
    empty = {
        "raw_bbox_mass": None,
        "gated_bbox_mass": None,
        "direct_bbox_signed_sum": None,
        "direct_bbox_absolute_mass_fraction": None,
    }
    if bbox is None:
        return empty
    overlap = cell_overlap_map(np.asarray(bbox, dtype=np.float64))
    raw_array = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
    gated_array = np.clip(np.asarray(gated, dtype=np.float64), 0.0, None)
    direct_array = np.asarray(direct, dtype=np.float64)
    raw_total = float(raw_array.sum())
    gated_total = float(gated_array.sum())
    direct_abs_total = float(np.abs(direct_array).sum())
    return {
        "raw_bbox_mass": float((raw_array / raw_total * overlap).sum()),
        "gated_bbox_mass": float((gated_array / gated_total * overlap).sum()),
        "direct_bbox_signed_sum": float((direct_array * overlap).sum()),
        "direct_bbox_absolute_mass_fraction": float(
            (np.abs(direct_array) * overlap).sum() / direct_abs_total
        ),
    }


def normalize_positive(values: np.ndarray) -> np.ndarray:
    """仅用于单幅空间图显示的峰值归一化。"""
    array = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    maximum = float(array.max())
    return array / maximum if maximum > 0 else np.zeros_like(array)


def signed_overlay(original: Image.Image, values: np.ndarray) -> Image.Image:
    """显示有符号局部删除效应；红增癌margin，蓝降癌margin。"""
    array = np.asarray(values, dtype=np.float32).reshape(7, 7)
    scale = float(np.abs(array).max())
    normalized = array / scale if scale > 0 else np.zeros_like(array)
    upsampled = Image.fromarray(np.uint8((normalized + 1.0) * 127.5)).resize(
        original.size, Image.Resampling.BILINEAR,
    )
    signed = np.asarray(upsampled, dtype=np.float32) / 127.5 - 1.0
    color = colormaps["coolwarm"]((signed + 1.0) / 2.0)[..., :3] * 255.0
    alpha = (np.abs(signed) * 0.68)[..., None]
    base = np.asarray(original, dtype=np.float32)
    mixed = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 255)))


def draw_bbox(image: Image.Image, bbox: np.ndarray | None) -> Image.Image:
    """在癌图原图上绘制冻结病灶框。"""
    output = image.copy()
    if bbox is None:
        return output
    width, height = output.size
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    ImageDraw.Draw(output).rectangle(
        (x1 * width, y1 * height, x2 * width, y2 * height),
        outline=(35, 190, 80), width=max(2, round(min(width, height) / 180)),
    )
    return output


def load_checkpoint_state() -> dict[str, torch.Tensor]:
    """读取冻结C-long学生权重并返回state_dict。"""
    payload = torch.load(CLONG_CHECKPOINT, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    if isinstance(payload, dict) and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise RuntimeError("无法从C-long checkpoint读取state_dict")


def load_inputs() -> dict:
    """载入并对齐pilot所需的冻结输入。"""
    required = [
        RPA_CACHE / "train_image_activations.npy",
        RPA_CACHE / "decoder_weight.npy",
        RPA_CACHE / "train_images.csv",
        CLONG_ATTENTION,
        SPATIAL_METADATA,
        RPB_MASTER,
        RPD_SELECTION / "rpd_anchor_manifest.csv",
        RPD_RENDER / "q99_scales.csv",
        RPC2_EFFECT_ROOT / "target_feature_ids.npy",
        RPC2_EFFECT_ROOT / "image_delta_margin_alpha_0p00.npy",
        CLONG_CHECKPOINT,
        FONT_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("pilot缺少冻结输入:\n" + "\n".join(missing))

    metadata = pd.read_csv(SPATIAL_METADATA, low_memory=False)
    images = pd.read_csv(RPA_CACHE / "train_images.csv", low_memory=False)
    if len(metadata) != 2350 or len(images) != 2350:
        raise RuntimeError("冻结train图像数不是2350")
    if not np.array_equal(metadata.relative_path.astype(str), images.relative_path.astype(str)):
        raise RuntimeError("空间metadata与RP-A图像行顺序不一致")

    anchors = pd.read_csv(RPD_SELECTION / "rpd_anchor_manifest.csv").rename(
        columns={"feature_42": "feature_id"}
    )
    q99 = pd.read_csv(RPD_RENDER / "q99_scales.csv")
    q99 = q99[q99.seed.eq(42)][["anchor_id", "feature_id", "q99_scale"]]
    anchors = anchors.merge(q99, on=["anchor_id", "feature_id"], validate="one_to_one")
    anchors = anchors.merge(
        pd.read_csv(RPB_MASTER)[[
            "anchor_id", "sharedness_class", "cancer_coverage", "noncancer_coverage",
            "label_auc", "energy_percentile",
        ]],
        on="anchor_id", validate="one_to_one",
    ).sort_values("anchor_id").reset_index(drop=True)
    if len(anchors) != 149:
        raise RuntimeError("pilot必须使用冻结149个RP-D Anchor")

    target_features = np.load(RPC2_EFFECT_ROOT / "target_feature_ids.npy")
    if not np.array_equal(target_features, anchors.feature_id.to_numpy(np.int64)):
        raise RuntimeError("RP-C2 target Feature顺序与RP-D Anchor不一致")
    actual_delta = np.load(
        RPC2_EFFECT_ROOT / "image_delta_margin_alpha_0p00.npy", mmap_mode="r"
    )
    if actual_delta.shape != (2350, 149):
        raise RuntimeError(f"RP-C2逐图效应shape异常: {actual_delta.shape}")

    state = load_checkpoint_state()
    classifier_weight = state["classifier.1.weight"].detach().cpu().numpy()
    margin_weight = classifier_weight[1] - classifier_weight[0]
    decoder = np.load(RPA_CACHE / "decoder_weight.npy", mmap_mode="r")
    if decoder.shape != (10240, 1280) or margin_weight.shape != (1280,):
        raise RuntimeError("decoder或C-long分类margin方向shape异常")
    decoder_direction = np.asarray(decoder[anchors.feature_id.to_numpy(np.int64)]) @ margin_weight

    return {
        "metadata": metadata,
        "anchors": anchors,
        "activation": np.load(RPA_CACHE / "train_image_activations.npy", mmap_mode="r"),
        "attention": np.load(CLONG_ATTENTION, mmap_mode="r"),
        "actual_delta": actual_delta,
        "decoder_direction": decoder_direction,
    }


def select_features(
    raw_score: np.ndarray, actual_delta: np.ndarray, anchors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按冻结语义规则和正式干预效应分别选择Top-6。"""
    ids = anchors.anchor_id.astype(str).to_numpy()
    raw_top = np.lexsort((ids, -raw_score))[:TOP_N]
    intervention_top = np.lexsort((ids, -np.abs(actual_delta)))[:TOP_N]
    union = np.asarray(list(dict.fromkeys([*raw_top.tolist(), *intervention_top.tolist()])))
    return raw_top, intervention_top, union


def render_case(data: dict, image_index: int, output: Path) -> list[dict]:
    """渲染一个病例的Raw与Intervention Top-6联合对照。"""
    metadata = data["metadata"].iloc[image_index]
    anchors = data["anchors"]
    feature_ids = anchors.feature_id.to_numpy(np.int64)
    raw = np.asarray(data["activation"][image_index][:, feature_ids], dtype=np.float64).T
    raw = raw.reshape(149, 7, 7)
    attention = np.asarray(data["attention"][image_index], dtype=np.float64).reshape(7, 7)
    actual_delta = np.asarray(data["actual_delta"][image_index], dtype=np.float64)
    raw_score = raw.max(axis=(1, 2)) / anchors.q99_scale.to_numpy(np.float64)
    raw_top, intervention_top, union = select_features(raw_score, actual_delta, anchors)
    raw_rank = {int(index): rank + 1 for rank, index in enumerate(raw_top)}
    intervention_rank = {
        int(index): rank + 1 for rank, index in enumerate(intervention_top)
    }

    bbox = None
    if int(metadata.label) == 1 and bool(metadata.bbox_valid):
        bbox = np.asarray([
            metadata.bbox_x1_norm, metadata.bbox_y1_norm,
            metadata.bbox_x2_norm, metadata.bbox_y2_norm,
        ], dtype=np.float64)
    image_path = PROJECT_ROOT / str(metadata.image_relpath)
    with Image.open(image_path) as source:
        original = source.convert("RGB")

    tile = (360, 270)
    gap = 12
    title_h = 74
    header_h = 410
    row_h = title_h + tile[1]
    canvas_w = gap * 4 + tile[0] * 3
    canvas_h = header_h + gap + len(union) * (row_h + gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype(str(FONT_PATH), 22)
    font_head = ImageFont.truetype(str(FONT_PATH), 18)
    font_small = ImageFont.truetype(str(FONT_PATH), 12)
    font_tiny = ImageFont.truetype(str(FONT_PATH), 10)

    draw.text((gap, 10), f"Decision-aware SAE pilot | image_index={image_index}",
              fill=(25, 29, 32), font=font_title)
    draw.text(
        (gap, 44),
        f"label={int(metadata.label)} | source={metadata.source} | "
        f"C-long癌概率={float(metadata.cancer_probability):.3f} | train-only diagnostic",
        fill=(65, 70, 73), font=font_small,
    )
    draw.text(
        (gap, 66),
        "左列 h：编码位置；中列 a×h：编码且被原模型关注；右列 -a×h×(dᵀw)：固定注意力下删除该Feature的局部Δmargin",
        fill=(65, 70, 73), font=font_small,
    )
    original_boxed = pad_image(draw_bbox(original, bbox), tile)
    attention_display = attention / float(attention.max())
    attention_overlay = pad_image(
        render_overlay(
            original, render_heatmap(attention_display, original.size), attention_display, 0.62,
        ), tile,
    )
    canvas.paste(original_boxed, (gap, 110))
    canvas.paste(attention_overlay, (gap * 2 + tile[0], 110))
    draw.text((gap, 88), "原图（癌图绿色为冻结病灶框）", fill=(25, 29, 32), font=font_head)
    draw.text((gap * 2 + tile[0], 88), "C-long原始注意力", fill=(25, 29, 32), font=font_head)

    summary_x = gap * 3 + tile[0] * 2
    draw.text((summary_x, 88), "两套Top-6", fill=(25, 29, 32), font=font_head)
    raw_names = [str(anchors.iloc[index].anchor_id) for index in raw_top]
    intervention_names = [str(anchors.iloc[index].anchor_id) for index in intervention_top]
    summary_lines = [
        "视觉语义 Top-6（peak/train Q99）:",
        ", ".join(raw_names[:3]), ", ".join(raw_names[3:]), "",
        "功能 Top-6（正式RP-C2 |Δmargin|）:",
        ", ".join(intervention_names[:3]), ", ".join(intervention_names[3:]), "",
        f"交集: {len(set(raw_top) & set(intervention_top))}/6",
        "红：删除后癌margin上升；蓝：删除后癌margin下降",
    ]
    for line_no, line in enumerate(summary_lines):
        draw.text((summary_x, 120 + line_no * 25), line, fill=(50, 56, 59), font=font_small)

    records: list[dict] = []
    for row_number, anchor_index in enumerate(union):
        anchor = anchors.iloc[anchor_index]
        h = raw[anchor_index]
        gated = attention * h
        direct = fixed_attention_local_ablation_map(
            h, attention, data["decoder_direction"][anchor_index]
        )
        overlap, cosine, peak_distance = attention_alignment(h, attention)
        box_stats = bbox_metrics(h, gated, direct, bbox)
        direct_sum = float(direct.sum())
        y = header_h + gap + row_number * (row_h + gap)
        raw_text = f"Raw h | 视觉rank={raw_rank.get(int(anchor_index), '-')}"
        gated_text = f"a×h | overlap={overlap:.3f} cosine={cosine:.3f} 峰距={peak_distance:.1f}"
        direct_text = (
            f"固定attention局部删除Δm | Σ={direct_sum:+.4f} | "
            f"实际重算={actual_delta[anchor_index]:+.4f}"
        )
        for column, text in enumerate((raw_text, gated_text, direct_text)):
            draw.text(
                (gap + column * (tile[0] + gap), y), text,
                fill=(25, 29, 32), font=font_small,
            )
        subtitle = (
            f"{anchor.anchor_id}/F{int(anchor.feature_id)} | 功能rank="
            f"{intervention_rank.get(int(anchor_index), '-')} | source-risk={bool(anchor.source_risk)}"
        )
        draw.text((gap, y + 22), subtitle, fill=(80, 85, 88), font=font_tiny)
        if bbox is not None:
            draw.text(
                (gap, y + 42),
                f"框内质量 raw={box_stats['raw_bbox_mass']:.3f} gated={box_stats['gated_bbox_mass']:.3f} | "
                f"direct框内signed={box_stats['direct_bbox_signed_sum']:+.4f} abs占比="
                f"{box_stats['direct_bbox_absolute_mass_fraction']:.3f}",
                fill=(80, 85, 88), font=font_tiny,
            )
        raw_display = display_response(h, float(anchor.q99_scale))
        gated_display = normalize_positive(gated)
        images = [
            render_overlay(original, render_heatmap(raw_display, original.size), raw_display, 0.62),
            render_overlay(original, render_heatmap(gated_display, original.size), gated_display, 0.62),
            signed_overlay(original, direct),
        ]
        for column, panel in enumerate(images):
            canvas.paste(
                pad_image(panel, tile),
                (gap + column * (tile[0] + gap), y + title_h),
            )
        record = {
            "image_index": image_index,
            "patient_id": str(metadata.patient_id),
            "label": int(metadata.label),
            "source": str(metadata.source),
            "anchor_id": str(anchor.anchor_id),
            "feature_id": int(anchor.feature_id),
            "raw_top6_rank": raw_rank.get(int(anchor_index)),
            "intervention_top6_rank": intervention_rank.get(int(anchor_index)),
            "raw_relative_q99_score": float(raw_score[anchor_index]),
            "actual_recomputed_attention_delta_margin": float(actual_delta[anchor_index]),
            "actual_recomputed_attention_abs_delta_margin": float(abs(actual_delta[anchor_index])),
            "fixed_attention_delta_margin_sum": direct_sum,
            "decoder_margin_direction": float(data["decoder_direction"][anchor_index]),
            "attention_overlap": overlap,
            "attention_cosine": cosine,
            "attention_peak_distance_grid": peak_distance,
            "raw_grid_edge_mass_ratio": grid_edge_mass_ratio(h),
            "gated_grid_edge_mass_ratio": grid_edge_mass_ratio(gated),
            "source_risk": bool(anchor.source_risk),
            "sharedness_class": str(anchor.sharedness_class),
            **box_stats,
        }
        records.append(record)

    output_file = output / f"image_{image_index:04d}_raw_vs_intervention_top6.png"
    canvas.save(output_file, optimize=True)
    return records


def write_readme(path: Path) -> None:
    """说明pilot问题、图层和证据边界。"""
    path.write_text("""# Decision-aware SAE pilot

本pilot只比较两个冻结train病例，不重新训练模型，不改变149 Anchor的RP-D正式交付包，也不产生新的
PASS/FAIL。它回答两个不同问题：raw SAE激活说明模型内部编码了哪些视觉模式；RP-C2真实干预效应
说明当前病例预测对哪些Feature更敏感。

## 两套Top-6

- 视觉语义Top-6：沿用RP-D规则，按本图空间峰值除以该Anchor完整train正激活Q99排序；
- 功能Top-6：按正式RP-C2保存的单图 `abs(delta_margin at alpha=0)` 排序。这里
  `delta_margin = ablated - original`，并使用重算attention的正式路径。

## 三张Feature空间图

1. `h`：这个视觉模式在哪里被编码；显示缩放沿用冻结train Q99；
2. `a×h`：这个模式与C-long原始注意力同时出现在哪里；仅在单Feature内部按峰值归一化显示；
3. `-a×h×(d^T w_margin)`：删除该Feature时，固定原attention的逐位置margin变化。红色表示删除后
   癌margin上升，蓝色表示删除后癌margin下降。该图不包含attention重排，真实最终效应以正式RP-C2
   的重算attention标量为准。

`attention overlap/cosine/peak distance`比较raw Feature空间分布与C-long原始attention。癌图另外按
冻结7×7 cell-overlap几何口径报告raw/gated框内质量；有符号direct图分别报告框内signed sum和框内
absolute-mass fraction，避免正负贡献相互抵消。`7×7 grid-edge mass ratio`只描述最外圈网格响应，
不能称为黑边分数，也不能据此认定artifact。

## 解释边界

- raw activation ranking不等于functional ranking；
- attention-gated图是描述性乘积，不是独立定位改进；
- SAE字典过完备且非正交，单Feature图不是唯一的加性因果分解；
- 所有内容均为train-only技术诊断，未读取val/internal test/external；
- Feature的医学命名和临床意义仍由医学生或医生完成。
""", encoding="utf-8")


def main() -> None:
    """生成两个病例、明细CSV和诊断配置。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    data = load_inputs()
    records: list[dict] = []
    for image_index in IMAGE_INDICES:
        records.extend(render_case(data, image_index, OUTPUT_ROOT))
        print(f"rendered image_index={image_index}", flush=True)
    detail = pd.DataFrame(records)
    detail.to_csv(OUTPUT_ROOT / "pilot_feature_metrics.csv", index=False, encoding="utf-8-sig")
    write_readme(OUTPUT_ROOT / "README_结果怎么看.md")
    config = {
        "status": "decision_aware_pilot_complete",
        "image_indices": list(IMAGE_INDICES),
        "canonical_seed": 42,
        "display_anchor_count": 149,
        "raw_top_n": TOP_N,
        "intervention_top_n": TOP_N,
        "raw_ranking": "image_peak_activation / frozen_anchor_train_positive_q99",
        "intervention_ranking": "abs(formal_rpc2_image_delta_margin_alpha_0p00)",
        "delta_margin_sign": "ablated_minus_original",
        "direct_map": "-original_attention * activation * dot(decoder, cancer_margin_weight)",
        "attention_mode_for_actual_delta": "recomputed",
        "diagnostic_only": True,
        "scientific_pass_fail": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
