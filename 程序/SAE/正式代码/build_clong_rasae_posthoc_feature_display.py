#!/usr/bin/env python3
"""用已有审核病例制作C-long诊断＋事后Feature解释的三张内部展示图。"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import torch

from build_clong_rasae_first_medical_batch import project_selected_maps


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
SELECTED_PATH = BASE / "functional_first_medical_batch_20260915/selected_images_internal.csv"
EFFECT_PATH = BASE / "reviewed_response_function_20260915/reviewed_pair_effects.csv"
Q99_PATH = BASE / "decision_alignment/ra_train_q99.npy"
HEAT_ALPHA = 0.25
CONTOUR_THRESHOLD = 0.5

CASES = (
    {
        "display": "示例A", "feature_id": 2444,
        "summary": "仅限这两个例子：候选模式图的删除效应更大。",
        "rows": (
            ("val_0359", "候选描述", "医学生观察：病灶处，可能与凹陷有关。"),
            ("train_0983", "存在反例", "医学生观察：高响应位于透明帽处。"),
        ),
    },
    {
        "display": "示例B", "feature_id": 868,
        "summary": "反例同样有明显margin影响，当前Feature保留“语义混杂”判断。",
        "rows": (
            ("val_0310", "候选描述", "医学生观察：病灶处，可能与凹陷或糜烂有关。"),
            ("train_1222", "存在反例", "医学生观察：高响应主要位于正常黏膜处。"),
        ),
    },
    {
        "display": "示例C", "feature_id": 1089,
        "summary": "两种观察的margin影响为相近量级，暂无法解释为单一视觉含义。",
        "rows": (
            ("train_1601", "候选描述", "医学生观察：幽门附近，高响应主要位于幽门口。"),
            ("train_0737", "暂无法解释", "医学生观察：隆起样区域，无幽门。"),
        ),
    },
)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def wrap(value: str, width: int = 32) -> str:
    """按字符宽度换行，供图中中文说明使用。"""
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=True))


def load_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """读取并核对6个既有选例、医学反馈、删除效应和原始C-long输出。"""
    selected = pd.read_csv(SELECTED_PATH)
    effects = pd.read_csv(EFFECT_PATH)
    frames = {
        split: pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(
            columns={"index": "source_row"}
        )
        for split in ("train", "val")
    }
    requested = []
    for display in CASES:
        feature = int(display["feature_id"])
        for image_code, status, observation in display["rows"]:
            chosen = selected[
                selected.feature_id.eq(feature) & selected.image_code.eq(image_code)
            ]
            effect = effects[
                effects.feature_id.eq(feature) & effects.image_code.eq(image_code)
            ]
            if len(chosen) != 1 or len(effect) != 1:
                raise RuntimeError(f"RA-F{feature:04d}/{image_code}未唯一对应已有记录")
            row = chosen.iloc[0].to_dict()
            effect_row = effect.iloc[0]
            metadata = frames[str(row["split"])].iloc[int(row["source_row"])]
            if str(metadata.patient_id) != str(row["patient_id"]):
                raise RuntimeError(f"{image_code}元数据患者不一致")
            row.update({
                "display": display["display"], "status": status,
                "observation": observation, "summary": display["summary"],
                "data_label": int(metadata.label),
                "original_probability": float(metadata.cancer_probability),
                "delta_margin": float(effect_row.delta_margin),
                "delta_probability": float(effect_row.delta_probability),
            })
            row["ablated_probability"] = row["original_probability"] + row["delta_probability"]
            requested.append(row)
    result = pd.DataFrame(requested)
    if len(result) != 6 or result[["feature_id", "image_code"]].duplicated().any():
        raise RuntimeError("展示病例应恰好为6个唯一Feature-图片对")
    return result, frames


def render_case(
    display: dict, selected: pd.DataFrame, maps: dict[tuple[str, int], np.ndarray],
    q99: np.ndarray, output: Path,
) -> None:
    """生成一个Feature的两例内部展示图。"""
    feature = int(display["feature_id"])
    rows = selected[selected.feature_id.eq(feature)].set_index("image_code")
    figure = plt.figure(figsize=(16, 11.5), facecolor="white")
    grid = figure.add_gridspec(
        3, 3, height_ratios=(0.15, 1, 1), width_ratios=(1, 1, 1.12),
        hspace=0.30, wspace=0.08,
    )
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0, 0.88, f"{display['display']}｜RA-F{feature:04d}｜原C-long诊断＋事后Feature解释",
        fontsize=20, weight="bold", va="top",
    )
    title_axis.text(
        0, 0.28,
        "左：原图　中：同一原图上的Feature响应　右：数据标签、模型输出、内部删除影响与医学观察",
        fontsize=12, color="#333333", va="top",
    )

    for row_index, (image_code, _, _) in enumerate(display["rows"], start=1):
        record = rows.loc[image_code]
        image = Image.open(ROOT / record.image_relpath).convert("RGB")
        image.thumbnail((900, 900))
        rgb = np.asarray(image)
        normalized = (
            maps[(str(record.split), int(record.source_row))][:, feature].reshape(7, 7)
            / q99[feature]
        )
        dense = torch.nn.functional.interpolate(
            torch.from_numpy(normalized).reshape(1, 1, 7, 7),
            size=rgb.shape[:2], mode="bilinear", align_corners=False,
        )[0, 0].numpy()

        original_axis = figure.add_subplot(grid[row_index, 0])
        original_axis.imshow(rgb)
        original_axis.set_title(f"{image_code}｜原图", fontsize=12)
        original_axis.axis("off")

        response_axis = figure.add_subplot(grid[row_index, 1])
        response_axis.imshow(rgb)
        response_axis.imshow(
            dense, cmap="magma", vmin=0, vmax=1, alpha=HEAT_ALPHA, interpolation="nearest"
        )
        if float(dense.max()) >= CONTOUR_THRESHOLD > float(dense.min()):
            response_axis.contour(
                dense, levels=[CONTOUR_THRESHOLD], colors=["#00ff66"], linewidths=2.2
            )
        elif float(dense.min()) >= CONTOUR_THRESHOLD:
            response_axis.add_patch(Rectangle(
                (-0.5, -0.5), rgb.shape[1], rgb.shape[0], fill=False,
                edgecolor="#00ff66", linewidth=2.2,
            ))
        response_axis.set_title("Feature响应（固定尺度）", fontsize=12)
        response_axis.axis("off")

        text_axis = figure.add_subplot(grid[row_index, 2])
        text_axis.axis("off")
        label = "癌" if int(record.data_label) == 1 else "非癌"
        lines = [
            f"状态：{record.status}",
            f"数据标签：{label}",
            f"原C-long输出：癌概率 {record.original_probability:.4f}",
            "内部干预后的模型输出：",
            f"癌概率 {record.ablated_probability:.4f}",
            f"Δ概率（干预后−原始）：{record.delta_probability:+.4f}",
            f"Δmargin（干预后−原始）：{record.delta_margin:+.4f}",
            "",
            wrap(str(record.observation), 27),
        ]
        text_axis.text(
            0.03, 0.97, "\n".join(lines), fontsize=12.5, va="top", linespacing=1.45,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#f6f7f9", "edgecolor": "#c9cdd4"},
        )

    figure.text(
        0.05, 0.055, wrap(str(display["summary"]), 72),
        fontsize=12.5, weight="bold", color="#333333", va="bottom",
    )
    footer = (
        "参数速读：癌概率是模型对“癌”类别的输出，不等同于真实患癌概率；数据标签来自数据集。"
        "Δ概率和Δmargin均为“内部干预后−原始”，margin=癌logit−非癌logit。"
        "内部干预指删除该Feature在整张图49个位置的分量后重新计算注意力和分类输出，"
        "不表示去除了真实组织。绿色线为≥0.5×训练集正激活Q99的较强激活区域，不是病灶边界。"
    )
    figure.text(0.05, 0.012, wrap(footer, 105), fontsize=9.5, color="#555555", va="bottom")
    figure.savefig(
        output / f"{display['display']}_RA-F{feature:04d}.png",
        dpi=150, bbox_inches="tight", facecolor="white",
    )
    plt.close(figure)


def write_readme(output: Path) -> None:
    """写入不含表格的展示阅读说明。"""
    text = """这3张图是“原C-long诊断＋事后Feature解释”的内部展示小样，只用于判断这种输出是否帮助理解模型，不验证临床解释有效性。

每张图怎么看

左边是原始胃镜图；中间是在同一原图上叠加的单项SAE Feature响应；右边依次写明数据标签、原模型输出、内部干预后的模型输出、数值变化和医学生已有观察。数据标签来自数据集，模型预测则是原C-long给出的癌概率，两者不是同一个概念。

原C-long输出／癌概率

这是原分类器对“癌”类别给出的0到1之间的模型输出。数值越大，模型越偏向癌类别；它不是临床校准后的个人患癌风险。SAE不替代分类器，展示前后的诊断主体始终是原C-long及其冻结分类头。

内部干预后的模型输出

技术侧将指定Feature在整张图全部49个空间位置上的分量同时删除，再重新计算原注意力、特征汇聚和分类输出。这个数值只是内部表示干预结果，不代表从患者图像中去除了某个真实组织，也不能说明变化一定来自绿色线圈出的局部区域。

Δ概率

定义为“内部干预后的癌概率−原C-long癌概率”。负值表示删除后模型癌概率下降，正值表示删除后模型癌概率上升。原始概率不同的病例不能只按概率变化大小比较Feature影响。

Δmargin

margin定义为“癌类别logit−非癌类别logit”；logit是模型在转成概率前的原始分类分数。Δmargin定义为“内部干预后margin−原始margin”。负值表示删除后模型更偏向非癌方向，正值表示更偏向癌方向。比较不同原始概率病例的功能影响时，本展示优先看|Δmargin|，但它仍是内部干预敏感性，不是医学因果贡献。

RA-F编号

RA-F2444等编号只是当前100轮RA-SAE字典中Feature的技术编号，不是医学Concept名称。当前模型字典共有2560项Feature；编码器K=256表示每个7×7空间位置最多保留256项激活，不是本页展示了256项，也不是256个医学概念。

热图、Q99、7×7和49个位置

SAE在7×7空间网格上产生响应，共49个位置，再用双线性插值放大到原图尺寸。热图表示该Feature的响应强弱，不是原C-long注意力，也不是病灶概率图。每项Feature使用训练集正激活值的第99百分位数Q99作固定尺度：显示值1相当于该Feature的Q99，超过1的区域使用同一最高颜色。这样同一Feature可以跨图比较；不同Feature各有自己的Q99，不应直接比较颜色绝对强弱。

热图透明度和绿色线

彩色热图透明度固定为0.25，目的是尽量保留黏膜原有颜色。绿色线标出插值后响应达到0.5×Q99的区域，表示“较强激活区域”，不是病灶边界；没有绿色线只表示没有达到这个固定显示阈值，不代表图中没有相关结构或Feature完全不响应。

三张图的文字状态

“候选描述”表示医学生在部分已看图片中提出的可能视觉含义，尚未确认为Concept。“存在反例”表示该图的观察与候选模式不一致。“暂无法解释”表示现有病例不足以把响应稳定归为一个视觉含义。三种文字都保留不确定性，不自动生成医学概念。

示例A的比较只限所选两例：RA-F2444候选模式图的|Δmargin|大于透明帽反例。示例B中，RA-F0868正常黏膜反例仍有明显margin影响。示例C中，RA-F1089在幽门附近和无幽门的隆起样区域都出现相近量级的margin影响，暂不能归为单一视觉含义。以上均不能推广到所有病例。
"""
    (output / "阅读说明.txt").write_text(text, encoding="utf-8")


def main() -> None:
    """核对既有记录、投影6张固定病例并制作3张展示图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected, frames = load_cases()
    maps = project_selected_maps(selected, frames, torch.device(args.device), args.batch_size)
    q99 = np.load(Q99_PATH)
    for display in CASES:
        render_case(display, selected, maps, q99, args.output)
    write_readme(args.output)
    print({
        "images": len(CASES), "cases": len(selected),
        "files": sorted(path.name for path in args.output.glob("*")),
        "new_experiment": False,
    }, flush=True)


if __name__ == "__main__":
    main()
