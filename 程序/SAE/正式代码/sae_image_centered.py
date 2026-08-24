"""以单张图像为中心展示Top-K SAE feature及其空间响应。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from PIL import Image

import torch

from sae_discovery import (
    ACTIVE_EPS,
    EVAL_TRANSFORM,
    ActivationCapture,
    load_resnet50,
)
from sae_project_analyze import load_locked_sae


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "结果" / "SAE图像中心可视化"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLORS = np.asarray(
    [
        [230, 25, 75],
        [0, 170, 220],
        [255, 190, 0],
        [145, 30, 180],
        [0, 170, 90],
        [240, 90, 20],
        [70, 90, 220],
        [220, 70, 150],
    ],
    dtype=np.float32,
) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(description="生成单图Top-K SAE多颜色热图")
    parser.add_argument("--sae-run", required=True, help="锁定的正式SAE目录")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", action="append", help="图片路径，可重复传入")
    source.add_argument("--manifest", help="批量图片清单CSV")
    parser.add_argument("--patient-id", help="清单模式下只处理指定患者")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5, help="清单模式默认仅处理5张")
    parser.add_argument("--all", action="store_true", help="清单模式处理筛选后的全部图片")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rank-by",
        choices=["activation", "cancer_contribution", "absolute_contribution"],
        default="activation",
        help="Top-K排序口径",
    )
    parser.add_argument(
        "--color-mode",
        choices=["contribution", "feature"],
        default="contribution",
        help="contribution为正红负蓝，feature为不同feature使用不同颜色",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="热图在热点处的最大叠加权重，默认0.55（Grad-CAM脚本为0.40）",
    )
    parser.add_argument(
        "--response-gamma",
        type=float,
        default=0.75,
        help="空间响应显色gamma；小于1可增强中等响应，默认0.75",
    )
    parser.add_argument("--names", help="可选的feature命名CSV或Excel")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--experiment", help="输出目录名；默认使用时间戳")
    return parser.parse_args()


def load_input_rows(args):
    if args.image:
        return pd.DataFrame(
            {
                "image_path": [str(Path(path).resolve()) for path in args.image],
                "patient_id": [Path(path).parent.name for path in args.image],
                "label": [np.nan] * len(args.image),
            }
        )

    dataframe = pd.read_csv(args.manifest, encoding="utf-8-sig")
    path_column = next(
        (
            column
            for column in ["processed_path", "image_relpath", "image_path", "图片名字"]
            if column in dataframe
        ),
        None,
    )
    if path_column is None:
        raise ValueError("manifest缺少图片路径字段")
    dataframe = dataframe.copy()
    dataframe["image_path"] = dataframe[path_column].astype(str)
    if args.patient_id is not None:
        dataframe = dataframe[
            dataframe["patient_id"].astype(str) == str(args.patient_id)
        ]
    dataframe = dataframe.iloc[args.start_index :]
    if not args.all:
        dataframe = dataframe.head(args.limit)
    if dataframe.empty:
        raise ValueError("没有符合条件的图片")
    if "patient_id" not in dataframe:
        dataframe["patient_id"] = dataframe["image_path"].map(
            lambda value: Path(value).parent.name
        )
    if "label" not in dataframe:
        dataframe["label"] = np.nan
    return dataframe.reset_index(drop=True)


def load_names(path):
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        dataframe = pd.read_csv(path, encoding="utf-8-sig")
    if "feature_id" not in dataframe:
        raise ValueError("feature命名表缺少feature_id")
    name_column = next(
        (
            column
            for column in ["暂定医学名称", "concept_name", "概念名称"]
            if column in dataframe
        ),
        None,
    )
    if name_column is None:
        return {}
    names = {}
    for _, row in dataframe.iterrows():
        value = row[name_column]
        if pd.notna(value) and str(value).strip():
            names[int(row["feature_id"])] = str(value).strip()
    return names


def normalize_map(response):
    response = np.maximum(response, 0)
    maximum = float(response.max())
    return response / maximum if maximum > 1e-12 else np.zeros_like(response)


def resize_map(response, size):
    image = Image.fromarray(np.uint8(np.clip(response, 0, 1) * 255))
    image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def enhance_response(response, gamma):
    """增强中等空间响应，同时保持0和1两个端点不变。"""
    return np.power(np.clip(response, 0, 1), gamma)


def color_overlay(original_array, response, color, alpha, response_gamma):
    color_layer = np.zeros_like(original_array, dtype=np.float32)
    color_layer[:] = color
    weight = alpha * enhance_response(response, response_gamma)[..., None]
    return np.clip(original_array * (1 - weight) + color_layer * weight, 0, 1)


def make_composite(
    original_array,
    maps,
    colors,
    strengths,
    overlay_alpha,
    response_gamma,
):
    weighted = np.stack(
        [
            enhance_response(response, response_gamma) * strength
            for response, strength in zip(maps, strengths)
        ],
        axis=0,
    )
    total = weighted.sum(axis=0)
    color_sum = np.zeros_like(original_array, dtype=np.float32)
    for response, color in zip(weighted, colors):
        color_sum += response[..., None] * color
    mixed = color_sum / np.maximum(total[..., None], 1e-12)
    alpha = overlay_alpha * np.clip(weighted.max(axis=0), 0, 1)[..., None]
    return np.clip(original_array * (1 - alpha) + mixed * alpha, 0, 1)


def selected_feature_table(hidden, kept_mask, cancer_direction, top_k, rank_by):
    activation = hidden.detach().cpu().numpy()
    contribution = activation * cancer_direction
    candidates = np.flatnonzero(kept_mask & (activation > ACTIVE_EPS))
    if len(candidates) == 0:
        raise ValueError("该图片没有激活任何筛选后保留的SAE feature")
    if rank_by == "activation":
        score = activation
    elif rank_by == "cancer_contribution":
        score = contribution
        candidates = candidates[contribution[candidates] > 0]
        if len(candidates) == 0:
            raise ValueError("该图片没有正向推动癌判断的保留feature")
    else:
        score = np.abs(contribution)
    order = candidates[np.argsort(score[candidates])[::-1]][:top_k]
    return order, activation, contribution, score


def save_overview(
    output_path,
    original,
    composite,
    overlays,
    rows,
    original_probability,
    reconstructed_probability,
    color_mode,
    max_abs_contribution,
):
    font = FontProperties(fname=FONT_PATH)
    count = len(overlays)
    columns = 2
    feature_rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        1 + feature_rows,
        columns,
        figsize=(14, 6 + 5.4 * feature_rows),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("原图", fontproperties=font, fontsize=15)
    axes[0, 1].imshow(composite)
    axes[0, 1].set_title("Top SAE feature 多颜色叠加图", fontproperties=font, fontsize=15)
    axes[0, 1].legend(
        handles=[
            Patch(
                color=np.asarray(row["color_rgb"].split(","), dtype=float) / 255,
                label=f'Feature {row["feature_id"]}',
            )
            for row in rows
        ],
        loc="lower right",
        framealpha=0.85,
        prop=font,
    )
    if color_mode == "contribution":
        colorbar = figure.colorbar(
            ScalarMappable(
                norm=Normalize(
                    vmin=-max_abs_contribution,
                    vmax=max_abs_contribution,
                ),
                cmap="bwr",
            ),
            ax=axes[0, 1],
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("癌 logit margin 贡献：蓝=负向，红=正向", fontproperties=font)
    for index, (overlay, row) in enumerate(zip(overlays, rows)):
        axis = axes[1 + index // columns, index % columns]
        axis.imshow(overlay)
        sign = "+" if row["cancer_margin_contribution"] >= 0 else ""
        title = (
            f'Feature {row["feature_id"]}  激活={row["activation"]:.3f}  '
            f'癌贡献={sign}{row["cancer_margin_contribution"]:.3f}'
        )
        if row["concept_name"]:
            title += f'  {row["concept_name"]}'
        axis.set_title(title, fontproperties=font, fontsize=12)
    figure.suptitle(
        f"原模型癌概率={original_probability:.3f}    "
        f"筛选后SAE重构癌概率={reconstructed_probability:.3f}",
        fontproperties=font,
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def save_composite(
    output_path,
    composite,
    rows,
    color_mode,
    max_abs_contribution,
):
    """单独保存带feature颜色图例的多颜色叠加图。"""
    font = FontProperties(fname=FONT_PATH)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(composite)
    axis.axis("off")
    axis.set_title("Top SAE feature 多颜色叠加图", fontproperties=font, fontsize=16)
    axis.legend(
        handles=[
            Patch(
                color=np.asarray(row["color_rgb"].split(","), dtype=float) / 255,
                label=(
                    f'Feature {row["feature_id"]}  '
                    f'激活={row["activation"]:.3f}  '
                    f'癌贡献={row["cancer_margin_contribution"]:+.3f}'
                ),
            )
            for row in rows
        ],
        loc="lower right",
        framealpha=0.88,
        prop=font,
    )
    if color_mode == "contribution":
        colorbar = figure.colorbar(
            ScalarMappable(
                norm=Normalize(
                    vmin=-max_abs_contribution,
                    vmax=max_abs_contribution,
                ),
                cmap="bwr",
            ),
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("癌 logit margin 贡献：蓝=负向，红=正向", fontproperties=font)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


@torch.no_grad()
def process_image(
    image_row,
    index,
    output_dir,
    model,
    capture,
    sae,
    kept_mask,
    cancer_direction,
    names,
    args,
    device,
):
    image_path = Path(image_row.image_path)
    if not image_path.is_absolute():
        image_path = Path(args.manifest).resolve().parent / image_path
    original = Image.open(image_path).convert("RGB")
    image_tensor = EVAL_TRANSFORM(original).unsqueeze(0).to(device)
    logits = model(image_tensor)
    feature_map = capture.output[0]
    gap_feature = feature_map.mean(dim=(1, 2), keepdim=False).unsqueeze(0)
    _, hidden_batch = sae(gap_feature)
    hidden = hidden_batch[0]
    pruned_hidden = hidden * torch.as_tensor(kept_mask, device=device)
    pruned_reconstruction = sae.decode(pruned_hidden.unsqueeze(0))
    original_probability = torch.softmax(logits, dim=1)[0, 1].item()
    reconstructed_logits = model.fc(pruned_reconstruction)
    reconstructed_probability = torch.softmax(reconstructed_logits, dim=1)[0, 1].item()

    selected, activation, contribution, score = selected_feature_table(
        hidden, kept_mask, cancer_direction, args.top_k, args.rank_by
    )
    max_abs_contribution = max(
        float(np.abs(contribution[selected]).max()),
        1e-12,
    )
    if args.color_mode == "contribution":
        selected_colors = [
            np.asarray(
                [1.0, 0.0, 0.0]
                if contribution[current] >= 0
                else [0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for current in selected
        ]
    else:
        selected_colors = [
            COLORS[index % len(COLORS)] for index in range(len(selected))
        ]
    original_array = np.asarray(original, dtype=np.float32) / 255.0
    maps = []
    colors = []
    overlays = []
    rows = []
    image_dir = output_dir / f"{index + 1:04d}_{image_path.stem}"
    image_dir.mkdir(parents=True)
    maximum_activation = max(float(activation[selected].max()), 1e-12)

    for rank, current in enumerate(selected, 1):
        direction = sae.encoder.weight[current]
        raw_response = torch.relu(
            (feature_map * direction[:, None, None]).sum(dim=0)
        ).cpu().numpy()
        raw_maximum = float(raw_response.max())
        response = resize_map(normalize_map(raw_response), original.size)
        color = selected_colors[rank - 1]
        overlay = color_overlay(
            original_array,
            response,
            color,
            args.overlay_alpha,
            args.response_gamma,
        )
        maps.append(response)
        colors.append(color)
        overlays.append(overlay)
        concept_name = names.get(int(current), "")
        row = {
            "rank": rank,
            "feature_id": int(current),
            "concept_name": concept_name,
            "activation": float(activation[current]),
            "cancer_margin_direction": float(cancer_direction[current]),
            "cancer_margin_contribution": float(contribution[current]),
            "ranking_score": float(score[current]),
            "color_rgb": ",".join(str(int(value * 255)) for value in color),
            "color_mode": args.color_mode,
            "spatial_response_max_before_normalization": raw_maximum,
        }
        rows.append(row)
        Image.fromarray(np.uint8(overlay * 255)).save(
            image_dir / f"rank_{rank:02d}_feature_{int(current):04d}.png"
        )

    if args.color_mode == "contribution":
        strengths = [
            float(abs(contribution[current]) / max_abs_contribution)
            for current in selected
        ]
    else:
        strengths = [
            float(activation[current] / maximum_activation) for current in selected
        ]
    composite = make_composite(
        original_array,
        maps,
        colors,
        strengths,
        args.overlay_alpha,
        args.response_gamma,
    )
    save_composite(
        image_dir / "Top_feature多颜色叠加图.png",
        composite,
        rows,
        args.color_mode,
        max_abs_contribution,
    )
    pd.DataFrame(rows).to_csv(
        image_dir / "Top_feature统计.csv", index=False, encoding="utf-8-sig"
    )
    save_overview(
        image_dir / "图像中心_SAE_Top_feature总览.png",
        original,
        composite,
        overlays,
        rows,
        original_probability,
        reconstructed_probability,
        args.color_mode,
        max_abs_contribution,
    )
    label = image_row.label if pd.notna(image_row.label) else ""
    return {
        "order": index + 1,
        "image_path": str(image_path),
        "patient_id": str(image_row.patient_id),
        "label": label,
        "original_cancer_probability": original_probability,
        "sae_pruned_cancer_probability": reconstructed_probability,
        "active_kept_feature_count": int(
            np.sum(kept_mask & (activation > ACTIVE_EPS))
        ),
        "top_feature_ids": ";".join(str(int(value)) for value in selected),
        "output_dir": str(image_dir),
    }


def main():
    args = parse_args()
    if args.top_k < 1 or args.top_k > len(COLORS):
        raise ValueError(f"--top-k必须在1到{len(COLORS)}之间")
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha必须在0到1之间")
    if args.response_gamma <= 0:
        raise ValueError("--response-gamma必须大于0")
    sae_run = Path(args.sae_run).resolve()
    with open(sae_run / "config.json", encoding="utf-8") as file:
        config = json.load(file)
    output_root = Path(args.output_root).resolve()
    experiment = args.experiment or datetime.now().strftime(
        "image_centered_%Y%m%d_%H%M%S"
    )
    output_dir = output_root / experiment
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = load_input_rows(args)
    names = load_names(args.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet50(device, config["weight_path"])
    capture = ActivationCapture(model.layer4[-1])
    sae, _, kept_mask = load_locked_sae(sae_run, device)
    fc_margin = model.fc.weight[1] - model.fc.weight[0]
    cancer_direction = (
        sae.decoder_weight @ fc_margin
    ).detach().cpu().numpy()

    records = []
    for index, row in enumerate(rows.itertuples(index=False)):
        records.append(
            process_image(
                row,
                index,
                output_dir,
                model,
                capture,
                sae,
                kept_mask,
                cancer_direction,
                names,
                args,
                device,
            )
        )
        print(f"完成 [{index + 1}/{len(rows)}]: {row.image_path}")
    capture.close()
    pd.DataFrame(records).to_csv(
        output_dir / "image_centered_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
    (output_dir / "README.md").write_text(
        """# 图像中心SAE可视化

- `Top_feature多颜色叠加图.png`：不同颜色对应不同SAE feature，图例给出激活和癌贡献。
- `rank_XX_feature_XXXX.png`：每个feature单独的空间响应图。
- `Top_feature统计.csv`：激活、癌方向和癌logit margin贡献。
- `图像中心_SAE_Top_feature总览.png`：原图、综合图和独立feature图。

“癌贡献”是`feature激活 × decoder癌logit方向`，单位是二分类logit margin，不是癌概率
百分点。默认`color_mode=contribution`：正向推动癌判断为高饱和红色，负向为高饱和
蓝色，绝对贡献越大、空间响应越高，叠加颜色越明显；`color_mode=feature`则使用不同
类别色区分feature身份。`overlay_alpha`控制热点最大叠加强度，`response_gamma`控制
中等响应的显色程度。热图是7×7深层特征响应上采样后的大致位置，不是病灶分割结果。
""",
        encoding="utf-8",
    )
    print(f"图像中心SAE结果: {output_dir}")


if __name__ == "__main__":
    main()
