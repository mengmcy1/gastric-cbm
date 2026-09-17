#!/usr/bin/env python3
"""生成RA-SAE候选视觉线索的三张盲化同图Feature并排图。

三条候选线索仅保留在内部manifest；医学图只显示“对照A/B/C”、图片代号和
Feature编号，不标记主例、反例、癌/非癌或功能结果。所有原图均来自已审核首批，
部分Feature在同图上的响应为本轮新增并排展示。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
FIRST_BATCH = BASE / "functional_first_medical_batch_20260915"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
K = 256
HEAT_ALPHA = 0.25
CONTOUR_THRESHOLD = 0.5
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


COMPARISONS = {
    "A": {
        "internal_clue": "凹陷/发红相关",
        "features": [219, 868, 2388, 2444],
        "images": [
            "val_0317", "val_0315", "val_0496", "val_0424",
            "val_0045", "train_1222", "train_0507", "train_0983",
        ],
    },
    "B": {
        "internal_clue": "隆起/边界相关",
        "features": [1385, 2464],
        "images": ["train_1334", "val_0295", "train_0382", "val_0132", "train_1326", "val_0384"],
    },
    "C": {
        "internal_clue": "幽门附近结构相关",
        "features": [601, 1089],
        "images": ["train_0277", "val_0106", "val_0489", "val_0317", "val_0132", "val_0288"],
    },
}


def resolve_images(selection: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """将通过审核的图片代号对应到缓存行与原图路径。

    Args:
        selection (pd.DataFrame): 首批64张选例明细。
        frames (dict[str, pd.DataFrame]): train/val全量缓存元数据。

    Returns:
        pd.DataFrame: 每个对照图位一行，包含内部线索、图片代号和路径。
    """
    known = selection[["image_code", "split", "source_row"]].drop_duplicates()
    if known.image_code.duplicated().any():
        raise RuntimeError("同一图片代号对应多个缓存行")
    known = known.set_index("image_code")
    rows = []
    for comparison, config in COMPARISONS.items():
        for order, code in enumerate(config["images"], 1):
            if code not in known.index:
                raise RuntimeError(f"对照{comparison}含未审核原图: {code}")
            split = str(known.loc[code, "split"])
            source_row = int(known.loc[code, "source_row"])
            record = frames[split].iloc[source_row]
            rows.append({
                "comparison": comparison, "internal_clue": config["internal_clue"],
                "row_order": order, "image_code": code, "split": split,
                "source_row": source_row, "patient_id": str(record.patient_id),
                "image_relpath": record.image_relpath,
                "features": ",".join(f"RA-F{feature:04d}" for feature in config["features"]),
            })
    manifest = pd.DataFrame(rows)
    expected = {"A": 8, "B": 6, "C": 6}
    if manifest.groupby("comparison").size().to_dict() != expected:
        raise RuntimeError("三张对照图的行数不是8/6/6")
    if not manifest.groupby("comparison").patient_id.nunique().eq(manifest.groupby("comparison").size()).all():
        raise RuntimeError("同一对照图出现重复患者")
    return manifest


def project_maps(
    manifest: pd.DataFrame, frames: dict[str, pd.DataFrame], device: torch.device, batch_size: int,
) -> dict[tuple[str, int], np.ndarray]:
    """仅为18张已看原图重算RA-SAE K=256空间激活。

    Args:
        manifest (pd.DataFrame): 20个图位、18张唯一原图的对照manifest。
        frames (dict[str, pd.DataFrame]): train/val全量元数据。
        device (torch.device): CPU或已核对的CUDA设备。
        batch_size (int): 投影批量。

    Returns:
        dict[tuple[str, int], np.ndarray]: ``(split,source_row)``到``[49,2560]``激活。
    """
    sae, _, _, _ = load_models(device)
    maps: dict[tuple[str, int], np.ndarray] = {}
    unique = manifest[["split", "source_row"]].drop_duplicates()
    with torch.no_grad():
        for split in ("train", "val"):
            indices = sorted(unique.loc[unique.split.eq(split), "source_row"].astype(int))
            for start in range(0, len(indices), batch_size):
                current = indices[start:start + batch_size]
                data = read_subset(frames[split].iloc[current], split, device)
                hidden = sae.encode(data["spatial"], K).cpu().numpy()
                for local, source_row in enumerate(current):
                    maps[(split, source_row)] = hidden[local]
    if len(maps) != len(unique):
        raise RuntimeError("未恰好投影全18张唯一原图")
    return maps


def render_comparisons(
    manifest: pd.DataFrame, maps: dict[tuple[str, int], np.ndarray], q99: np.ndarray, output: Path,
) -> None:
    """生成对照A/B/C三张医学盲化长图。

    Args:
        manifest (pd.DataFrame): 对照图的固定行顺序与原图路径。
        maps (dict): 18张原图的``[49,2560]``激活。
        q99 (np.ndarray): 2560项train-only正激活Q99尺度。
        output (Path): 三张PNG的输出目录。

    Returns:
        None: 弱响应保留原尺度，不换图、不逐图增强。
    """
    output.mkdir(parents=True, exist_ok=True)
    for comparison, config in COMPARISONS.items():
        rows = manifest[manifest.comparison.eq(comparison)].sort_values("row_order")
        features = config["features"]
        figure, axes = plt.subplots(
            len(rows), len(features) + 1,
            figsize=(5 * (len(features) + 1), 3.8 * len(rows)), squeeze=False,
        )
        for row_index, record in enumerate(rows.itertuples()):
            image = Image.open(ROOT / record.image_relpath).convert("RGB")
            image.thumbnail((800, 800))
            rgb = np.asarray(image)
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 0].set_title(f"{record.image_code} | 原图", fontsize=10)
            for column, feature in enumerate(features, 1):
                normalized = maps[(record.split, int(record.source_row))][:, feature].reshape(7, 7) / q99[feature]
                dense = torch.nn.functional.interpolate(
                    torch.from_numpy(normalized).reshape(1, 1, 7, 7), size=rgb.shape[:2],
                    mode="bilinear", align_corners=False,
                )[0, 0].numpy()
                axes[row_index, column].imshow(rgb)
                axes[row_index, column].imshow(
                    dense, cmap="magma", vmin=0, vmax=1, alpha=HEAT_ALPHA, interpolation="nearest"
                )
                if float(dense.max()) >= CONTOUR_THRESHOLD > float(dense.min()):
                    axes[row_index, column].contour(
                        dense, levels=[CONTOUR_THRESHOLD], colors=["#00ff66"], linewidths=1.8
                    )
                elif float(dense.min()) >= CONTOUR_THRESHOLD:
                    axes[row_index, column].add_patch(Rectangle(
                        (-0.5, -0.5), rgb.shape[1], rgb.shape[0], fill=False,
                        edgecolor="#00ff66", linewidth=1.8,
                    ))
                axes[row_index, column].set_title(f"{record.image_code} | RA-F{feature:04d}", fontsize=10)
            for axis in axes[row_index]:
                axis.axis("off")
        feature_text = "、".join(f"RA-F{feature:04d}" for feature in features)
        figure.suptitle(
            f"对照{comparison} | {feature_text}\n"
            "绿色线：较强激活区域（不是病灶边界）",
            fontsize=15,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        figure.savefig(output / f"对照{comparison}.png", dpi=130, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    """解析命令行并生成三张审核通过的候选视觉线索对照图。

    Args:
        None: 输出目录、设备与批量由CLI提供。

    Returns:
        None: 写入内部manifest、定义、验证和3张PNG。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = pd.read_csv(FIRST_BATCH / "selected_images_internal.csv")
    frames = {
        split: pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(columns={"index": "source_row"})
        for split in ("train", "val")
    }
    manifest = resolve_images(selection, frames)
    manifest.to_csv(args.output / "comparison_manifest_internal.csv", index=False)
    definition = {
        "comparisons_internal": COMPARISONS,
        "medical_titles": "comparison A/B/C plus Feature IDs only",
        "source_images": "all previously viewed; 20 panel rows / 18 unique images",
        "new_content": "some Feature responses are newly shown side-by-side on previously viewed originals",
        "cross_figure_repeats": {"val_0317": ["A", "C"], "val_0132": ["B", "C"]},
        "display": "fixed per-Feature train Q99; alpha=0.25; no per-image enhancement or replacement",
        "medical_hidden": ["candidate clue name", "main/counterexample status", "label", "functional rank/effect"],
        "new_training": False, "new_clustering": False, "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    maps = project_maps(manifest, frames, torch.device(args.device), args.batch_size)
    q99 = np.load(BASE / "decision_alignment/ra_train_q99.npy")
    render_comparisons(manifest, maps, q99, args.output / "医学对照图")
    verification = {
        "panel_rows": manifest.groupby("comparison").size().to_dict(),
        "unique_images": int(manifest[["split", "source_row"]].drop_duplicates().shape[0]),
        "patients_unique_within_each_comparison": True,
        "all_source_images_from_first_batch": True,
        "png_count": len(list((args.output / "医学对照图").glob("*.png"))),
        "weak_responses_retained": True, "per_image_rescaling": False,
        "test_read": False, "external_read": False,
    }
    if verification["panel_rows"] != {"A": 8, "B": 6, "C": 6} or verification["png_count"] != 3:
        raise RuntimeError("三张对照图验证失败")
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
