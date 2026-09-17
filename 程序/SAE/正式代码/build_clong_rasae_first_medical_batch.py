#!/usr/bin/env python3
"""按已有单Feature删除效应整理首批RA-SAE医学自由描述图片。

候选只使用现有val98残差保留删除的患者等权绝对delta margin排名，
每个技术组本批最多取一项。医学包不显示功能方向、排名或癌/非癌标签，
只要求判断跨图是否有稳定可描述的视觉内容。
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
GROUP_ROOT = BASE / "full_dictionary_groups_20260909"
USAGE_ROOT = BASE / "feature_usage_audit_20260915"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
K = 256
BATCH_FEATURES = 8
HEAT_ALPHA = 0.25
CONTOUR_THRESHOLD = 0.5
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def select_candidates(statistics: pd.DataFrame) -> pd.DataFrame:
    """按功能绝对效应选取首批不同技术组的Feature。

    Args:
        statistics (pd.DataFrame): 2560行Feature使用和功能统计。

    Returns:
        pd.DataFrame: 排名顺序的8项Feature，同一``group``仅保留首项。
    """
    ordered = statistics.sort_values(
        ["val98_abs_effect_rank_desc", "feature_id"], kind="stable"
    )
    selected = ordered.drop_duplicates("group", keep="first").head(BATCH_FEATURES).copy()
    selected.insert(0, "batch_order", np.arange(1, len(selected) + 1))
    if len(selected) != BATCH_FEATURES or selected.group.nunique() != BATCH_FEATURES:
        raise RuntimeError("无法选出8个不同技术组的候选")
    return selected


def select_images(
    candidates: pd.DataFrame, frames: dict[str, pd.DataFrame], peaks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """为每项Feature选取train/val、癌/非癌各2位高响应患者。

    Args:
        candidates (pd.DataFrame): 首批8项Feature。
        frames (dict[str, pd.DataFrame]): train/val全量元数据。
        peaks (dict[str, np.ndarray]): train/val``[N,2560]`` peak/Q99矩阵。

    Returns:
        pd.DataFrame: 每项8张、同一Feature内患者不重复的选例表。
    """
    rows = []
    for candidate in candidates.itertuples():
        feature = int(candidate.feature_id)
        used: set[str] = set()
        for split in ("train", "val"):
            frame = frames[split]
            score = peaks[split][:, feature]
            for label in (1, 0):
                count = 0
                for index in np.argsort(-score, kind="stable"):
                    record = frame.iloc[int(index)]
                    patient = str(record.patient_id)
                    if int(record.label) != label or patient in used or score[index] <= 0:
                        continue
                    used.add(patient)
                    rows.append({
                        "batch_order": int(candidate.batch_order), "feature_id": feature,
                        "group": candidate.group, "split": split, "source_row": int(index),
                        "image_code": f"{split}_{int(index):04d}", "label_internal": label,
                        "patient_id": patient, "image_relpath": record.image_relpath,
                        "peak_q99": float(score[index]),
                    })
                    count += 1
                    if count == 2:
                        break
                if count != 2:
                    raise RuntimeError(f"RA-F{feature:04d} {split}/label={label}不足2位高响应患者")
        if len(used) != 8:
            raise RuntimeError(f"RA-F{feature:04d}未获得8位不同患者")
    return pd.DataFrame(rows)


def project_selected_maps(
    selected: pd.DataFrame, frames: dict[str, pd.DataFrame], device: torch.device, batch_size: int,
) -> dict[tuple[str, int], np.ndarray]:
    """仅投影选中图片并保留其``[49,2560]`` RA-SAE激活。

    Args:
        selected (pd.DataFrame): 64张固定选例。
        frames (dict[str, pd.DataFrame]): train/val全量元数据。
        device (torch.device): CPU或已核对的CUDA设备。
        batch_size (int): 小批投影图片数。

    Returns:
        dict[tuple[str, int], np.ndarray]: ``(split,source_row)``到空间激活的映射。
    """
    sae, _, _, _ = load_models(device)
    maps: dict[tuple[str, int], np.ndarray] = {}
    with torch.no_grad():
        for split in ("train", "val"):
            indices = sorted(selected.loc[selected.split.eq(split), "source_row"].astype(int).unique())
            for start in range(0, len(indices), batch_size):
                current = indices[start:start + batch_size]
                data = read_subset(frames[split].iloc[current], split, device)
                hidden = sae.encode(data["spatial"], K).cpu().numpy()
                for local, source_row in enumerate(current):
                    maps[(split, source_row)] = hidden[local]
    expected = selected[["split", "source_row"]].drop_duplicates().shape[0]
    if len(maps) != expected:
        raise RuntimeError("选例激活未全部投影")
    return maps


def render_features(
    candidates: pd.DataFrame, selected: pd.DataFrame, maps: dict[tuple[str, int], np.ndarray],
    q99: np.ndarray, output: Path,
) -> None:
    """生成8张“原图+单Feature淡热图”医学复核图。

    Args:
        candidates (pd.DataFrame): 8项Feature及内部选择顺序。
        selected (pd.DataFrame): 每项8张图的选例表。
        maps (dict): 选中图片的``[49,2560]``激活。
        q99 (np.ndarray): 2560项train-only正激活Q99。
        output (Path): PNG输出目录。

    Returns:
        None: 一项Feature写入一张PNG。
    """
    output.mkdir(parents=True, exist_ok=True)
    for candidate in candidates.itertuples():
        feature = int(candidate.feature_id)
        rows = selected[selected.feature_id.eq(feature)].reset_index(drop=True)
        figure, axes = plt.subplots(8, 2, figsize=(10, 30), squeeze=False)
        for row_index, record in rows.iterrows():
            image = Image.open(ROOT / record.image_relpath).convert("RGB")
            image.thumbnail((800, 800))
            rgb = np.asarray(image)
            normalized = maps[(record.split, int(record.source_row))][:, feature].reshape(7, 7) / q99[feature]
            dense = torch.nn.functional.interpolate(
                torch.from_numpy(normalized).reshape(1, 1, 7, 7), size=rgb.shape[:2],
                mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 0].set_title(str(record.image_code), fontsize=10)
            axes[row_index, 1].imshow(rgb)
            axes[row_index, 1].imshow(
                dense, cmap="magma", vmin=0, vmax=1, alpha=HEAT_ALPHA, interpolation="nearest"
            )
            if float(dense.max()) >= CONTOUR_THRESHOLD > float(dense.min()):
                axes[row_index, 1].contour(
                    dense, levels=[CONTOUR_THRESHOLD], colors=["#00ff66"], linewidths=1.8
                )
            elif float(dense.min()) >= CONTOUR_THRESHOLD:
                axes[row_index, 1].add_patch(Rectangle(
                    (-0.5, -0.5), rgb.shape[1], rgb.shape[0], fill=False,
                    edgecolor="#00ff66", linewidth=1.8,
                ))
            axes[row_index, 1].set_title(f"{record.image_code} | Feature响应", fontsize=10)
            for axis in axes[row_index]:
                axis.axis("off")
        figure.suptitle(
            f"RA-F{feature:04d}\n左：原图　右：Feature响应　绿色线：较强激活区域（不是病灶边界）",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(output / f"RA-F{feature:04d}.png", dpi=140, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    """选取首批功能相关Feature并生成内部产物与医学图片。

    Args:
        None: 输出目录、设备和批量由CLI提供。

    Returns:
        None: 写入选例、口径、核对与8张PNG。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    statistics = pd.read_csv(USAGE_ROOT / "feature_usage_statistics.csv")
    candidates = select_candidates(statistics)
    frames = {
        split: pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(columns={"index": "source_row"})
        for split in ("train", "val")
    }
    if set(frames["train"].patient_id) & set(frames["val"].patient_id):
        raise RuntimeError("train/val患者交叉")
    peaks = {split: np.load(GROUP_ROOT / f"{split}_peak_q99.npy", mmap_mode="r")
             for split in ("train", "val")}
    selected = select_images(candidates, frames, peaks)
    candidates.to_csv(args.output / "candidate_features_internal.csv", index=False)
    selected.to_csv(args.output / "selected_images_internal.csv", index=False)
    definition = {
        "candidate_count": BATCH_FEATURES,
        "candidate_selection": "descending existing val98 patient-equal mean per-image abs(delta_margin); one Feature per technical group",
        "image_selection": "for each Feature, top two distinct patients in each train/val x cancer/noncancer stratum",
        "medical_blinding": "PNG omits label, functional rank, effect size and direction",
        "heatmap": "unclipped 7x7 response bilinear interpolation; alpha 0.25; same dense map for 0.5xQ99 contour",
        "purpose": "first-pass free description of stable/possible artifact/mixed/no-obvious-pattern",
        "no_family_merge": True, "no_deletion": True, "new_training": False,
        "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    maps = project_selected_maps(selected, frames, torch.device(args.device), args.batch_size)
    q99 = np.load(BASE / "decision_alignment/ra_train_q99.npy")
    render_features(candidates, selected, maps, q99, args.output / "医学复核图")
    verification = {
        "candidate_features": [f"RA-F{int(value):04d}" for value in candidates.feature_id],
        "unique_technical_groups": int(candidates.group.nunique()),
        "selected_images": int(len(selected)),
        "patients_unique_within_feature": bool(
            selected.groupby("feature_id").patient_id.nunique().eq(8).all()
        ),
        "png_count": len(list((args.output / "医学复核图").glob("*.png"))),
        "labels_hidden_in_png": True, "functional_results_hidden_in_png": True,
        "test_read": False, "external_read": False,
    }
    if verification["png_count"] != BATCH_FEATURES or not verification["patients_unique_within_feature"]:
        raise RuntimeError("医学首批图片数或患者去重异常")
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
