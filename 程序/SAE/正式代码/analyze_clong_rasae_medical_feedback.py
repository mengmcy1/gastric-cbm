#!/usr/bin/env python3
"""整理RA-SAE医学反馈、复核重点混杂组并执行四个候选方向组干预。

脚本只读固定RA-SAE 100轮权重、K=256、现有train/val空间特征缓存和
全字典分组。它不重训、不重分组、不读取test/external，也不将医生对
技术代表的初步观察自动扩展到组内所有成员。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "程序/MAGE/正式代码"))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from build_mage_teacher_roi_manifest import file_sha256
from clong_rasae_core import ArchetypalMatryoshkaSAE
from clong_rpc_core import remove_feature_group
from clong_s2b_core import attention_from_features, pooled_from_features
from clong_sae_discovery import CLONG_CHECKPOINT_SHA256, FROZEN_PATIENT_THRESHOLD
from run_clong_rasae_pilot import read_subset


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
GROUP_ROOT = BASE / "full_dictionary_groups_20260909"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
K = 256
HEAT_ALPHA = 0.25
CONTOUR_THRESHOLD = 0.5
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


FEEDBACK_CATEGORIES = [
    ("黑色四角边框", "001 002 011 012 034 046 062 068 076 080 204"),
    ("透明帽", "003 005 010 036 037 041 044 047 070 071 072 073 078 096 104 123 203 216"),
    ("反光", "2313 009 033 049 061 137 143 233 266 267"),
    ("活检钳", "107"),
    ("红色凹陷", "014"),
    ("发红？", "020 217 257"),
    ("发红，但有图片响应泡沫", "031"),
    ("幽门口", "022 052 067 095 099 110 138 189"),
    ("镜身，可能与部位有关", "039 074 112"),
    ("发白、萎缩黏膜", "025"),
    ("白苔", "028 149"),
    ("溃疡、白苔", "040 113"),
    ("萎缩黏膜", "035 105"),
    ("凹凸不平", "050 177"),
    ("凹陷？", "026 075"),
    ("凹陷或纹理？", "053"),
    ("凹陷或边界？", "194"),
    ("隆起", "069"),
    ("胃体皱襞", "092 094 157 160"),
]

MIXED_FEEDBACK = {
    "G0038": "有的响应病灶，有的响应边框、透明帽。",
    "G0048": "像病灶隆起边缘或凹陷，但有图片响应镜身。",
    "G0055": "像凹陷，但有图片激活位置不在病灶。",
    "G0097": "像病灶凹陷，但有图片响应幽门口。",
    "G0174": "像隆起型病灶，但有图片响应透明帽或较暗区域。",
}

PRIORITY_GROUPS = ("G0031", "G0038", "G0048", "G0055", "G0097", "G0174")
INTERVENTIONS = {
    "黑色四角边框": "G0080",
    "透明帽": "G0203",
    "反光": "G2313",
    "活检钳": "G0107",
}


def group_id(value: str) -> str:
    """将医学反馈中的数字组号标准化为G加四位数。

    Args:
        value (str): 如``001``或``2313``的原始组号。

    Returns:
        str: 如``G0001``或``G2313``的现有分组主键。
    """
    return "G" + value.removeprefix("G").zfill(4)


def parse_members(value: str) -> list[int]:
    """解析groups.csv的成员列。

    Args:
        value (str): 逗号分隔的Feature ID。

    Returns:
        list[int]: 保持原表顺序的Feature ID。
    """
    return [int(item) for item in str(value).split(",")]


def build_feedback_tables(groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成保留原始描述的观察表和独立组号解析表。

    Args:
        groups (pd.DataFrame): 以``group``为索引的现有分组表。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 观察表和79个唯一反馈组的成员表。
    """
    rows = []
    for description, values in FEEDBACK_CATEGORIES:
        for value in values.split():
            gid = group_id(value)
            counterexample = ""
            if gid == "G0031":
                counterexample = "有图片响应泡沫；未提供具体图片代号"
            rows.append({
                "group_as_received": "G" + value,
                "group": gid,
                "raw_description": description,
                "question_mark_preserved": "？" in description or "?" in description,
                "counterexample_or_caveat": counterexample,
                "priority_mixed_review": gid in PRIORITY_GROUPS,
                "review_scope": "医学生仅查看部分图片；初步观察，非全量标注或已确认概念",
                "applies_to": "医生观察的代表项/图片；不自动扩展到组内全部成员",
            })
    for gid, description in MIXED_FEEDBACK.items():
        rows.append({
            "group_as_received": gid.replace("G0", "G", 1),
            "group": gid,
            "raw_description": description,
            "question_mark_preserved": "？" in description or "?" in description,
            "counterexample_or_caveat": description + " 未提供具体图片代号",
            "priority_mixed_review": True,
            "review_scope": "医学生仅查看部分图片；初步观察，非全量标注或已确认概念",
            "applies_to": "医生观察的代表项/图片；不自动扩展到组内全部成员",
        })
    observations = pd.DataFrame(rows)
    if observations.group.nunique() != 79 or len(observations) != 79:
        raise RuntimeError("医学反馈应恰好解析为79个唯一组")
    missing = sorted(set(observations.group) - set(groups.index))
    if missing:
        raise RuntimeError(f"医学反馈含不存在组号: {missing}")
    lookup_rows = []
    for gid in observations.group:
        record = groups.loc[gid]
        members = parse_members(record.members)
        lookup_rows.append({
            "group": gid,
            "size": int(record["size"]),
            "technical_representative": f"RA-F{int(record.representative):04d}",
            "members": ",".join(f"RA-F{member:04d}" for member in members),
        })
    lookup = pd.DataFrame(lookup_rows)
    if lookup["size"].sum() != 161:
        raise RuntimeError("反馈组成员总数不是已核对的161")
    return observations, lookup


def load_models(device: torch.device) -> tuple[ArchetypalMatryoshkaSAE, nn.Module, torch.Tensor, torch.Tensor]:
    """读取固定RA-SAE和C-long下游注意力/分类头。

    Args:
        device (torch.device): CPU或已核对的CUDA设备。

    Returns:
        tuple: RA-SAE、冻结注意力头、``[2,1280]``分类权重和``[2]``偏置。
    """
    checkpoint = torch.load(BASE / "duration100/ra_final.pth", map_location=device, weights_only=False)
    state, config = checkpoint["state_dict"], checkpoint["config"]
    if config["hidden_dim"] != 2560 or K not in config["k_list"]:
        raise RuntimeError("RA-SAE权重不是预期的2560宽度/K=256")
    sae = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], config["hidden_dim"], tuple(config["k_list"]),
        config["delta"], True, config["seed"], config["initialization"],
    ).to(device)
    sae.load_state_dict(state)
    sae.requires_grad_(False).eval()
    summary_path = ROOT / "结果/MAGE/MG2L训练轮数敏感性_20260818/mg2l_attention_sae_summary_seed42.json"
    clong_path = Path(json.loads(summary_path.read_text())["selected_checkpoint"])
    if file_sha256(clong_path) != CLONG_CHECKPOINT_SHA256:
        raise RuntimeError("C-long checkpoint SHA不等于冻结值")
    clong_state = torch.load(clong_path, map_location=device, weights_only=False)["model_state_dict"]
    head = nn.Conv2d(1280, 1, 1).to(device)
    head.load_state_dict({"weight": clong_state["attention_head.weight"],
                          "bias": clong_state["attention_head.bias"]})
    head.requires_grad_(False).eval()
    return sae, head, clong_state["classifier.1.weight"].to(device), clong_state["classifier.1.bias"].to(device)


def choose_review_images(
    frames: dict[str, pd.DataFrame], peaks: dict[str, np.ndarray], groups: pd.DataFrame,
) -> pd.DataFrame:
    """按冻结组高响应和成员归一化峰值差选择不重复患者。

    Args:
        frames (dict[str, pd.DataFrame]): train/val全量缓存元数据。
        peaks (dict[str, np.ndarray]): train/val的``[N,2560]`` peak/Q99。
        groups (pd.DataFrame): 以组号为索引的分组表。

    Returns:
        pd.DataFrame: 每组最多8张、组内患者唯一的确定性选例表。
    """
    rows = []
    for gid in PRIORITY_GROUPS:
        members = parse_members(groups.loc[gid, "members"])
        if len(members) != 2:
            raise RuntimeError(f"{gid}本轮预期为二项组")
        used_patients: set[str] = set()

        def take(split: str, scores: np.ndarray, reason: str, label: int | None = None) -> None:
            frame = frames[split]
            order = np.argsort(-scores, kind="stable")
            for index in order:
                record = frame.iloc[int(index)]
                patient = str(record.patient_id)
                if scores[index] <= 0 or patient in used_patients:
                    continue
                if label is not None and int(record.label) != label:
                    continue
                used_patients.add(patient)
                rows.append({"group": gid, "split": split, "source_row": int(index),
                             "label": int(record.label), "patient_id": patient,
                             "image_relpath": record.image_relpath, "selection_reason": reason,
                             "selection_score": float(scores[index])})
                return

        for split in ("train", "val"):
            group_score = peaks[split][:, members].max(1)
            for label in (1, 0):
                take(split, group_score, f"组高响应：max(成员peak/Q99)，label={label}", label)
        for member in members:
            other = members[1] if member == members[0] else members[0]
            for split in ("train", "val"):
                dominance = peaks[split][:, member] - peaks[split][:, other]
                take(split, dominance,
                     f"RA-F{member:04d}占优：该成员peak/Q99-另一成员peak/Q99")
    selected = pd.DataFrame(rows)
    if selected.groupby("group").patient_id.nunique().ne(selected.groupby("group").size()).any():
        raise RuntimeError("重点组选例出现患者重复")
    return selected


def patient_table(image_rows: pd.DataFrame) -> pd.DataFrame:
    """将干预的逐图变化按现有患者概率聚合口径汇总。

    Args:
        image_rows (pd.DataFrame): 单一split的逐图干预明细。

    Returns:
        pd.DataFrame: 每组每患者一行，同时保留带符号均值与逐图绝对变化均值。
    """
    work = image_rows.copy()
    work["abs_image_delta_margin"] = work.delta_margin.abs()
    work["abs_image_delta_probability"] = work.delta_probability.abs()
    grouped = work.groupby(["split", "group", "feedback_category", "patient_id"], sort=True)
    result = grouped.agg(
        label=("label", "first"), image_count=("source_row", "size"),
        original_probability=("original_probability", "mean"),
        ablated_probability=("ablated_probability", "mean"),
        signed_mean_delta_margin=("delta_margin", "mean"),
        mean_image_abs_delta_margin=("abs_image_delta_margin", "mean"),
        signed_mean_delta_probability=("delta_probability", "mean"),
        mean_image_abs_delta_probability=("abs_image_delta_probability", "mean"),
        mean_attention_l1=("attention_l1", "mean"),
        mean_attention_cosine=("attention_cosine", "mean"),
        peak_change_fraction=("peak_changed", "mean"),
        active=("active", "max"),
    ).reset_index()
    return result


def summarize_interventions(
    image_effects: pd.DataFrame, patient_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别生成癌/非癌影响与合并队列AUC/翻转结果。

    Args:
        image_effects (pd.DataFrame): 逐图效应与激活明细。
        patient_effects (pd.DataFrame): 逐患者聚合结果。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 分类患者统计和癌/非癌合并队列总体统计。
    """
    class_rows, overall_rows = [], []
    for (split, gid, category), frame in patient_effects.groupby(
        ["split", "group", "feedback_category"], sort=False
    ):
        original_prediction = frame.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
        changed_prediction = frame.ablated_probability.ge(FROZEN_PATIENT_THRESHOLD)
        original_correct = original_prediction.eq(frame.label.astype(bool))
        changed_correct = changed_prediction.eq(frame.label.astype(bool))
        overall_rows.append({
            "split": split, "group": gid, "feedback_category": category,
            "patients": len(frame),
            "original_patient_auc_combined_classes": roc_auc_score(frame.label, frame.original_probability),
            "ablated_patient_auc_combined_classes": roc_auc_score(frame.label, frame.ablated_probability),
            "auc_change": roc_auc_score(frame.label, frame.ablated_probability)
                          - roc_auc_score(frame.label, frame.original_probability),
            "patient_flip_count": int((original_prediction != changed_prediction).sum()),
            "prediction_cancer_to_noncancer_count": int((original_prediction & ~changed_prediction).sum()),
            "prediction_noncancer_to_cancer_count": int((~original_prediction & changed_prediction).sum()),
            "incorrect_to_correct_count": int((~original_correct & changed_correct).sum()),
            "correct_to_incorrect_count": int((original_correct & ~changed_correct).sum()),
            "patient_threshold": FROZEN_PATIENT_THRESHOLD,
        })
        image_subset = image_effects[(image_effects.split == split) & (image_effects.group == gid)]
        for label, label_name in ((1, "cancer"), (0, "noncancer")):
            selected = frame[frame.label.eq(label)]
            selected_images = image_subset[image_subset.label.eq(label)]
            class_rows.append({
                "split": split, "group": gid, "feedback_category": category,
                "label": label, "label_name": label_name, "patients": len(selected),
                "images": len(selected_images), "active_patients": int(selected.active.sum()),
                "mean_patient_group_peak_q99": float(selected_images.groupby("patient_id").group_peak_q99.mean().mean()),
                "mean_patient_image_active_fraction": float(selected_images.groupby("patient_id").active.mean().mean()),
                "mean_patient_signed_delta_margin": float(selected.signed_mean_delta_margin.mean()),
                "median_patient_signed_delta_margin": float(selected.signed_mean_delta_margin.median()),
                "mean_patient_of_mean_image_abs_delta_margin": float(selected.mean_image_abs_delta_margin.mean()),
                "mean_patient_signed_delta_probability": float(selected.signed_mean_delta_probability.mean()),
                "mean_patient_of_mean_image_abs_delta_probability": float(selected.mean_image_abs_delta_probability.mean()),
                "mean_patient_attention_l1": float(selected.mean_attention_l1.mean()),
                "mean_patient_attention_cosine": float(selected.mean_attention_cosine.mean()),
                "mean_patient_peak_change_fraction": float(selected.peak_change_fraction.mean()),
            })
    return pd.DataFrame(class_rows), pd.DataFrame(overall_rows)


def render_review_figures(
    selected: pd.DataFrame, groups: pd.DataFrame, maps: dict[tuple[str, int], np.ndarray],
    q99: np.ndarray, output: Path,
) -> None:
    """为六个重点组生成原图与全成员淡热图并排图。

    Args:
        selected (pd.DataFrame): 固定选例表。
        groups (pd.DataFrame): 现有分组表。
        maps (dict): ``(split, source_row)``到``[49,2560]``激活的映射。
        q99 (np.ndarray): train-only正激活Q99，shape为``[2560]``。
        output (Path): 图片输出目录。

    Returns:
        None: 每组写入一张PNG，不修改原图。
    """
    output.mkdir(parents=True, exist_ok=True)
    for gid in PRIORITY_GROUPS:
        rows = selected[selected.group.eq(gid)].reset_index(drop=True)
        members = parse_members(groups.loc[gid, "members"])
        figure, axes = plt.subplots(len(rows), 3, figsize=(15, 3.8 * len(rows)), squeeze=False)
        for row_index, record in rows.iterrows():
            image = Image.open(ROOT / record.image_relpath).convert("RGB")
            image.thumbnail((800, 800))
            rgb = np.asarray(image)
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 0].set_title(
                f"{record.split}_{int(record.source_row):04d} | {'癌' if record.label else '非癌'}\n"
                f"{record.selection_reason}", fontsize=9,
            )
            for column, member in enumerate(members, 1):
                normalized = maps[(record.split, int(record.source_row))][:, member].reshape(7, 7) / q99[member]
                dense = torch.nn.functional.interpolate(
                    torch.from_numpy(normalized).reshape(1, 1, 7, 7),
                    size=rgb.shape[:2], mode="bilinear", align_corners=False,
                )[0, 0].numpy()
                axes[row_index, column].imshow(rgb)
                axes[row_index, column].imshow(
                    dense, cmap="magma", vmin=0, vmax=1, alpha=HEAT_ALPHA,
                    interpolation="nearest",
                )
                threshold_note = ""
                if float(dense.max()) >= CONTOUR_THRESHOLD > float(dense.min()):
                    axes[row_index, column].contour(
                        dense, levels=[CONTOUR_THRESHOLD], colors=["#00ff66"], linewidths=1.8,
                    )
                elif float(dense.min()) >= CONTOUR_THRESHOLD:
                    axes[row_index, column].add_patch(Rectangle(
                        (-0.5, -0.5), rgb.shape[1], rgb.shape[0], fill=False,
                        edgecolor="#00ff66", linewidth=1.8,
                    ))
                else:
                    threshold_note = "\n未达到0.5×Q99显示阈值"
                axes[row_index, column].set_title(
                    f"RA-F{member:04d} | peak/Q99={normalized.max():.3f}{threshold_note}", fontsize=9,
                )
            for axis in axes[row_index]:
                axis.axis("off")
        figure.suptitle(
            f"{gid}：同一批图片并排全部成员\n"
            "绿色线=高激活区域（≥0.5×train Q99），不是病灶边界；热图透明度=0.25",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output / f"{gid}_全成员同图并排.png", dpi=140, bbox_inches="tight")
        plt.close(figure)


def repair_existing_delivery(output: Path, device: torch.device, batch_size: int) -> None:
    """从已有明细表修正翻转列，并仅重算48张选例激活以重画六张图。

    Args:
        output (Path): 已完成全量干预的结果目录。
        device (torch.device): 用于小批RA-SAE投影的CPU或CUDA设备。
        batch_size (int): 每次读取的选例数。

    Returns:
        None: 覆盖本轮六张有误图片和翻转汇总，不重算全量干预。
    """
    if not output.is_dir():
        raise FileNotFoundError(f"待修正结果目录不存在: {output}")
    groups = pd.read_csv(GROUP_ROOT / "groups.csv").set_index("group")
    selected = pd.read_csv(output / "重点混杂组选例.csv")
    frames = {
        split: pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(columns={"index": "source_row"})
        for split in ("train", "val")
    }
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
    if len(maps) != selected[["split", "source_row"]].drop_duplicates().shape[0]:
        raise RuntimeError("修正绘图时选例激活未全部复算")
    q99 = np.load(BASE / "decision_alignment/ra_train_q99.npy")
    render_review_figures(selected, groups, maps, q99, output / "重点混杂组并排图")
    image_effects = pd.read_csv(output / "四组干预_逐图结果.csv")
    patients = pd.read_csv(output / "四组干预_逐患者结果.csv")
    _, overall_metrics = summarize_interventions(image_effects, patients)
    overall_metrics.to_csv(output / "四组干预_合并队列AUC与翻转.csv", index=False)
    definition_path = output / "analysis_definition.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    definition["heatmap"]["interpolation"] = (
        "unclipped 7x7 response bilinearly interpolated with align_corners=False; "
        "the same dense response drives heatmap colors and 0.5 contour"
    )
    definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    verification_path = output / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification.update({
        "figures_recomputed_from_selected_48_only": True,
        "heatmap_contour_same_interpolated_response": True,
        "flip_direction_columns_corrected_from_existing_patient_csv": True,
        "full_intervention_rerun_for_delivery_repair": False,
    })
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """读取CLI输出路径和设备，执行反馈解析、小批图与四组干预。

    Args:
        None: 参数由命令行解析。

    Returns:
        None: 写入独立结果目录；已存在目录会快速失败。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repair-existing", action="store_true",
                        help="仅修正已有结果的六张图和翻转汇总，不重跑全量干预")
    args = parser.parse_args()
    if args.repair_existing:
        repair_existing_delivery(args.output, torch.device(args.device), args.batch_size)
        print("修正完成：六张图与翻转列已更新，未重跑全量干预", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    groups = pd.read_csv(GROUP_ROOT / "groups.csv").set_index("group")
    observations, lookup = build_feedback_tables(groups)
    observations.to_csv(args.output / "医学反馈原始观察.csv", index=False, encoding="utf-8-sig")
    lookup.to_csv(args.output / "医学反馈组号成员核对.csv", index=False, encoding="utf-8-sig")
    definition = {
        "model": "fixed_RA_duration100_K256_width2560",
        "training_changed": False, "grouping_changed": False,
        "cohorts": "full available train 2350 images/1212 patients and val 497/260",
        "test_read": False, "external_read": False,
        "medical_feedback_scope": "partial-image preliminary observations; representative item only; no group-member semantic expansion",
        "original_counterexample_images_located": False,
        "review_selection": {
            "group_high": "max member peak/Q99 within each train/val x cancer/noncancer stratum",
            "member_dominance": "member peak/Q99 minus the other member peak/Q99, one per member per split",
            "patient_deduplication": "strict within group; insufficient candidates are not backfilled",
        },
        "interventions": INTERVENTIONS,
        "intervention_selection_limit": "four candidate direction-groups, not representatives of all feedback-category Features",
        "group_removal": "F_prime = F - sum_j_in_group(h_j*d_j), followed by recomputed attention, pooling and original classifier",
        "heatmap": {"scale": "fixed train positive Q99 per feature", "alpha": HEAT_ALPHA,
                    "contour_threshold_q99": CONTOUR_THRESHOLD,
                    "contour_label": "high activation region, not lesion boundary"},
        "auc": "combined cancer/noncancer patient cohort after mean image probability per patient",
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frames = {
        split: pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(columns={"index": "source_row"})
        for split in ("train", "val")
    }
    if set(frames["train"].patient_id) & set(frames["val"].patient_id):
        raise RuntimeError("train/val患者交叉")
    peaks = {split: np.load(GROUP_ROOT / f"{split}_peak_q99.npy", mmap_mode="r")
             for split in ("train", "val")}
    selected = choose_review_images(frames, peaks, groups)
    selected.to_csv(args.output / "重点混杂组选例.csv", index=False, encoding="utf-8-sig")
    device = torch.device(args.device)
    sae, head, classifier_weight, classifier_bias = load_models(device)
    decoder = sae.decoder_weight
    q99 = np.load(BASE / "decision_alignment/ra_train_q99.npy")
    needed_map_rows = {(row.split, int(row.source_row)) for row in selected.itertuples()}
    selected_maps: dict[tuple[str, int], np.ndarray] = {}
    image_rows = []
    intervention_members = {
        category: parse_members(groups.loc[gid, "members"])
        for category, gid in INTERVENTIONS.items()
    }
    with torch.no_grad():
        for split, frame in frames.items():
            for start in range(0, len(frame), args.batch_size):
                stop = min(start + args.batch_size, len(frame))
                batch_frame = frame.iloc[start:stop]
                data = read_subset(batch_frame, split, device)
                spatial = data["spatial"]
                hidden = sae.encode(spatial, K)
                original_attention = attention_from_features(spatial, head)
                original_pool = pooled_from_features(spatial, original_attention)
                original_logits = original_pool @ classifier_weight.T + classifier_bias
                original_margin = original_logits[:, 1] - original_logits[:, 0]
                original_probability = original_logits.softmax(1)[:, 1]
                expected = torch.as_tensor(
                    batch_frame.cancer_probability.to_numpy(dtype=np.float32, copy=True), device=device
                )
                torch.testing.assert_close(original_probability, expected, atol=1e-4, rtol=1e-4)
                for local, source_row in enumerate(range(start, stop)):
                    if (split, source_row) in needed_map_rows:
                        selected_maps[(split, source_row)] = hidden[local].cpu().numpy()
                for category, gid in INTERVENTIONS.items():
                    members = intervention_members[category]
                    changed = remove_feature_group(spatial, hidden, decoder, members)
                    changed_attention = attention_from_features(changed, head)
                    changed_pool = pooled_from_features(changed, changed_attention)
                    changed_logits = changed_pool @ classifier_weight.T + classifier_bias
                    changed_margin = changed_logits[:, 1] - changed_logits[:, 0]
                    changed_probability = changed_logits.softmax(1)[:, 1]
                    attention_cosine = torch.nn.functional.cosine_similarity(
                        original_attention, changed_attention, dim=1
                    )
                    group_peaks = torch.as_tensor(
                        np.asarray(peaks[split][start:stop, members]).copy(), device=device
                    ).max(1).values
                    active = hidden[:, :, members].gt(1e-8).any((1, 2))
                    for local, record in enumerate(batch_frame.itertuples()):
                        image_rows.append({
                            "split": split, "group": gid, "feedback_category": category,
                            "source_row": int(record.source_row), "patient_id": str(record.patient_id),
                            "label": int(record.label), "group_members": ",".join(map(str, members)),
                            "group_peak_q99": float(group_peaks[local]), "active": bool(active[local]),
                            "original_margin": float(original_margin[local]),
                            "ablated_margin": float(changed_margin[local]),
                            "delta_margin": float(changed_margin[local] - original_margin[local]),
                            "original_probability": float(original_probability[local]),
                            "ablated_probability": float(changed_probability[local]),
                            "delta_probability": float(changed_probability[local] - original_probability[local]),
                            "attention_l1": float((changed_attention[local] - original_attention[local]).abs().sum()),
                            "attention_cosine": float(attention_cosine[local]),
                            "peak_changed": bool(changed_attention[local].argmax() != original_attention[local].argmax()),
                        })
                print(f"{split}: {stop}/{len(frame)}", flush=True)
    if set(selected_maps) != needed_map_rows:
        raise RuntimeError("选例空间激活未全部捕获")
    image_effects = pd.DataFrame(image_rows)
    patients = patient_table(image_effects)
    class_metrics, overall_metrics = summarize_interventions(image_effects, patients)
    image_effects.to_csv(args.output / "四组干预_逐图结果.csv", index=False)
    patients.to_csv(args.output / "四组干预_逐患者结果.csv", index=False)
    class_metrics.to_csv(args.output / "四组干预_癌非癌分层汇总.csv", index=False)
    overall_metrics.to_csv(args.output / "四组干预_合并队列AUC与翻转.csv", index=False)
    render_review_figures(selected, groups, selected_maps, q99, args.output / "重点混杂组并排图")
    verification = {
        "feedback_unique_groups": int(observations.group.nunique()),
        "feedback_total_members": int(lookup["size"].sum()),
        "G2313_preserved": bool(lookup.loc[lookup.group.eq("G2313"), "members"].item() == "RA-F2555"),
        "priority_groups": list(PRIORITY_GROUPS),
        "review_patients_unique_within_group": True,
        "selected_review_images": int(len(selected)),
        "intervention_groups": INTERVENTIONS,
        "simultaneous_group_removal": True,
        "original_probability_cache_match_atol": 1e-4,
        "train_val_patient_overlap": 0,
        "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
