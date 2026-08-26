#!/usr/bin/env python3
"""对完整train执行RP-SAE视觉排名与功能敏感度排名数值审计。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAE_ROOT = PROJECT_ROOT / "结果/SAE"
CODE_ROOT = PROJECT_ROOT / "程序/SAE/正式代码"
PROTOCOL = CODE_ROOT / "decision_aware_audit_protocol_v1.json"
RPA_CACHE = SAE_ROOT / "RP_A_Development_20260824/analysis_cache/seed42"
RPD_SELECTION = SAE_ROOT / "RP_D_Technical_Atlas_20260825/manifest_dry_run_v1_retry1"
RPD_RENDER = SAE_ROOT / "RP_D_Technical_Atlas_20260825/render_v1_retry2"
RPC2_EFFECT = (
    SAE_ROOT
    / "RP_C2_Intervention_20260825/formal_retry1/unique_feature_facts/seed42/target_patient_effects"
)
OUTPUT_ROOT = SAE_ROOT / "Decision_Aware_Full_Numeric_Audit_v1_20260826"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TOP_K = (1, 3, 6, 10)
TOP_N = 6
NEAR_ZERO = 1e-6


def deterministic_ranks(scores: np.ndarray, anchor_ids: np.ndarray) -> np.ndarray:
    """按分数降序、Anchor ID升序返回1-based ordinal ranks。"""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != len(anchor_ids):
        raise ValueError("scores与anchor_ids shape不一致")
    ranks = np.empty(values.shape, dtype=np.int16)
    anchor_order = np.argsort(np.asarray(anchor_ids, dtype=str), kind="stable")
    anchor_priority = np.empty(len(anchor_ids), dtype=np.int32)
    anchor_priority[anchor_order] = np.arange(len(anchor_ids))
    for row_index, row in enumerate(values):
        order = np.lexsort((anchor_priority, -row))
        ranks[row_index, order] = np.arange(1, len(anchor_ids) + 1, dtype=np.int16)
    return ranks


def top_indices_from_ranks(ranks: np.ndarray, k: int) -> np.ndarray:
    """从确定性rank矩阵返回每行按rank排列的Top-k列索引。"""
    return np.argsort(ranks, axis=1, kind="stable")[:, :k]


def overlap_counts(raw_ranks: np.ndarray, functional_ranks: np.ndarray, k: int) -> np.ndarray:
    """计算每图两个固定大小Top-k集合的交集数。"""
    return ((raw_ranks <= k) & (functional_ranks <= k)).sum(axis=1)


def support_class(labels: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """按标签和signed delta生成正确方向支持、反向或严格零描述。"""
    labels = np.asarray(labels, dtype=np.int8)
    delta = np.asarray(delta, dtype=np.float64)
    result = np.full(delta.shape, "opposing", dtype=object)
    result[delta == 0] = "exact_zero"
    correct = ((labels == 1) & (delta < 0)) | ((labels == 0) & (delta > 0))
    result[correct] = "correct_label_supporting"
    return result


def load_inputs() -> dict:
    """加载并核对冻结2350图、149 Anchor、Q99和RP-C2逐图效应。"""
    required = [
        PROTOCOL,
        RPA_CACHE / "train_image_activations.npy",
        RPA_CACHE / "train_images.csv",
        RPD_SELECTION / "rpd_anchor_manifest.csv",
        RPD_RENDER / "q99_scales.csv",
        RPC2_EFFECT / "target_feature_ids.npy",
        RPC2_EFFECT / "image_delta_margin_alpha_0p00.npy",
        FONT_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("全量审计缺少冻结输入:\n" + "\n".join(missing))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_2026-08-26":
        raise RuntimeError("Decision-aware audit protocol尚未冻结")

    images = pd.read_csv(RPA_CACHE / "train_images.csv", low_memory=False)
    if len(images) != 2350:
        raise RuntimeError(f"冻结train图像数异常: {len(images)}")
    images = images.reset_index().rename(columns={"index": "image_index"})
    anchors = pd.read_csv(RPD_SELECTION / "rpd_anchor_manifest.csv")
    anchors = anchors.rename(columns={"feature_42": "feature_id"}).sort_values("anchor_id")
    q99 = pd.read_csv(RPD_RENDER / "q99_scales.csv")
    q99 = q99[q99.seed.eq(42)][["anchor_id", "feature_id", "q99_scale"]]
    anchors = anchors.merge(q99, on=["anchor_id", "feature_id"], validate="one_to_one")
    anchors = anchors.sort_values("anchor_id").reset_index(drop=True)
    if len(anchors) != 149 or anchors.q99_scale.le(0).any():
        raise RuntimeError("冻结149 Anchor或seed42 Q99不完整")
    target_ids = np.load(RPC2_EFFECT / "target_feature_ids.npy")
    if not np.array_equal(target_ids, anchors.feature_id.to_numpy(np.int64)):
        raise RuntimeError("RP-C2 target Feature顺序与Anchor顺序不一致")
    delta = np.load(RPC2_EFFECT / "image_delta_margin_alpha_0p00.npy", mmap_mode="r")
    if delta.shape != (2350, 149) or not np.isfinite(delta).all():
        raise RuntimeError("RP-C2 signed delta矩阵异常")
    return {"protocol": protocol, "images": images, "anchors": anchors, "delta": delta}


def compute_raw_scores(anchors: pd.DataFrame) -> np.ndarray:
    """分块读取空间激活并计算冻结Q99相对峰值。"""
    activation = np.load(RPA_CACHE / "train_image_activations.npy", mmap_mode="r")
    if activation.shape != (2350, 49, 10240):
        raise RuntimeError(f"空间激活shape异常: {activation.shape}")
    feature_ids = anchors.feature_id.to_numpy(np.int64)
    q99 = anchors.q99_scale.to_numpy(np.float32)
    scores = np.empty((2350, 149), dtype=np.float32)
    for start in range(0, 2350, 100):
        stop = min(start + 100, 2350)
        selected = np.asarray(activation[start:stop, :, feature_ids], dtype=np.float32)
        scores[start:stop] = selected.max(axis=1) / q99
    if not np.isfinite(scores).all():
        raise RuntimeError("raw score矩阵包含NaN/Inf")
    return scores


def build_reverse_table(
    images: pd.DataFrame,
    anchors: pd.DataFrame,
    raw_scores: np.ndarray,
    delta: np.ndarray,
    raw_ranks: np.ndarray,
    functional_ranks: np.ndarray,
    select_by: str,
) -> pd.DataFrame:
    """生成Raw Top-6→功能rank或功能Top-6→Raw rank长表。"""
    selection_ranks = raw_ranks if select_by == "raw" else functional_ranks
    selected = top_indices_from_ranks(selection_ranks, TOP_N)
    rows = []
    labels = images.label.to_numpy(np.int8)
    for image_index in range(len(images)):
        for anchor_index in selected[image_index]:
            signed = float(delta[image_index, anchor_index])
            rows.append({
                "image_index": image_index,
                "patient_id": str(images.iloc[image_index].patient_id),
                "label": int(labels[image_index]),
                "source": str(images.iloc[image_index].source),
                "anchor_id": str(anchors.iloc[anchor_index].anchor_id),
                "feature_id": int(anchors.iloc[anchor_index].feature_id),
                "source_risk": bool(anchors.iloc[anchor_index].source_risk),
                "raw_rank": int(raw_ranks[image_index, anchor_index]),
                "functional_rank": int(functional_ranks[image_index, anchor_index]),
                "rank_gap_functional_minus_raw": int(
                    functional_ranks[image_index, anchor_index]
                    - raw_ranks[image_index, anchor_index]
                ),
                "raw_relative_q99_score": float(raw_scores[image_index, anchor_index]),
                "delta_margin": signed,
                "abs_delta_margin": abs(signed),
                "net_direction": "suppresses_cancer_margin" if signed > 0 else (
                    "elevates_cancer_margin" if signed < 0 else "exact_zero"
                ),
                "label_support": str(support_class(
                    np.asarray([labels[image_index]]), np.asarray([signed])
                )[0]),
            })
    frame = pd.DataFrame(rows)
    if len(frame) != 14100:
        raise RuntimeError(f"{select_by}反向排名表行数异常: {len(frame)}")
    return frame


def build_image_summary(
    images: pd.DataFrame,
    delta: np.ndarray,
    raw_scores: np.ndarray,
    raw_ranks: np.ndarray,
    functional_ranks: np.ndarray,
) -> pd.DataFrame:
    """生成2350行逐图总体排名和方向摘要。"""
    output = images[["image_index", "patient_id", "label", "source", "relative_path"]].copy()
    for k in TOP_K:
        count = overlap_counts(raw_ranks, functional_ranks, k)
        output[f"top{k}_intersection_count"] = count
        output[f"top{k}_overlap_fraction"] = count / k
    correlations = [
        float(spearmanr(raw_scores[index], np.abs(delta[index])).statistic)
        for index in range(len(images))
    ]
    output["spearman_raw_vs_functional"] = correlations
    output["positive_delta_fraction_all149"] = (delta > 0).mean(axis=1)
    output["negative_delta_fraction_all149"] = (delta < 0).mean(axis=1)
    output["exact_zero_delta_fraction_all149"] = (delta == 0).mean(axis=1)
    output["near_zero_abs_le_1e_6_fraction_all149"] = (np.abs(delta) <= NEAR_ZERO).mean(axis=1)

    functional_top = top_indices_from_ranks(functional_ranks, TOP_N)
    selected_delta = np.take_along_axis(delta, functional_top, axis=1)
    labels = images.label.to_numpy(np.int8)[:, None]
    correct = ((labels == 1) & (selected_delta < 0)) | ((labels == 0) & (selected_delta > 0))
    output["functional_top6_correct_support_fraction"] = correct.mean(axis=1)
    output["functional_top6_opposing_fraction"] = (
        ((selected_delta != 0) & ~correct).mean(axis=1)
    )
    output["functional_top6_exact_zero_fraction"] = (selected_delta == 0).mean(axis=1)
    return output


def build_extremes(
    images: pd.DataFrame,
    anchors: pd.DataFrame,
    raw_scores: np.ndarray,
    delta: np.ndarray,
    raw_ranks: np.ndarray,
    functional_ranks: np.ndarray,
) -> pd.DataFrame:
    """按冻结rank gap输出正负各100条极端解耦记录。"""
    image_index, anchor_index = np.indices(raw_ranks.shape)
    flat = pd.DataFrame({
        "image_index": image_index.ravel(),
        "anchor_index": anchor_index.ravel(),
        "rank_gap_functional_minus_raw": (functional_ranks - raw_ranks).ravel(),
    })
    anchor_ids = anchors.anchor_id.astype(str).to_numpy()
    flat["anchor_id"] = anchor_ids[flat.anchor_index]
    positive = flat.sort_values(
        ["rank_gap_functional_minus_raw", "image_index", "anchor_id"],
        ascending=[False, True, True], kind="stable",
    ).head(100).copy()
    positive["extreme_type"] = "visual_prominent_functionally_weak"
    negative = flat.sort_values(
        ["rank_gap_functional_minus_raw", "image_index", "anchor_id"],
        ascending=[True, True, True], kind="stable",
    ).head(100).copy()
    negative["extreme_type"] = "visual_subtle_functionally_sensitive"
    result = pd.concat([positive, negative], ignore_index=True)
    rows = result.image_index.to_numpy(np.int64)
    columns = result.anchor_index.to_numpy(np.int64)
    result["patient_id"] = images.patient_id.astype(str).to_numpy()[rows]
    result["label"] = images.label.to_numpy(np.int8)[rows]
    result["source"] = images.source.astype(str).to_numpy()[rows]
    result["feature_id"] = anchors.feature_id.to_numpy(np.int64)[columns]
    result["source_risk"] = anchors.source_risk.to_numpy(bool)[columns]
    result["raw_rank"] = raw_ranks[rows, columns]
    result["functional_rank"] = functional_ranks[rows, columns]
    result["raw_relative_q99_score"] = raw_scores[rows, columns]
    result["delta_margin"] = delta[rows, columns]
    result["abs_delta_margin"] = np.abs(delta[rows, columns])
    return result.drop(columns="anchor_index")


def build_stratified_summary(
    images: pd.DataFrame,
    anchors: pd.DataFrame,
    image_summary: pd.DataFrame,
    raw_ranks: np.ndarray,
    functional_ranks: np.ndarray,
    delta: np.ndarray,
) -> pd.DataFrame:
    """汇总总体、标签以及source-risk记录层级的描述统计。"""
    rows: list[dict] = []
    for group_name, mask in [
        ("all_images", np.ones(len(images), dtype=bool)),
        ("cancer_images", images.label.to_numpy() == 1),
        ("noncancer_images", images.label.to_numpy() == 0),
    ]:
        frame = image_summary.loc[mask]
        row = {
            "summary_level": "image",
            "group": group_name,
            "n": int(mask.sum()),
            "spearman_mean": float(frame.spearman_raw_vs_functional.mean()),
            "spearman_median": float(frame.spearman_raw_vs_functional.median()),
            "functional_top6_correct_support_fraction_mean": float(
                frame.functional_top6_correct_support_fraction.mean()
            ),
            "functional_top6_opposing_fraction_mean": float(
                frame.functional_top6_opposing_fraction.mean()
            ),
        }
        for k in TOP_K:
            row[f"top{k}_overlap_fraction_mean"] = float(
                frame[f"top{k}_overlap_fraction"].mean()
            )
            row[f"top{k}_overlap_fraction_median"] = float(
                frame[f"top{k}_overlap_fraction"].median()
            )
        rows.append(row)

    risk = anchors.source_risk.to_numpy(bool)
    for label_name, image_mask in [
        ("all_labels", np.ones(len(images), dtype=bool)),
        ("cancer", images.label.to_numpy() == 1),
        ("noncancer", images.label.to_numpy() == 0),
    ]:
        for risk_name, anchor_mask in [
            ("source_risk", risk), ("non_source_risk", ~risk),
        ]:
            subset = np.ix_(image_mask, anchor_mask)
            rows.append({
                "summary_level": "image_anchor",
                "group": f"{label_name}__{risk_name}",
                "n": int(image_mask.sum() * anchor_mask.sum()),
                "raw_rank_mean": float(raw_ranks[subset].mean()),
                "functional_rank_mean": float(functional_ranks[subset].mean()),
                "absolute_rank_gap_mean": float(
                    np.abs(functional_ranks[subset] - raw_ranks[subset]).mean()
                ),
                "abs_delta_margin_mean": float(np.abs(delta[subset]).mean()),
                "raw_top6_fraction": float((raw_ranks[subset] <= TOP_N).mean()),
                "functional_top6_fraction": float((functional_ranks[subset] <= TOP_N).mean()),
            })
    return pd.DataFrame(rows)


def make_figures(
    output: Path,
    images: pd.DataFrame,
    image_summary: pd.DataFrame,
    raw_to_functional: pd.DataFrame,
    functional_to_raw: pd.DataFrame,
) -> None:
    """绘制四张冻结总体统计图，不绘制胃镜图片。"""
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False})
    figure_root = output / "figures"
    figure_root.mkdir()
    colors = {0: "#20A4B8", 1: "#F05A47"}
    labels = {0: "非癌", 1: "癌"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, label in zip(axes, (0, 1)):
        frame = image_summary[image_summary.label.eq(label)]
        values = [frame[f"top{k}_overlap_fraction"] for k in TOP_K]
        boxes = axis.boxplot(values, tick_labels=[f"Top-{k}" for k in TOP_K], patch_artist=True)
        for box in boxes["boxes"]:
            box.set_facecolor(colors[label])
            box.set_alpha(0.78)
        axis.set_title(labels[label])
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("交集数 / k")
    fig.suptitle("视觉语义排名与功能敏感度排名的Top-k重合")
    fig.tight_layout()
    fig.savefig(figure_root / "figure1_topk_overlap_by_label.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for label in (0, 1):
        values = image_summary.loc[
            image_summary.label.eq(label), "spearman_raw_vs_functional"
        ]
        axis.hist(values, bins=35, alpha=0.62, color=colors[label], label=labels[label])
    axis.set_xlabel("每图149 Anchor的Spearman相关")
    axis.set_ylabel("图像数")
    axis.set_title("视觉语义分数与功能敏感度分数的秩相关")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_root / "figure2_spearman_by_label.png", dpi=180)
    plt.close(fig)

    for number, (frame, column, title, filename) in enumerate([
        (raw_to_functional, "functional_rank", "视觉语义Top-6在功能排名中的位置",
         "figure3_raw_top6_functional_rank.png"),
        (functional_to_raw, "raw_rank", "功能敏感度Top-6在视觉语义排名中的位置",
         "figure4_functional_top6_raw_rank.png"),
    ], start=3):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for axis, risk in zip(axes, (False, True)):
            values = frame.loc[frame.source_risk.eq(risk), column]
            axis.hist(values, bins=np.arange(1, 151, 5), color="#4169E1" if not risk else "#F08A24")
            axis.set_title("非source-risk" if not risk else "source-risk")
            axis.set_xlabel("确定性排名（1最好，149最末）")
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("image-anchor记录数")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(figure_root / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    """执行冻结数值审计并写出矩阵、CSV、图和配置。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {OUTPUT_ROOT}")
    data = load_inputs()
    images = data["images"]
    anchors = data["anchors"]
    signed_delta = np.asarray(data["delta"], dtype=np.float32)
    raw_scores = compute_raw_scores(anchors)
    functional_scores = np.abs(signed_delta)
    anchor_ids = anchors.anchor_id.astype(str).to_numpy()
    raw_ranks = deterministic_ranks(raw_scores, anchor_ids)
    functional_ranks = deterministic_ranks(functional_scores, anchor_ids)

    image_summary = build_image_summary(
        images, signed_delta, raw_scores, raw_ranks, functional_ranks
    )
    raw_to_functional = build_reverse_table(
        images, anchors, raw_scores, signed_delta, raw_ranks, functional_ranks, "raw"
    )
    functional_to_raw = build_reverse_table(
        images, anchors, raw_scores, signed_delta, raw_ranks, functional_ranks, "functional"
    )
    extremes = build_extremes(
        images, anchors, raw_scores, signed_delta, raw_ranks, functional_ranks
    )
    stratified = build_stratified_summary(
        images, anchors, image_summary, raw_ranks, functional_ranks, signed_delta
    )

    OUTPUT_ROOT.mkdir(parents=True)
    np.save(OUTPUT_ROOT / "raw_score_matrix.npy", raw_scores)
    np.save(OUTPUT_ROOT / "functional_score_matrix.npy", functional_scores)
    np.save(OUTPUT_ROOT / "signed_delta_matrix.npy", signed_delta)
    np.save(OUTPUT_ROOT / "anchor_ids.npy", anchor_ids)
    image_summary.to_csv(OUTPUT_ROOT / "image_level_summary.csv", index=False, encoding="utf-8-sig")
    raw_to_functional.to_csv(
        OUTPUT_ROOT / "raw_top6_functional_ranks.csv", index=False, encoding="utf-8-sig"
    )
    functional_to_raw.to_csv(
        OUTPUT_ROOT / "functional_top6_raw_ranks.csv", index=False, encoding="utf-8-sig"
    )
    extremes.to_csv(OUTPUT_ROOT / "extreme_rank_decoupling.csv", index=False, encoding="utf-8-sig")
    stratified.to_csv(OUTPUT_ROOT / "stratified_summary.csv", index=False, encoding="utf-8-sig")
    make_figures(OUTPUT_ROOT, images, image_summary, raw_to_functional, functional_to_raw)

    config = {
        "status": "decision_aware_full_numeric_audit_v1_complete",
        "protocol": data["protocol"],
        "image_count": len(images),
        "anchor_count": len(anchors),
        "image_anchor_record_count": int(len(images) * len(anchors)),
        "raw_top6_reverse_rows": len(raw_to_functional),
        "functional_top6_reverse_rows": len(functional_to_raw),
        "extreme_rows": len(extremes),
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
