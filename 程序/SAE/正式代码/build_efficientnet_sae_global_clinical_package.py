#!/usr/bin/env python3
"""从通过冻结门槛的全局SAE中匹配稳定Feature并生成医学生探索材料。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from efficientnet_sae_discovery import (
    ActivationCapture,
    TopKSparseAutoencoder,
    load_model,
    make_overviews,
    product_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804"
RUNS = {
    42: RESULT_ROOT / (
        "SAE-A1c_Margin保真/"
        "a1c_global_efficientnet_b0_seed42_h10240_topk1024_margin01"
    ),
    202: RESULT_ROOT / (
        "SAE-C全局多种子复现/"
        "sae_c_global_efficientnet_b0_seed202_h10240_topk1024_margin01"
    ),
    503: RESULT_ROOT / (
        "SAE-C全局多种子复现/"
        "sae_c_global_efficientnet_b0_seed503_h10240_topk1024_margin01"
    ),
}
REPLICATION_SUMMARY = RESULT_ROOT / "SAE-C全局多种子复现/sae_c_global_replication_summary.csv"
OUTPUT = PROJECT_ROOT / "结果/SAE分析/EfficientNet全局分支_0804/医学生提交版_v1"
MIN_ACTIVE_PATIENTS = 20
MIN_SPEARMAN = 0.50
PER_DIRECTION = 15


def load_run(seed: int) -> dict:
    """加载一个正式run的开发集元数据、激活、Feature统计与门槛状态。"""
    run = RUNS[seed]
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    parts, activations = [], []
    for split in ("train", "val"):
        parts.append(pd.read_csv(run / f"特征缓存/{split}_metadata.csv", encoding="utf-8-sig"))
        projection = np.load(run / f"特征缓存/{split}_sae_projection.npz")
        activations.append(projection["activations"])
    metadata = pd.concat(parts, ignore_index=True)
    if metadata.sha256.duplicated().any():
        raise ValueError(f"seed{seed}开发集sha256不唯一，无法稳定对齐图像")
    summary = pd.read_csv(run / "feature_summary.csv", encoding="utf-8-sig").set_index("feature_id")
    pruning = metrics["pruning"]
    passed = bool(
        metrics["core_fidelity_gate"]["passed"]
        and pruning["dead_feature_rate_train"] <= 0.10
        and pruning["decoder_duplicate_feature_rate_abs_cosine_ge_0p95"] <= 0.10
    )
    return {
        "run": run, "config": config, "metrics": metrics, "metadata": metadata,
        "activations": np.concatenate(activations), "summary": summary, "passed": passed,
    }


def rank_standardize(values: np.ndarray) -> np.ndarray:
    """对每个Feature的跨图像激活做秩变换和标准化，用于Spearman相关。"""
    ranks = pd.DataFrame(values).rank(axis=0, method="average").to_numpy(np.float32).copy()
    ranks -= ranks.mean(axis=0, keepdims=True)
    ranks /= np.maximum(ranks.std(axis=0, keepdims=True), 1e-8)
    return ranks


def candidate_ids(run: dict) -> np.ndarray:
    """返回剪枝保留且覆盖足够患者的分类相关候选Feature。"""
    summary = run["summary"]
    mask = summary.kept_after_pruning.astype(bool) & summary.active_patient_count.ge(MIN_ACTIVE_PATIENTS)
    return summary.index[mask].to_numpy(int)


def match_features(reference: dict, replicate: dict) -> pd.DataFrame:
    """在共同图像上以双向最近邻Spearman相关匹配两个seed的候选Feature。"""
    common = sorted(set(reference["metadata"].sha256) & set(replicate["metadata"].sha256))
    if len(common) < 100:
        raise ValueError("两个seed共同开发图像不足，不能进行稳定Feature匹配")
    ref_pos = reference["metadata"].reset_index().set_index("sha256").loc[common, "index"].to_numpy()
    rep_pos = replicate["metadata"].reset_index().set_index("sha256").loc[common, "index"].to_numpy()
    ref_ids, rep_ids = candidate_ids(reference), candidate_ids(replicate)
    ref_rank = rank_standardize(reference["activations"][ref_pos][:, ref_ids])
    rep_rank = rank_standardize(replicate["activations"][rep_pos][:, rep_ids])
    correlation = ref_rank.T @ rep_rank / len(common)

    ref_summary, rep_summary = reference["summary"], replicate["summary"]
    ref_margin = ref_summary.loc[ref_ids, "cancer_margin_direction"].to_numpy()
    rep_margin = rep_summary.loc[rep_ids, "cancer_margin_direction"].to_numpy()
    ref_contrast = (
        ref_summary.loc[ref_ids, "mean_patient_max_label_1"].to_numpy()
        - ref_summary.loc[ref_ids, "mean_patient_max_label_0"].to_numpy()
    )
    rep_contrast = (
        rep_summary.loc[rep_ids, "mean_patient_max_label_1"].to_numpy()
        - rep_summary.loc[rep_ids, "mean_patient_max_label_0"].to_numpy()
    )
    compatible = (ref_margin[:, None] * rep_margin[None, :] > 0) & (
        ref_contrast[:, None] * rep_contrast[None, :] > 0
    )
    masked = np.where(compatible, correlation, -np.inf)
    ref_best, rep_best = masked.argmax(1), masked.argmax(0)

    ref_contribution = ref_summary.loc[ref_ids, "mean_abs_margin_contribution"].rank(pct=True).to_numpy()
    rep_contribution = rep_summary.loc[rep_ids, "mean_abs_margin_contribution"].rank(pct=True).to_numpy()
    rows = []
    for ref_index, rep_index in enumerate(ref_best):
        corr = float(masked[ref_index, rep_index])
        if rep_best[rep_index] != ref_index or corr < MIN_SPEARMAN:
            continue
        ref_id, rep_id = int(ref_ids[ref_index]), int(rep_ids[rep_index])
        ref_row, rep_row = ref_summary.loc[ref_id], rep_summary.loc[rep_id]
        source_gap_ref = abs(ref_row.mean_patient_max_provincial - ref_row.mean_patient_max_external) / (
            abs(ref_row.mean_patient_max_provincial) + abs(ref_row.mean_patient_max_external) + 1e-8
        )
        source_gap_rep = abs(rep_row.mean_patient_max_provincial - rep_row.mean_patient_max_external) / (
            abs(rep_row.mean_patient_max_provincial) + abs(rep_row.mean_patient_max_external) + 1e-8
        )
        rows.append({
            "seed42_feature_id": ref_id,
            "seed202_feature_id": rep_id,
            "spearman_activation": corr,
            "diagnostic_direction": "癌方向" if ref_row.cancer_margin_direction > 0 else "非癌方向",
            "seed42_margin_direction": ref_row.cancer_margin_direction,
            "seed202_margin_direction": rep_row.cancer_margin_direction,
            "seed42_label_contrast": ref_row.mean_patient_max_label_1 - ref_row.mean_patient_max_label_0,
            "seed202_label_contrast": rep_row.mean_patient_max_label_1 - rep_row.mean_patient_max_label_0,
            "seed42_active_patients": int(ref_row.active_patient_count),
            "seed202_active_patients": int(rep_row.active_patient_count),
            "seed42_source_gap_ratio": float(source_gap_ref),
            "seed202_source_gap_ratio": float(source_gap_rep),
            "source_bias_warning": bool(max(source_gap_ref, source_gap_rep) >= 0.50),
            "stability_score": float(corr * np.sqrt(
                ref_contribution[ref_index] * rep_contribution[rep_index]
            )),
        })
    return pd.DataFrame(rows).sort_values("stability_score", ascending=False).reset_index(drop=True)


def choose_shortlist(matches: pd.DataFrame) -> pd.DataFrame:
    """分别保留癌方向和非癌方向前若干对，防止候选被单一方向占满。"""
    selected = []
    for direction in ("癌方向", "非癌方向"):
        selected.append(matches.loc[matches.diagnostic_direction.eq(direction)].head(PER_DIRECTION))
    shortlist = pd.concat(selected, ignore_index=True).sort_values(
        ["diagnostic_direction", "stability_score"], ascending=[True, False],
    )
    shortlist.insert(0, "candidate_id", [f"G{i:02d}" for i in range(1, len(shortlist) + 1)])
    return shortlist


def load_sae(run: dict, device: torch.device) -> TopKSparseAutoencoder:
    """从正式checkpoint恢复Top-K SAE，仅用于生成固定Feature空间响应图。"""
    checkpoint = torch.load(run["run"] / "SAE模型/sae_best.pth", map_location="cpu", weights_only=False)
    sae = TopKSparseAutoencoder(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]),
        torch.zeros(int(checkpoint["input_dim"])), int(checkpoint["top_k"]),
    )
    sae.load_state_dict(checkpoint["sae_state_dict"], strict=True)
    return sae.to(device).eval()


def make_seed_overviews(seed: int, run: dict, feature_ids: list[int], output: Path) -> None:
    """为一个seed的入选Feature生成原图/响应热图并排概览。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("global", product_paths("global", seed), device)
    capture = ActivationCapture(model.features[8])
    sae = load_sae(run, device)
    summary = run["summary"].loc[feature_ids].reset_index().copy()
    summary["kept_after_pruning"] = True
    args = SimpleNamespace(overview_features=len(feature_ids), top_images=6)
    output.mkdir(parents=True, exist_ok=False)
    make_overviews(
        model, capture, sae, run["activations"], run["metadata"], summary,
        "global", args, device, output,
    )
    capture.close()


def write_readme(shortlist: pd.DataFrame, matches: pd.DataFrame) -> None:
    """写入面向医学生的简明阅读说明和临床命名边界。"""
    counts = shortlist.diagnostic_direction.value_counts().to_dict()
    text = f"""# EfficientNet-B0 全局 SAE 探索结果阅读说明

## 这份材料是什么

本目录解释的是完整胃镜图分类器的全局 1280 维特征。SAE把这些黑箱特征拆成较稀疏的候选
Feature，再寻找在两个独立随机种子中对同一批图像呈现相似激活模式的候选。

固定配置在三个种子中有2个通过全部保真门槛，seed503因recovered CE不足未纳入概念筛选。
因此本结果是“全局分支探索性概念分析”，不是已经确认的医学概念，也不是局部病灶分支的
完整解释。

## 怎么看热图

每张概览左侧是模型实际输入的完整胃镜图，右侧是该SAE Feature在末层7×7特征图上的空间
响应。红色表示该Feature在相应位置响应更强，蓝色表示更弱。它回答的是“这个Feature主要在
哪里出现”，不等同于最终分类器全部注意力，也不等同于Grad-CAM。

## 当前候选

- 跨seed双向稳定匹配共 {len(matches)} 对；
- 提交临床审核 {len(shortlist)} 对，其中癌方向 {counts.get('癌方向', 0)} 对、非癌方向
  {counts.get('非癌方向', 0)} 对；
- `source_bias_warning=True`表示该Feature在省人民/外院之间激活差异较大，需要优先排查设备、
  画幅、文字、黑边等伪特征；
- 两个seed的Feature编号不同，只有同一`candidate_id`才表示自动匹配的一对候选。

## 建议标注

请在`临床命名表.csv`中填写：是否存在一致可见模式、建议名称、是否可能为伪特征、与病灶
关系及备注。看不出稳定语义时请直接标“不确定”，不要为了凑概念强行命名。

## 不能得出的结论

剪枝候选只保留分类相关信息，不能完整重构模型看到的全部视觉内容；当前材料也不能证明某个
Feature就是癌的因果征象。后续需要结合医生标注、来源偏倚审计，以及锁定后在内部test/外部集
上的描述性投影进一步确认。
"""
    (OUTPUT / "00_阅读说明.md").write_text(text, encoding="utf-8")


def main() -> None:
    """执行门槛核验、跨seed匹配、候选筛选、热图生成和提交表整理。"""
    if OUTPUT.exists():
        raise FileExistsError(f"提交目录已存在，拒绝覆盖: {OUTPUT}")
    runs = {seed: load_run(seed) for seed in (42, 202, 503)}
    passed_seeds = [seed for seed, run in runs.items() if run["passed"]]
    if passed_seeds != [42, 202]:
        raise RuntimeError(f"本版冻结预期通过seed=[42,202]，实际={passed_seeds}")
    matches = match_features(runs[42], runs[202])
    if matches.empty:
        raise RuntimeError("没有Feature通过冻结的跨seed匹配门槛")
    shortlist = choose_shortlist(matches)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    matches.to_csv(OUTPUT / "01_全部跨seed稳定匹配.csv", index=False, encoding="utf-8-sig")
    shortlist.to_csv(OUTPUT / "02_临床审核候选.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(REPLICATION_SUMMARY, OUTPUT / "03_三种子SAE保真结果.csv")

    annotation = shortlist[[
        "candidate_id", "diagnostic_direction", "seed42_feature_id", "seed202_feature_id",
        "spearman_activation", "source_bias_warning",
    ]].copy()
    for column in ("是否有一致可见模式", "建议概念名称", "是否可能为伪特征", "与病灶关系", "备注"):
        annotation[column] = ""
    annotation.to_csv(OUTPUT / "04_临床命名表.csv", index=False, encoding="utf-8-sig")

    make_seed_overviews(
        42, runs[42], shortlist.seed42_feature_id.astype(int).tolist(), OUTPUT / "热图_seed42",
    )
    make_seed_overviews(
        202, runs[202], shortlist.seed202_feature_id.astype(int).tolist(), OUTPUT / "热图_seed202",
    )
    write_readme(shortlist, matches)
    config = {
        "passed_seeds": passed_seeds,
        "excluded_seed": 503,
        "minimum_active_patients": MIN_ACTIVE_PATIENTS,
        "minimum_spearman_activation": MIN_SPEARMAN,
        "matching": "common-development-images Spearman + compatible direction + mutual nearest neighbor",
        "per_direction_limit": PER_DIRECTION,
        "matched_pairs": len(matches),
        "shortlisted_pairs": len(shortlist),
        "test_evaluated": False,
        "external_evaluated": False,
    }
    (OUTPUT / "package_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"稳定匹配={len(matches)}对；临床候选={len(shortlist)}对")
    print(f"输出目录: {OUTPUT}")


if __name__ == "__main__":
    main()
