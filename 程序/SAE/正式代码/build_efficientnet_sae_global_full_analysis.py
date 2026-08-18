#!/usr/bin/env python3
"""扩展全局SAE提交版：展示全部剪枝候选并分析癌/非癌共有Feature。"""

from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_efficientnet_sae_global_clinical_package as clinical  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "结果/SAE分析/EfficientNet全局分支_0804/医学生提交版_v1"
OUTPUT = PACKAGE_ROOT / "05_全量候选分析"
ACTIVE_EPS = 1e-8
SHARED_ACTIVE_RATE = 0.20
SHARED_AUC_LOW = 0.40
SHARED_AUC_HIGH = 0.60
SHARED_MEAN_GAP = 0.20


def patient_feature_statistics(run: dict, feature_ids: np.ndarray) -> pd.DataFrame:
    """按患者最大激活统计每个Feature在癌与非癌中的共享性和标签区分度。"""
    metadata = run["metadata"][["patient_id", "label"]].copy()
    records = []
    for feature_id in feature_ids:
        frame = metadata.copy()
        frame["activation"] = run["activations"][:, int(feature_id)]
        patients = frame.groupby("patient_id", as_index=False).agg(
            label=("label", "first"), activation=("activation", "max"),
        )
        cancer = patients.loc[patients.label.eq(1), "activation"].to_numpy()
        noncancer = patients.loc[patients.label.eq(0), "activation"].to_numpy()
        auc = float(roc_auc_score(patients.label, patients.activation))
        cancer_mean, noncancer_mean = float(cancer.mean()), float(noncancer.mean())
        cancer_rate = float(np.mean(cancer > ACTIVE_EPS))
        noncancer_rate = float(np.mean(noncancer > ACTIVE_EPS))
        normalized_gap = abs(cancer_mean - noncancer_mean) / (
            abs(cancer_mean) + abs(noncancer_mean) + 1e-8
        )
        if (
            cancer_rate >= SHARED_ACTIVE_RATE
            and noncancer_rate >= SHARED_ACTIVE_RATE
            and SHARED_AUC_LOW <= auc <= SHARED_AUC_HIGH
            and normalized_gap <= SHARED_MEAN_GAP
        ):
            relationship = "癌与非癌共有"
        elif auc >= 0.65 and cancer_mean > noncancer_mean:
            relationship = "癌富集"
        elif auc <= 0.35 and noncancer_mean > cancer_mean:
            relationship = "非癌富集"
        else:
            relationship = "混合/不确定"
        records.append({
            "feature_id": int(feature_id),
            "cancer_patient_active_rate": cancer_rate,
            "noncancer_patient_active_rate": noncancer_rate,
            "cancer_patient_mean_max_activation": cancer_mean,
            "noncancer_patient_mean_max_activation": noncancer_mean,
            "feature_label_auc": auc,
            "normalized_label_mean_gap": float(normalized_gap),
            "cancer_noncancer_relationship": relationship,
        })
    return pd.DataFrame(records).set_index("feature_id")


def build_seed_table(seed: int, run: dict, stable: pd.DataFrame) -> pd.DataFrame:
    """合并剪枝候选、贡献排名、来源风险、共享性及跨seed稳定编号。"""
    summary = run["summary"].copy()
    retained = summary.loc[summary.kept_after_pruning.astype(bool)].copy()
    retained = retained.loc[retained.active_patient_count.ge(2)]
    retained["diagnostic_margin_direction"] = np.where(
        retained.cancer_margin_direction.ge(0), "癌方向", "非癌方向",
    )
    retained["label_activation_contrast"] = (
        retained.mean_patient_max_label_1 - retained.mean_patient_max_label_0
    )
    retained["source_gap_ratio"] = (
        (retained.mean_patient_max_provincial - retained.mean_patient_max_external).abs()
        / (
            retained.mean_patient_max_provincial.abs()
            + retained.mean_patient_max_external.abs()
            + 1e-8
        )
    )
    retained["source_bias_warning"] = retained.source_gap_ratio.ge(0.50)
    retained["contribution_rank_in_seed"] = retained.mean_abs_margin_contribution.rank(
        method="min", ascending=False,
    ).astype(int)
    patient_stats = patient_feature_statistics(run, retained.index.to_numpy(int))
    retained = retained.join(patient_stats)

    feature_column = f"seed{seed}_feature_id"
    stable_map = stable.set_index(feature_column).candidate_id.to_dict()
    retained["stable_pair_id"] = [stable_map.get(int(feature_id), "") for feature_id in retained.index]
    retained.insert(0, "seed", seed)
    retained.insert(1, "candidate_key", [f"S{seed}-F{int(i):05d}" for i in retained.index])
    return retained.reset_index().sort_values("contribution_rank_in_seed")


def save_plots(seed: int, table: pd.DataFrame, output: Path) -> None:
    """保存候选关系构成及贡献/来源风险分布图。"""
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = font_manager.FontProperties(fname=font_path)
    counts = table.cancer_noncancer_relationship.value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(counts.index, counts.values, color=["#c7473a", "#3f78a8", "#5a9367", "#8a8175"][:len(counts)])
    axes[0].set_title(f"seed{seed} 癌/非癌关系分组", fontproperties=font)
    axes[0].tick_params(axis="x", rotation=20)
    for label in axes[0].get_xticklabels():
        label.set_fontproperties(font)
    colors = np.where(table.diagnostic_margin_direction.eq("癌方向"), "#c7473a", "#3f78a8")
    axes[1].scatter(
        table.source_gap_ratio, table.mean_abs_margin_contribution,
        c=colors, alpha=0.72, edgecolors="none",
    )
    axes[1].axvline(0.50, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Source gap ratio")
    axes[1].set_ylabel("Mean absolute margin contribution")
    axes[1].set_title(f"seed{seed} 来源差异与分类贡献", fontproperties=font)
    fig.tight_layout()
    fig.savefig(output / f"seed{seed}_候选统计.png", dpi=180)
    plt.close(fig)


def write_html(tables: dict[int, pd.DataFrame]) -> None:
    """生成可按关系分组浏览全部候选热图的本地HTML图册。"""
    sections = []
    stable_cards = []
    for seed, table in tables.items():
        for row in table.loc[table.stable_pair_id.notna()].itertuples(index=False):
            feature_id = int(row.feature_id)
            relative = f"热图_seed{seed}_全部候选/feature_{feature_id:04d}.png"
            stable_cards.append(
                f'<article><a href="{html.escape(relative)}"><img loading="lazy" '
                f'src="{html.escape(relative)}"></a><p>{html.escape(row.candidate_key)} | '
                f'稳定对 {html.escape(str(row.stable_pair_id))}<br>'
                f'{html.escape(row.cancer_noncancer_relationship)} | '
                f'AUC={row.feature_label_auc:.3f} | 贡献排名={int(row.contribution_rank_in_seed)}</p></article>'
            )
    sections.append(
        f"<h2>优先审核：3对跨seed稳定Feature（共{len(stable_cards)}张seed内热图）</h2>"
        f'<section>{"".join(stable_cards)}</section>'
    )
    order = ("癌与非癌共有", "癌富集", "非癌富集", "混合/不确定")
    for seed, table in tables.items():
        for relationship in order:
            current = table.loc[table.cancer_noncancer_relationship.eq(relationship)]
            cards = []
            for row in current.itertuples(index=False):
                feature_id = int(row.feature_id)
                relative = f"热图_seed{seed}_全部候选/feature_{feature_id:04d}.png"
                stable = f" | 稳定对 {row.stable_pair_id}" if row.stable_pair_id else ""
                warning = " | 来源风险" if row.source_bias_warning else ""
                cards.append(
                    f'<article><a href="{html.escape(relative)}"><img loading="lazy" '
                    f'src="{html.escape(relative)}"></a><p>{html.escape(row.candidate_key)}'
                    f'{stable}{warning}<br>AUC={row.feature_label_auc:.3f} | '
                    f'贡献排名={int(row.contribution_rank_in_seed)}</p></article>'
                )
            sections.append(
                f"<h2>seed{seed} - {relationship}（{len(current)}个）</h2>"
                f'<section>{"".join(cards)}</section>'
            )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>全局SAE全量候选图册</title><style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:24px;color:#222}}
section{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
article{{border:1px solid #ccc;padding:8px;background:#fff}} img{{width:100%;height:420px;object-fit:cover;object-position:top}}
p{{font-size:13px;line-height:1.45;margin:6px 0}} h2{{margin-top:32px;border-bottom:2px solid #333;padding-bottom:6px}}
</style></head><body><h1>EfficientNet-B0 全局SAE全量候选图册</h1>
<p>点击缩略图查看完整的6位患者原图/热图。先看“癌与非癌共有”，再看癌富集、非癌富集和混合组。</p>
{''.join(sections)}</body></html>"""
    (OUTPUT / "08_全量候选图册.html").write_text(page, encoding="utf-8")


def write_readme(tables: dict[int, pd.DataFrame]) -> None:
    """写入全量材料的范围、共享Feature含义和推荐浏览顺序。"""
    lines = [
        "# 全局SAE全量候选分析说明", "",
        "本目录补充展示两个合格seed的全部剪枝候选。它不改变3对严格跨seed候选的优先级。", "",
        "## 数量", "",
    ]
    for seed, table in tables.items():
        counts = table.cancer_noncancer_relationship.value_counts().to_dict()
        lines.append(
            f"- seed{seed}：{len(table)}个；共有{counts.get('癌与非癌共有', 0)}，"
            f"癌富集{counts.get('癌富集', 0)}，非癌富集{counts.get('非癌富集', 0)}，"
            f"混合/不确定{counts.get('混合/不确定', 0)}。"
        )
    lines.extend([
        "", "## 癌与非癌共有是什么意思", "",
        "同一个SAE Feature在癌与非癌患者中都较常激活，而且单独用该Feature难以区分标签。它可能表示正常解剖结构、通用黏膜纹理、反光/器械等成像因素，也可能是两类都会出现的临床形态。共有不等于无用，需要医生结合热图判断。",
        "", "## 推荐顺序", "",
        "1. 先打开`08_全量候选图册.html`查看3对稳定候选；",
        "2. 查看“癌与非癌共有”组，识别正常结构和伪特征；",
        "3. 再看癌富集与非癌富集组；",
        "4. 在`07_临床全量审核表.csv`填写命名和风险判断。",
        "", "未剪枝的10240个Feature只用于工程追溯，不应逐个命名。seed503未通过保真门槛，因此没有生成热图。",
    ])
    (OUTPUT / "00_全量结果阅读说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """生成全量统计、共有Feature分析、热图、Excel、HTML和临床审核表。"""
    if OUTPUT.exists():
        raise FileExistsError(f"全量分析目录已存在，拒绝覆盖: {OUTPUT}")
    stable = pd.read_csv(PACKAGE_ROOT / "02_临床审核候选.csv", encoding="utf-8-sig")
    runs = {seed: clinical.load_run(seed) for seed in (42, 202)}
    tables = {seed: build_seed_table(seed, run, stable) for seed, run in runs.items()}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plot_dir = OUTPUT / "统计图"
    plot_dir.mkdir()

    for seed, table in tables.items():
        shutil.copy2(
            runs[seed]["run"] / "feature_summary.csv",
            OUTPUT / f"0{2 if seed == 42 else 3}_seed{seed}_全部10240Feature统计.csv",
        )
        table.to_csv(
            OUTPUT / f"0{4 if seed == 42 else 5}_seed{seed}_剪枝候选分析.csv",
            index=False, encoding="utf-8-sig",
        )
        save_plots(seed, table, plot_dir)

    all_candidates = pd.concat(tables.values(), ignore_index=True)
    shared = all_candidates.loc[
        all_candidates.cancer_noncancer_relationship.eq("癌与非癌共有")
    ].copy()
    shared.to_csv(OUTPUT / "06_癌与非癌共有Feature.csv", index=False, encoding="utf-8-sig")
    review = all_candidates[[
        "seed", "candidate_key", "feature_id", "stable_pair_id",
        "cancer_noncancer_relationship", "diagnostic_margin_direction",
        "feature_label_auc", "source_bias_warning", "contribution_rank_in_seed",
    ]].copy()
    for column in ("可见模式", "建议名称", "是否正常结构", "是否伪特征", "与病灶关系", "备注"):
        review[column] = ""
    review.to_csv(OUTPUT / "07_临床全量审核表.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT / "01_全量候选总表.xlsx", engine="openpyxl") as writer:
        for seed, table in tables.items():
            table.to_excel(writer, sheet_name=f"seed{seed}_剪枝候选", index=False)
        shared.to_excel(writer, sheet_name="癌非癌共有", index=False)
        stable.to_excel(writer, sheet_name="跨seed优先3对", index=False)

    for seed, run in runs.items():
        feature_ids = tables[seed].feature_id.astype(int).tolist()
        clinical.make_seed_overviews(
            seed, run, feature_ids, OUTPUT / f"热图_seed{seed}_全部候选",
        )
    write_html(tables)
    write_readme(tables)
    print(f"全量剪枝候选={len(all_candidates)}；癌与非癌共有={len(shared)}")
    print(f"输出目录: {OUTPUT}")


if __name__ == "__main__":
    main()
