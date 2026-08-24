"""复用冻结候选区域，运行 MOCE K-Means 聚类数量敏感性分析。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PROJECT_DIR = Path(__file__).resolve().parents[3]
MOCE_DIR = PROJECT_DIR / "程序" / "MOCE" / "正式代码"
sys.path.insert(0, str(MOCE_DIR))

import analyze_curated_moce_results as analyze  # noqa: E402
import moce_cluster as moce  # noqa: E402
import moce_curated_concept as curated  # noqa: E402
import render_curated_cluster_overview as render  # noqa: E402


SOURCE_ROOT = (
    PROJECT_DIR / "结果" / "MOCE聚类" / "概念严格平衡_v1" / "full"
)
RESULT_ROOT = PROJECT_DIR / "结果" / "MOCE聚类数量分析"
OUTPUT_ROOT = RESULT_ROOT / "原始结果"
ANALYSIS_ROOT = RESULT_ROOT / "自动分析"
MANIFEST_PATH = (
    PROJECT_DIR / "数据整理记录" / "概念提取训练集_v1"
    / "核心清单_v1" / "strict_pair_image_balanced_manifest.csv"
)
DATA_DIR = PROJECT_DIR / "数据" / "胃早癌概念提取训练集_裁剪后_v1_1"

MODEL_NAMES = ["resnet50", "efficientnet_b0"]
CLASS_LABELS = [0, 1]
BASELINE_K = 25
DEFAULT_MIN_K = 10
DEFAULT_MAX_K = 50
DEFAULT_STEP = 5

RAW_RESULT_FILES = [
    "cluster_assignments.csv",
    "cluster_summary.csv",
    "concept_clusters.png",
    "concept_importance.csv",
    "concept_scores_per_image.csv",
    "ssc_sdc_per_image.csv",
    "ssc_sdc_summary.csv",
    "kmeans_model.joblib",
]
REQUIRED_RAW_FILES = set(RAW_RESULT_FILES)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
PLOT_FONT = FontProperties(fname=FONT_PATH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MOCE K-Means 聚类数量敏感性分析",
    )
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], default="all")
    parser.add_argument("--class-label", choices=["all", "0", "1"], default="all")
    parser.add_argument("--cluster-min", type=int, default=DEFAULT_MIN_K)
    parser.add_argument("--cluster-max", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--cluster-step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--clusters", type=int, nargs="+", default=None,
        help="显式K值列表；提供后覆盖min/max/step",
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="仅为已经完成的K结果生成图表和跨K汇总",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="只生成聚类、重要性和SSC/SDC原始结果",
    )
    return parser.parse_args()


def selected_clusters(args):
    if args.clusters:
        values = sorted(set(args.clusters))
    else:
        if args.cluster_step <= 0:
            raise ValueError("--cluster-step必须为正整数")
        values = list(range(
            args.cluster_min, args.cluster_max + 1, args.cluster_step
        ))
    if not values or any(value <= 1 for value in values):
        raise ValueError("所有K值必须大于1")
    return values


def selected_models(args):
    return MODEL_NAMES if args.model == "all" else [args.model]


def selected_labels(args):
    return CLASS_LABELS if args.class_label == "all" else [int(args.class_label)]


def k_name(k):
    return f"k_{k:02d}"


def class_is_complete(class_dir):
    return class_dir.is_dir() and REQUIRED_RAW_FILES.issubset(
        {path.name for path in class_dir.iterdir() if path.is_file()}
    )


def load_candidate_source(source_root, model_name, label):
    class_dir = source_root / model_name / f"class_{label}"
    feature_path = class_dir / "candidate_features.npz"
    assignment_path = class_dir / "cluster_assignments.csv"
    if not feature_path.is_file() or not assignment_path.is_file():
        raise FileNotFoundError(
            f"缺少K=25冻结候选缓存: {class_dir}"
        )
    with np.load(feature_path) as cache:
        features = cache["features"].astype(np.float32)
    assignment = pd.read_csv(assignment_path, encoding="utf-8-sig")
    if len(features) != len(assignment):
        raise ValueError(
            f"候选特征与清单长度不一致: {len(features)} != {len(assignment)}"
        )
    records = assignment.drop(
        columns=["cluster_id", "distance_to_center"], errors="ignore"
    ).to_dict("records")
    return features, records, class_dir


def copy_baseline_results(source_class_dir, target_class_dir):
    target_class_dir.mkdir(parents=True, exist_ok=True)
    for filename in RAW_RESULT_FILES:
        source = source_class_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"K=25基线缺少文件: {source}")
        shutil.copy2(source, target_class_dir / filename)


def run_one_class(model, model_name, label, k, source_root, target_class_dir):
    features, records, source_class_dir = load_candidate_source(
        source_root, model_name, label
    )
    if k == BASELINE_K:
        copy_baseline_results(source_class_dir, target_class_dir)
        print(f"复用正式K=25结果: {model_name}/class_{label}")
        return

    target_class_dir.mkdir(parents=True, exist_ok=True)
    moce.N_CLUSTERS = k
    assignment, summary = moce.cluster_features(
        features,
        records,
        str(target_class_dir),
        save_feature_cache=False,
    )
    if assignment.empty or len(summary) != k:
        raise RuntimeError(
            f"K={k}聚类不完整: {model_name}/class_{label}"
        )
    moce.save_cluster_overview(
        label,
        assignment,
        str(target_class_dir / "concept_clusters.png"),
    )
    importance = moce.evaluate_concept_importance(
        model, label, assignment, str(target_class_dir)
    )
    moce.evaluate_ssc_sdc(
        model, label, assignment, importance, str(target_class_dir)
    )
    print(
        f"完成K={k}: {model_name}/class_{label}, "
        f"候选区域={len(features)}"
    )


def write_run_config(target_model_dir, args, model_name, k, labels):
    source_config_path = args.source_root / model_name / "run_config.json"
    source_config = {}
    if source_config_path.is_file():
        with open(source_config_path, encoding="utf-8") as file:
            source_config = json.load(file)
    config = {
        "analysis_type": "kmeans_cluster_count_sensitivity",
        "model": model_name,
        "clusters": k,
        "class_labels": labels,
        "random_seed": moce.RANDOM_SEED,
        "candidate_cache_source": str(
            args.source_root.resolve() / model_name
        ),
        "manifest_path": str(MANIFEST_PATH),
        "data_dir": str(DATA_DIR),
        "baseline_k": BASELINE_K,
        "baseline_run_config": source_config,
        "candidate_extraction_reused": True,
        "candidate_masks_reused": True,
        "kmeans_refit": k != BASELINE_K,
    }
    with open(
        target_model_dir / "run_config.json", "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def generate_raw_results(args, clusters, models, labels):
    moce.DATA_DIR = str(DATA_DIR)
    for model_name in models:
        needs_model = any(k != BASELINE_K for k in clusters)
        model = None if args.analysis_only or not needs_model else (
            curated.load_debiased_model(model_name)
        )
        moce.MODEL_NAME = model_name
        for k in clusters:
            target_model_dir = args.output_root / k_name(k) / model_name
            target_model_dir.mkdir(parents=True, exist_ok=True)
            for label in labels:
                target_class_dir = target_model_dir / f"class_{label}"
                if class_is_complete(target_class_dir):
                    print(f"跳过已完成: K={k} {model_name}/class_{label}")
                    continue
                if target_class_dir.exists():
                    raise FileExistsError(
                        f"发现残缺输出目录，拒绝自动删除或覆盖: {target_class_dir}"
                    )
                if args.analysis_only:
                    raise FileNotFoundError(
                        f"analysis-only缺少完整结果: {target_class_dir}"
                    )
                if target_class_dir.exists():
                    raise RuntimeError(
                        f"发现不完整输出: {target_class_dir}；"
                        "请人工核对并归档残缺目录后重跑"
                    )
                run_one_class(
                    model,
                    model_name,
                    label,
                    k,
                    args.source_root,
                    target_class_dir,
                )
            write_run_config(target_model_dir, args, model_name, k, labels)


def generate_analysis(args, clusters, models, labels):
    manifest = pd.read_csv(
        MANIFEST_PATH, encoding="utf-8-sig", low_memory=False
    ).rename(columns={
        "processed_relpath": "image_path",
        "source": "source_path",
    })
    for k in clusters:
        result_dir = args.output_root / k_name(k)
        analysis_dir = args.analysis_root / k_name(k)
        render.RESULT_DIR = str(result_dir)
        analyze.MOCE_DIR = str(result_dir)
        analyze.ANALYSIS_DIR = str(analysis_dir)
        for model_name in models:
            for label in labels:
                render.render_class(model_name, label)
                analyze.analyze_class(model_name, label, manifest)


def load_sensitivity_row(args, k, model_name, label):
    class_dir = args.output_root / k_name(k) / model_name / f"class_{label}"
    analysis_dir = (
        args.analysis_root / k_name(k) / model_name / f"class_{label}"
        / "01_自动分析结果"
    )
    assignment = pd.read_csv(
        class_dir / "cluster_assignments.csv", encoding="utf-8-sig"
    )
    summary = pd.read_csv(
        class_dir / "cluster_summary.csv", encoding="utf-8-sig"
    )
    importance = pd.read_csv(
        class_dir / "concept_importance.csv", encoding="utf-8-sig"
    ).sort_values("importance_rank")
    ssc_sdc = pd.read_csv(
        class_dir / "ssc_sdc_summary.csv", encoding="utf-8-sig"
    ).sort_values("step")
    dominance = pd.read_csv(
        analysis_dir / "cluster_patient_dominance.csv",
        encoding="utf-8-sig",
    )
    model = joblib.load(class_dir / "kmeans_model.joblib")
    return {
        "model": model_name,
        "class_label": label,
        "k": k,
        "candidate_count": len(assignment),
        "image_count": assignment["image_name"].nunique(),
        "patient_count": assignment["patient_id"].nunique(),
        "inertia_per_candidate": float(model.inertia_ / len(assignment)),
        "mean_distance_to_center": float(
            assignment["distance_to_center"].mean()
        ),
        "p90_distance_to_center": float(
            assignment["distance_to_center"].quantile(0.90)
        ),
        "median_cluster_candidates": float(summary["candidate_count"].median()),
        "min_cluster_candidates": int(summary["candidate_count"].min()),
        "median_cluster_patients": float(summary["patient_count"].median()),
        "min_cluster_patients": int(summary["patient_count"].min()),
        "clusters_under_5_patients": int((summary["patient_count"] < 5).sum()),
        "clusters_under_10_patients": int((summary["patient_count"] < 10).sum()),
        "mean_max_patient_share": float(
            dominance["max_patient_share"].mean()
        ),
        "worst_max_patient_share": float(
            dominance["max_patient_share"].max()
        ),
        "top1_importance_S_h": float(importance.iloc[0]["S_h"]),
        "top5_mean_probability_drop": float(
            importance.head(5)["mean_probability_drop"].mean()
        ),
        "ssc_step0_accuracy": float(ssc_sdc.iloc[0]["ssc_accuracy"]),
        "ssc_final_accuracy": float(ssc_sdc.iloc[-1]["ssc_accuracy"]),
        "sdc_step0_accuracy": float(ssc_sdc.iloc[0]["sdc_accuracy"]),
        "sdc_final_accuracy": float(ssc_sdc.iloc[-1]["sdc_accuracy"]),
    }, assignment[["patch_file", "cluster_id"]].copy()


def add_adjacent_stability(summary, assignments):
    summary = summary.sort_values(["model", "class_label", "k"]).copy()
    summary["previous_k"] = np.nan
    summary["adjusted_rand_vs_previous_k"] = np.nan
    summary["normalized_mutual_info_vs_previous_k"] = np.nan
    for (model_name, label), indexes in summary.groupby(
        ["model", "class_label"], sort=False
    ).groups.items():
        ordered = summary.loc[indexes].sort_values("k")
        previous_k = None
        previous = None
        for index, row in ordered.iterrows():
            current = assignments[(model_name, label, int(row["k"]))]
            if previous is not None:
                merged = previous.merge(
                    current,
                    on="patch_file",
                    suffixes=("_previous", "_current"),
                    validate="one_to_one",
                )
                summary.at[index, "previous_k"] = previous_k
                summary.at[index, "adjusted_rand_vs_previous_k"] = (
                    adjusted_rand_score(
                        merged["cluster_id_previous"],
                        merged["cluster_id_current"],
                    )
                )
                summary.at[index, "normalized_mutual_info_vs_previous_k"] = (
                    normalized_mutual_info_score(
                        merged["cluster_id_previous"],
                        merged["cluster_id_current"],
                    )
                )
            previous_k = int(row["k"])
            previous = current
    return summary


def save_summary_plots(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("inertia_per_candidate", "每候选区域聚类惯性"),
        ("median_cluster_patients", "每簇患者数中位数"),
        ("clusters_under_10_patients", "少于10位患者的簇数"),
        ("worst_max_patient_share", "最严重单患者主导比例"),
        ("adjusted_rand_vs_previous_k", "与前一K的ARI"),
        ("top5_mean_probability_drop", "前5概念平均概率下降"),
    ]
    for (model_name, label), group in summary.groupby(
        ["model", "class_label"], sort=True
    ):
        group = group.sort_values("k")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for axis, (column, title) in zip(axes.flat, panels):
            axis.plot(group["k"], group[column], marker="o")
            axis.set_title(title, fontproperties=PLOT_FONT)
            axis.set_xlabel("K-Means聚类数量K", fontproperties=PLOT_FONT)
            axis.grid(alpha=0.3)
        fig.suptitle(
            f"{model_name} 类别{label} K敏感性汇总",
            fontproperties=PLOT_FONT,
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{model_name}_class_{label}_K敏感性汇总.png",
            dpi=220,
        )
        plt.close(fig)


def discover_completed_results(args):
    completed = []
    if not args.output_root.is_dir():
        return completed
    for k_dir in sorted(args.output_root.glob("k_*")):
        try:
            k = int(k_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for model_name in MODEL_NAMES:
            for label in CLASS_LABELS:
                class_dir = k_dir / model_name / f"class_{label}"
                analysis_dir = (
                    args.analysis_root / k_dir.name / model_name
                    / f"class_{label}" / "01_自动分析结果"
                )
                if (
                    class_is_complete(class_dir)
                    and (analysis_dir / "cluster_patient_dominance.csv").is_file()
                ):
                    completed.append((k, model_name, label))
    return completed


def summarize_sensitivity(args):
    completed = discover_completed_results(args)
    if not completed:
        raise RuntimeError("没有可汇总的完整K敏感性分析结果")
    records = []
    assignments = {}
    for k, model_name, label in completed:
        record, assignment = load_sensitivity_row(
            args, k, model_name, label
        )
        records.append(record)
        assignments[(model_name, label, k)] = assignment
    summary = add_adjacent_stability(pd.DataFrame(records), assignments)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        args.analysis_root / "K敏感性汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(
        args.analysis_root / "K敏感性汇总.xlsx", engine="openpyxl"
    ) as writer:
        summary.to_excel(writer, sheet_name="全部结果", index=False)
        for model_name in sorted(summary["model"].unique()):
            for label in sorted(summary.loc[
                summary["model"] == model_name, "class_label"
            ].unique()):
                subset = summary[
                    (summary["model"] == model_name)
                    & (summary["class_label"] == label)
                ]
                subset.to_excel(
                    writer,
                    sheet_name=f"{model_name[:12]}_c{label}",
                    index=False,
                )
    save_summary_plots(summary, args.analysis_root)
    print(f"跨K汇总: {args.analysis_root}")


def main():
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.analysis_root = args.analysis_root.resolve()
    clusters = selected_clusters(args)
    models = selected_models(args)
    labels = selected_labels(args)
    print(f"K值: {clusters}")
    print(f"模型: {models}; 类别: {labels}")
    generate_raw_results(args, clusters, models, labels)
    if not args.skip_analysis:
        generate_analysis(args, clusters, models, labels)
        summarize_sensitivity(args)


if __name__ == "__main__":
    main()
