"""用正向概率下降重新评估S_R，不覆盖已有MOCE正式结果。"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.insert(0, BASE_DIR)

import moce_cluster as moce


SOURCE_DIR = os.path.join(PROJECT_DIR, "结果", "MOCE聚类", "第二批")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "结果", "MOCE_S_R修正验证")
MODEL_NAMES = ["resnet50", "efficientnet_b0"]


def positive_relative_ranks(values):
    """只给正向移除效应分配排名，非正效应排名为0。"""
    ranks = np.zeros(len(values), dtype=np.float32)
    for position, index in enumerate(np.argsort(values)):
        if values[index] <= 0:
            continue
        ranks[index] = position / len(values)
    return ranks


def evaluate_corrected_importance(model, label, assignment, output_dir):
    """按正向下降量归一化S_R，并重新计算S_h。"""
    per_image_records = []
    total_images = assignment["image_name"].nunique()

    for order, (image_name, image_assignment) in enumerate(
        assignment.groupby("image_name", sort=False), 1
    ):
        original = np.asarray(
            Image.open(os.path.join(moce.DATA_DIR, image_name)).convert("RGB")
        )
        cluster_masks = moce.select_cluster_masks(
            image_assignment, original.shape[:2]
        )
        cluster_ids = sorted(cluster_masks)
        keep_tensors = []
        removed_tensors = []

        for cluster_id in cluster_ids:
            mask = cluster_masks[cluster_id]
            concept, _ = moce.crop_and_resize(original, mask.astype(np.uint8))
            removed = original.copy()
            removed[mask] = 0
            keep_tensors.append(moce.CONCEPT_TRANSFORM(concept))
            removed_tensors.append(
                moce.MODEL_TRANSFORM(Image.fromarray(removed))
            )

        keep_probabilities = moce.predict_target_probabilities(
            model, keep_tensors, label
        )
        removed_probabilities = moce.predict_target_probabilities(
            model, removed_tensors, label
        )
        full_probability = float(image_assignment.iloc[0]["class_probability"])
        removal_drops = full_probability - removed_probabilities

        positive_drops = np.clip(removal_drops, 0, None)
        positive_total = positive_drops.sum()
        removal_scores = (
            np.zeros_like(positive_drops)
            if np.isclose(positive_total, 0)
            else positive_drops / positive_total
        )
        extraction_scores = keep_probabilities / keep_probabilities.sum()

        image_scores = pd.DataFrame({
            "cluster_id": cluster_ids,
            "keep_probability": keep_probabilities,
            "removed_probability": removed_probabilities,
            "probability_drop": removal_drops,
            "S_R": removal_scores,
            "S_E": extraction_scores,
        })
        image_scores["rank_R"] = positive_relative_ranks(
            image_scores["S_R"].to_numpy()
        )
        image_scores["rank_E"] = moce.relative_ranks(
            image_scores["S_E"].to_numpy()
        )
        image_scores["weighted_rank"] = (
            moce.ALPHA * image_scores["rank_R"]
            + moce.BETA * image_scores["rank_E"]
        )

        patient_id = image_assignment.iloc[0]["patient_id"]
        for row in image_scores.itertuples(index=False):
            per_image_records.append({
                "image_name": image_name,
                "patient_id": patient_id,
                "class_label": label,
                "cluster_id": row.cluster_id,
                "candidate_count": int(
                    (image_assignment["cluster_id"] == row.cluster_id).sum()
                ),
                "full_probability": full_probability,
                "keep_probability": row.keep_probability,
                "removed_probability": row.removed_probability,
                "probability_drop": row.probability_drop,
                "S_R": row.S_R,
                "S_E": row.S_E,
                "rank_R": row.rank_R,
                "rank_E": row.rank_E,
                "weighted_rank": row.weighted_rank,
            })

        if order % 100 == 0 or order == total_images:
            print(f"类别{label} 重要性重算：{order}/{total_images}")

    per_image = pd.DataFrame(per_image_records)
    importance = per_image.groupby("cluster_id").agg(
        image_count=("image_name", "nunique"),
        patient_count=("patient_id", "nunique"),
        mean_S_R=("S_R", "mean"),
        mean_S_E=("S_E", "mean"),
        mean_probability_drop=("probability_drop", "mean"),
        weighted_rank_sum=("weighted_rank", "sum"),
    ).reset_index()
    importance["image_coverage"] = importance["image_count"] / total_images
    importance["S_h"] = importance["weighted_rank_sum"] / total_images
    importance["importance_rank"] = importance["S_h"].rank(
        method="min", ascending=False
    ).astype(int)
    importance = importance.sort_values(
        ["importance_rank", "cluster_id"]
    ).reset_index(drop=True)

    per_image.to_csv(
        os.path.join(output_dir, "concept_scores_per_image.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    importance.to_csv(
        os.path.join(output_dir, "concept_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return per_image, importance


def save_comparison(source_dir, output_dir, corrected_scores, corrected_importance):
    """保存旧/新排名和S_R诊断对照。"""
    old_scores = pd.read_csv(
        os.path.join(source_dir, "concept_scores_per_image.csv"),
        encoding="utf-8-sig",
    )
    old_importance = pd.read_csv(
        os.path.join(source_dir, "concept_importance.csv"),
        encoding="utf-8-sig",
    )

    comparison = old_importance[[
        "cluster_id", "importance_rank", "S_h", "mean_S_R",
        "mean_probability_drop",
    ]].merge(
        corrected_importance[[
            "cluster_id", "importance_rank", "S_h", "mean_S_R",
            "mean_probability_drop",
        ]],
        on="cluster_id",
        suffixes=("_old", "_corrected"),
    )
    comparison.insert(0, "concept_number", comparison["cluster_id"] + 1)
    comparison["rank_change_corrected_minus_old"] = (
        comparison["importance_rank_corrected"]
        - comparison["importance_rank_old"]
    )
    comparison.sort_values("importance_rank_corrected").to_csv(
        os.path.join(output_dir, "importance_rank_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    old_total = old_scores.groupby("image_name")["probability_drop"].sum()
    old_mismatch = (
        ((old_scores["probability_drop"] > 0) & (old_scores["S_R"] < 0))
        | ((old_scores["probability_drop"] < 0) & (old_scores["S_R"] > 0))
    )
    corrected_mismatch = (
        ((corrected_scores["probability_drop"] > 0) & (corrected_scores["S_R"] < 0))
        | ((corrected_scores["probability_drop"] < 0) & (corrected_scores["S_R"] > 0))
    )
    diagnostics = pd.DataFrame({
        "metric": [
            "image_count",
            "images_with_nonpositive_total_drop",
            "old_sign_mismatch_records",
            "corrected_sign_mismatch_records",
            "old_top5_concept_numbers",
            "corrected_top5_concept_numbers",
        ],
        "value": [
            len(old_total),
            int((old_total <= 0).sum()),
            int(old_mismatch.sum()),
            int(corrected_mismatch.sum()),
            ",".join(
                map(str, old_importance.nsmallest(5, "importance_rank")["cluster_id"] + 1)
            ),
            ",".join(
                map(str, corrected_importance.nsmallest(5, "importance_rank")["cluster_id"] + 1)
            ),
        ],
    })
    diagnostics.to_csv(
        os.path.join(output_dir, "sr_correction_diagnostics.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_replacement_scope(model_output_dir):
    """记录验证通过后应替换或重新生成的结果。"""
    scope = pd.DataFrame([
        ("concept_scores_per_image.csv", "替换", "S_R、rank_R和weighted_rank已改变"),
        ("concept_importance.csv", "替换", "S_h和importance_rank已改变"),
        ("ssc_sdc_per_image.csv", "替换", "使用修正后的Top 5重新评估"),
        ("ssc_sdc_summary.csv", "替换", "使用修正后的Top 5重新汇总"),
        ("概念聚类清晰版/", "重新生成", "图中重要性排名和S_h需更新"),
        ("结果/MOCE分析/{model}/", "重新生成", "全部自动分析依赖修正后的重要性和SSC/SDC"),
        ("MOCE聚类最终结果分析指南_医学生版.docx", "更新", "文档包含旧Top 5和旧SSC/SDC解读"),
        ("cluster_assignments.csv", "保留", "聚类分配不受S_R影响"),
        ("cluster_summary.csv", "保留", "簇统计不受S_R影响"),
        ("candidate_features.npz", "保留", "候选特征不受S_R影响"),
        ("kmeans_model.joblib", "保留", "K-Means中心不受S_R影响"),
        ("候选区域/、候选掩码/", "保留", "无需重新提取"),
        ("concept_clusters.png", "保留", "原始总览不显示S_h或重要性排名"),
    ], columns=["result", "action", "reason"])
    scope["result"] = scope["result"].str.replace("{model}", os.path.basename(model_output_dir))
    scope.to_csv(
        os.path.join(model_output_dir, "需要替换的文件.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser(description="旁路验证MOCE的S_R正向下降修正")
    parser.add_argument("--model", choices=MODEL_NAMES, default="efficientnet_b0")
    parser.add_argument("--class-label", choices=["all", "0", "1"], default="all")
    parser.add_argument(
        "--stage",
        choices=["all", "importance", "ssc"],
        default="all",
        help="all执行全部；importance仅重算重要性；ssc读取修正结果续跑SSC/SDC",
    )
    args = parser.parse_args()

    weight_file = moce.MODEL_REGISTRY[args.model][0]
    weight_path = os.path.join(PROJECT_DIR, "结果", "模型权重", weight_file)
    model = moce.load_model(args.model, weight_path)
    labels = [0, 1] if args.class_label == "all" else [int(args.class_label)]
    model_output_dir = os.path.join(OUTPUT_DIR, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    for label in labels:
        source_dir = os.path.join(SOURCE_DIR, args.model, f"class_{label}")
        output_dir = os.path.join(model_output_dir, f"class_{label}")
        os.makedirs(output_dir, exist_ok=True)
        assignment = pd.read_csv(
            os.path.join(source_dir, "cluster_assignments.csv"),
            encoding="utf-8-sig",
        )

        if args.stage in ["all", "importance"]:
            corrected_scores, corrected_importance = evaluate_corrected_importance(
                model, label, assignment, output_dir
            )
        else:
            corrected_scores = pd.read_csv(
                os.path.join(output_dir, "concept_scores_per_image.csv"),
                encoding="utf-8-sig",
            )
            corrected_importance = pd.read_csv(
                os.path.join(output_dir, "concept_importance.csv"),
                encoding="utf-8-sig",
            )
        if args.stage in ["all", "ssc"]:
            moce.evaluate_ssc_sdc(
                model, label, assignment, corrected_importance, output_dir
            )
        save_comparison(
            source_dir, output_dir, corrected_scores, corrected_importance
        )
        old_top5 = pd.read_csv(
            os.path.join(source_dir, "concept_importance.csv"),
            encoding="utf-8-sig",
        ).nsmallest(5, "importance_rank")["cluster_id"].add(1).tolist()
        new_top5 = corrected_importance.nsmallest(
            5, "importance_rank"
        )["cluster_id"].add(1).tolist()
        print(f"类别{label}旧Top 5：{old_top5}")
        print(f"类别{label}新Top 5：{new_top5}")
        print(f"验证输出：{output_dir}")

    save_replacement_scope(model_output_dir)
    print("正式MOCE结果未被覆盖。")


if __name__ == "__main__":
    main()
