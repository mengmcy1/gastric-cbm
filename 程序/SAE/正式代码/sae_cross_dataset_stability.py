"""Compare locked SAE features across train, val, internal test, and external test."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ACTIVE_EPS = 1e-8
PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SAE_RUN = (
    PROJECT_DIR / "结果/SAE/去偏重训练_v1/resnet50/"
    "formal_expA_resnet50_sae_l1_0005_20260724"
)
DEFAULT_TEST_RUN = (
    PROJECT_DIR / "结果/SAE/去偏重训练_v1/resnet50/"
    "internal_test_resnet50_sae_20260731"
)
DEFAULT_EXTERNAL_RUN = (
    PROJECT_DIR / "结果/SAE/去偏重训练_v1/resnet50/"
    "external_multicenter_resnet50_sae_20260728"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "结果/SAE跨数据集稳定性/去偏重训练_v1/resnet50/"
    "cross_dataset_feature_stability_20260731"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="锁定ResNet50 SAE的四数据集Feature稳定性分析",
    )
    parser.add_argument("--sae-run", type=Path, default=DEFAULT_SAE_RUN)
    parser.add_argument("--test-run", type=Path, default=DEFAULT_TEST_RUN)
    parser.add_argument("--external-run", type=Path, default=DEFAULT_EXTERNAL_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-features", type=int, default=30,
        help="仅由train/val排名后进入跨数据集候选表的Feature数量",
    )
    parser.add_argument(
        "--top-patients", type=int, default=6,
        help="每个候选Feature在每个数据集保存的Top患者数",
    )
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_dataset(name, projection_path, metadata_path):
    with np.load(require_file(projection_path)) as projection:
        activations = projection["activations"].astype(np.float32)
    metadata = pd.read_csv(require_file(metadata_path), encoding="utf-8-sig")
    required = {"patient_id", "label", "image_relpath"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{name} metadata缺少字段: {sorted(missing)}")
    if len(metadata) != len(activations):
        raise ValueError(
            f"{name}行数不一致: metadata={len(metadata)}, activations={len(activations)}",
        )
    metadata = metadata.copy()
    metadata["patient_id"] = metadata["patient_id"].astype(str)
    metadata["label"] = metadata["label"].astype(int)
    if "domain" not in metadata:
        metadata["domain"] = "未知"
    if "hospital" not in metadata:
        metadata["hospital"] = "未知"
    return {"name": name, "activations": activations, "metadata": metadata}


def patient_max_table(dataset):
    metadata = dataset["metadata"]
    activations = dataset["activations"]
    patient_codes, patient_ids = pd.factorize(metadata["patient_id"], sort=True)
    patient_max = np.zeros((len(patient_ids), activations.shape[1]), dtype=np.float32)
    np.maximum.at(patient_max, patient_codes, activations)

    first = metadata.drop_duplicates("patient_id").set_index("patient_id").reindex(
        patient_ids,
    )
    first.insert(0, "patient_id", patient_ids.astype(str))
    first = first.reset_index(drop=True)
    return patient_max, first, patient_codes


def safe_auc(labels, values):
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    if np.unique(labels).size < 2 or np.allclose(values, values[0]):
        return np.nan
    return float(roc_auc_score(labels, values))


def feature_metrics(dataset):
    activations = dataset["activations"]
    patient_max, patients, _ = patient_max_table(dataset)
    labels = patients["label"].to_numpy(dtype=int)
    domains = patients["domain"].fillna("未知").astype(str).to_numpy()
    active = patient_max > ACTIVE_EPS
    label_0 = labels == 0
    label_1 = labels == 1

    rows = []
    for feature_id in range(activations.shape[1]):
        values = patient_max[:, feature_id]
        label_auc = safe_auc(labels, values)
        provincial = domains == "省人民"
        external = domains == "外院"
        source_auc = (
            safe_auc(external.astype(int), values)
            if provincial.any() and external.any() else np.nan
        )
        mean_0 = float(values[label_0].mean()) if label_0.any() else np.nan
        mean_1 = float(values[label_1].mean()) if label_1.any() else np.nan
        rows.append({
            "dataset": dataset["name"],
            "feature_id": feature_id,
            "image_count": len(activations),
            "patient_count": len(patients),
            "active_image_count": int((activations[:, feature_id] > ACTIVE_EPS).sum()),
            "active_patient_count": int(active[:, feature_id].sum()),
            "active_patient_rate": float(active[:, feature_id].mean()),
            "mean_patient_max": float(values.mean()),
            "median_patient_max": float(np.median(values)),
            "mean_patient_max_label_0": mean_0,
            "mean_patient_max_label_1": mean_1,
            "cancer_activation_difference": mean_1 - mean_0,
            "label_auc": label_auc,
            "label_discrimination": (
                abs(label_auc - 0.5) * 2 if np.isfinite(label_auc) else np.nan
            ),
            "source_auc_external": source_auc,
            "source_discrimination": (
                abs(source_auc - 0.5) * 2 if np.isfinite(source_auc) else np.nan
            ),
        })
    return pd.DataFrame(rows), patient_max, patients


def load_feature_registry(sae_run):
    summary = pd.read_csv(
        require_file(sae_run / "feature_summary.csv"), encoding="utf-8-sig",
    ).sort_values("feature_id")
    decisions = pd.read_csv(
        require_file(sae_run / "feature筛选/feature_pruning_decisions.csv"),
        encoding="utf-8-sig",
    ).sort_values("feature_id")
    registry = summary[["feature_id", "cancer_margin_direction"]].merge(
        decisions[["feature_id", "kept_after_pruning"]],
        on="feature_id", validate="one_to_one",
    )
    return registry


def build_wide_table(long_metrics, registry):
    metric_names = [
        "active_patient_count", "active_patient_rate", "mean_patient_max",
        "cancer_activation_difference", "label_auc", "label_discrimination",
        "source_auc_external", "source_discrimination",
    ]
    wide = registry.copy()
    for dataset in ["train", "val", "internal_test", "external_test"]:
        subset = (
            long_metrics[long_metrics["dataset"] == dataset]
            .set_index("feature_id")[metric_names]
            .add_suffix(f"_{dataset}")
            .reset_index()
        )
        wide = wide.merge(subset, on="feature_id", validate="one_to_one")

    train_sign = np.sign(wide["cancer_activation_difference_train"])
    val_sign = np.sign(wide["cancer_activation_difference_val"])
    test_sign = np.sign(wide["cancer_activation_difference_internal_test"])
    external_sign = np.sign(wide["cancer_activation_difference_external_test"])
    wide["train_val_label_direction_consistent"] = (
        (train_sign != 0) & (train_sign == val_sign)
    )
    wide["all_dataset_label_direction_consistent"] = (
        wide["train_val_label_direction_consistent"]
        & (train_sign == test_sign) & (train_sign == external_sign)
    )
    wide["train_val_min_label_discrimination"] = wide[
        ["label_discrimination_train", "label_discrimination_val"]
    ].min(axis=1)
    wide["confirmation_min_label_discrimination"] = wide[
        ["label_discrimination_internal_test", "label_discrimination_external_test"]
    ].min(axis=1)
    wide["min_active_patient_rate_all"] = wide[[
        "active_patient_rate_train", "active_patient_rate_val",
        "active_patient_rate_internal_test", "active_patient_rate_external_test",
    ]].min(axis=1)
    wide["active_patient_rate_range_all"] = (
        wide[[
            "active_patient_rate_train", "active_patient_rate_val",
            "active_patient_rate_internal_test", "active_patient_rate_external_test",
        ]].max(axis=1)
        - wide[[
            "active_patient_rate_train", "active_patient_rate_val",
            "active_patient_rate_internal_test", "active_patient_rate_external_test",
        ]].min(axis=1)
    )
    return wide


def select_candidates(wide, top_features):
    eligible = wide[
        wide["kept_after_pruning"].astype(bool)
        & wide["train_val_label_direction_consistent"]
        & (wide["active_patient_count_train"] >= 5)
        & (wide["active_patient_count_val"] >= 5)
    ].copy()
    eligible["selection_score_train_val"] = (
        eligible["train_val_min_label_discrimination"]
        * np.sqrt(
            eligible["active_patient_rate_train"]
            * eligible["active_patient_rate_val"],
        )
    )
    eligible["candidate_direction"] = np.where(
        eligible["cancer_activation_difference_train"] > 0,
        "cancer_associated",
        "noncancer_associated",
    )
    eligible["activation_margin_aligned_train_val"] = (
        np.sign(eligible["cancer_activation_difference_train"])
        == np.sign(eligible["cancer_margin_direction"])
    )
    eligible = eligible.sort_values(
        [
            "selection_score_train_val",
            "train_val_min_label_discrimination",
            "feature_id",
        ],
        ascending=[False, False, True],
    )
    eligible["selection_rank_train_val"] = np.arange(1, len(eligible) + 1)
    eligible["direction_rank_train_val"] = (
        eligible.groupby("candidate_direction", sort=False).cumcount() + 1
    )
    cancer_count = (top_features + 1) // 2
    noncancer_count = top_features // 2
    eligible["selected_for_confirmation"] = (
        (
            (eligible["candidate_direction"] == "cancer_associated")
            & (eligible["direction_rank_train_val"] <= cancer_count)
        )
        | (
            (eligible["candidate_direction"] == "noncancer_associated")
            & (eligible["direction_rank_train_val"] <= noncancer_count)
        )
    )
    return eligible


def select_source_candidates(wide, top_features):
    train_effect = wide["source_auc_external_train"] - 0.5
    val_effect = wide["source_auc_external_val"] - 0.5
    eligible = wide[
        wide["kept_after_pruning"].astype(bool)
        & train_effect.notna() & val_effect.notna()
        & (np.sign(train_effect) == np.sign(val_effect))
        & (wide["active_patient_count_train"] >= 5)
        & (wide["active_patient_count_val"] >= 5)
    ].copy()
    eligible["train_val_min_source_discrimination"] = eligible[[
        "source_discrimination_train", "source_discrimination_val",
    ]].min(axis=1)
    eligible["source_selection_score_train_val"] = (
        eligible["train_val_min_source_discrimination"]
        * np.sqrt(
            eligible["active_patient_rate_train"]
            * eligible["active_patient_rate_val"],
        )
    )
    eligible = eligible.sort_values(
        ["source_selection_score_train_val", "feature_id"],
        ascending=[False, True],
    ).head(top_features)
    eligible["source_selection_rank_train_val"] = np.arange(1, len(eligible) + 1)
    return eligible


def correlation_table(long_metrics, metric):
    table = long_metrics.pivot(
        index="feature_id", columns="dataset", values=metric,
    )
    order = ["train", "val", "internal_test", "external_test"]
    return table[order].corr(method="spearman")


def top_patient_rows(dataset, patient_max, patients, feature_ids, top_patients):
    metadata = dataset["metadata"]
    rows = []
    for feature_id in feature_ids:
        values = patient_max[:, feature_id]
        order = np.argsort(-values, kind="stable")
        kept = 0
        for patient_index in order:
            activation = float(values[patient_index])
            if activation <= ACTIVE_EPS:
                break
            patient_id = str(patients.iloc[patient_index]["patient_id"])
            image_mask = metadata["patient_id"].astype(str).to_numpy() == patient_id
            image_indices = np.flatnonzero(image_mask)
            best_image_index = image_indices[
                np.argmax(dataset["activations"][image_indices, feature_id])
            ]
            image = metadata.iloc[best_image_index]
            rows.append({
                "dataset": dataset["name"],
                "feature_id": int(feature_id),
                "rank": kept + 1,
                "patient_id": patient_id,
                "label": int(patients.iloc[patient_index]["label"]),
                "domain": patients.iloc[patient_index]["domain"],
                "hospital": patients.iloc[patient_index]["hospital"],
                "activation": activation,
                "image_relpath": image["image_relpath"],
            })
            kept += 1
            if kept >= top_patients:
                break
    return rows


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {args.output}")
    if args.top_features <= 0 or args.top_patients <= 0:
        raise ValueError("top-features和top-patients必须为正整数")

    datasets = [
        load_dataset(
            "train",
            args.sae_run / "特征缓存/train_sae_projection.npz",
            args.sae_run / "特征缓存/train_metadata.csv",
        ),
        load_dataset(
            "val",
            args.sae_run / "特征缓存/val_sae_projection.npz",
            args.sae_run / "特征缓存/val_metadata.csv",
        ),
        load_dataset(
            "internal_test",
            args.test_run / "特征缓存/test_sae_projection.npz",
            args.test_run / "特征缓存/test_metadata.csv",
        ),
        load_dataset(
            "external_test",
            args.external_run / "特征缓存/external_sae_projection.npz",
            args.external_run / "特征缓存/external_metadata.csv",
        ),
    ]
    hidden_dims = {dataset["activations"].shape[1] for dataset in datasets}
    if hidden_dims != {512}:
        raise ValueError(f"预期四个数据集均为512维，实际为: {sorted(hidden_dims)}")

    registry = load_feature_registry(args.sae_run)
    long_frames = []
    patient_data = {}
    for dataset in datasets:
        metrics, patient_max, patients = feature_metrics(dataset)
        long_frames.append(metrics)
        patient_data[dataset["name"]] = (patient_max, patients)
        print(
            f"{dataset['name']}: {len(dataset['metadata'])}张/"
            f"{len(patients)}位患者",
        )
    long_metrics = pd.concat(long_frames, ignore_index=True)
    wide = build_wide_table(long_metrics, registry)
    candidates = select_candidates(wide, min(args.top_features, len(wide)))
    selected = (
        candidates[candidates["selected_for_confirmation"]]
        .sort_values(["candidate_direction", "direction_rank_train_val"])
        .copy()
    )
    source_candidates = select_source_candidates(
        wide, min(args.top_features, len(wide)),
    )
    selected_ids = selected["feature_id"].astype(int).tolist()

    top_rows = []
    for dataset in datasets:
        patient_max, patients = patient_data[dataset["name"]]
        top_rows.extend(
            top_patient_rows(
                dataset, patient_max, patients, selected_ids, args.top_patients,
            ),
        )

    args.output.mkdir(parents=True)
    long_metrics.to_csv(
        args.output / "feature_metrics_long.csv", index=False, encoding="utf-8-sig",
    )
    wide.to_csv(
        args.output / "feature_stability_all.csv", index=False, encoding="utf-8-sig",
    )
    selected.to_csv(
        args.output / "train_val_selected_features.csv",
        index=False, encoding="utf-8-sig",
    )
    source_candidates.to_csv(
        args.output / "train_val_source_related_features.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(top_rows).to_csv(
        args.output / "selected_feature_top_patients.csv",
        index=False, encoding="utf-8-sig",
    )
    for metric in [
        "cancer_activation_difference", "active_patient_rate",
        "label_discrimination",
    ]:
        correlation_table(long_metrics, metric).to_csv(
            args.output / f"spearman_{metric}.csv",
            encoding="utf-8-sig",
        )

    summary = {
        "selection_rule": (
            "kept feature; >=5 active patients in train and val; "
            "same cancer activation direction in train/val; ranked only by "
            "train/val minimum label discrimination times geometric mean coverage"
        ),
        "test_external_role": (
            "internal_test and external_test are confirmation-only and do not "
            "participate in candidate selection"
        ),
        "dataset_counts": {
            dataset["name"]: {
                "images": int(len(dataset["metadata"])),
                "patients": int(len(patient_data[dataset["name"]][1])),
            }
            for dataset in datasets
        },
        "total_features": int(len(wide)),
        "kept_features": int(wide["kept_after_pruning"].sum()),
        "eligible_train_val_features": int(len(candidates)),
        "selected_feature_count": int(len(selected)),
        "selected_feature_ids": selected_ids,
        "selected_feature_direction_counts": {
            str(name): int(count)
            for name, count in selected["candidate_direction"].value_counts().items()
        },
        "selected_all_dataset_direction_consistent": int(
            selected["all_dataset_label_direction_consistent"].sum()
        ),
    }
    with open(args.output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(args.output / "README.md", "w", encoding="utf-8") as handle:
        handle.write(
            "# SAE跨数据集Feature稳定性\n\n"
            "候选Feature只由train/val激活覆盖和标签区分度筛选。internal test与"
            "external test仅用于锁定后的稳定性描述，不参与筛选、剪枝或调参。\n\n"
            "- `feature_metrics_long.csv`：四数据集逐Feature长表；\n"
            "- `feature_stability_all.csv`：512个Feature横向宽表；\n"
            "- `train_val_selected_features.csv`：train/val分层预选的癌/非癌候选；\n"
            "- `train_val_source_related_features.csv`：train/val来源相关候选；\n"
            "- `selected_feature_top_patients.csv`：候选Feature每域Top患者与代表图；\n"
            "- `spearman_*.csv`：四域Feature排序的Spearman相关矩阵。\n\n"
            "本目录的自动稳定性不等于医学概念真实性。空间响应图和临床命名需在"
            "固定候选名单上另行复核。\n"
        )
    print(f"候选Feature: {selected_ids}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
