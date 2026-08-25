#!/usr/bin/env python3
"""使用医生bbox评价M0-F、A-long和C-long的外部空间关注。

A-long/C-long直接复用2026-08-19外部投影保存的内置注意力；
M0-F仅补跑同一checkpoint的后验CAM。三张病灶不清图只从bbox
空间评价和v2分类口径排除，不改写原始数据或既有正式结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from analyze_m0f_cam_internal_temporal import (
    paired_patient_spatial_difference,
)
from evaluate_mage_external_projection import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    MODEL_SPECS,
    ProjectionDataset,
    load_model_spec,
    patient_mean,
    run_val_self_test,
    shared_patient_bootstrap,
    threshold_metrics,
)
from evaluate_mage_internal_temporal_test import (
    GRID_SIZE,
    paired_spatial_bootstrap,
    spatial_predictions,
    summarize_spatial,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BBOX_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃镜多中心测试集_M0Keep预处理_v1_20260805/"
    "03_外部癌图标注整合_v2_20260824"
)
BBOX_MANIFEST = BBOX_ROOT / "外部多中心癌图bbox空间评价清单_v2.csv"
EXCLUSION_MANIFEST = BBOX_ROOT / "外部多中心癌图bbox排除清单_v2.csv"
FORMAL_EXTERNAL_ROOT = PROJECT_ROOT / "结果/MAGE/外部多中心描述性投影_20260819"
FORMAL_IMAGE_PREDICTIONS = FORMAL_EXTERNAL_ROOT / "external_image_predictions.csv"
FORMAL_SUMMARY = FORMAL_EXTERNAL_ROOT / "external_projection_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/外部多中心空间关注描述性评价_20260825"
SPATIAL_BOOTSTRAP_ITERATIONS = 2000
QUALITATIVE_PER_GROUP = 20


def parse_args() -> argparse.Namespace:
    """解析运行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", help="cpu或cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def build_spatial_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """连接bbox清单与既有外部逐图预测，返回539张空间队列。"""
    bbox = pd.read_csv(BBOX_MANIFEST, encoding="utf-8-sig", dtype={"patient_id": "string"})
    excluded = pd.read_csv(
        EXCLUSION_MANIFEST, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    formal = pd.read_csv(
        FORMAL_IMAGE_PREDICTIONS,
        encoding="utf-8-sig",
        dtype={"patient_id": "string"},
    )
    summary = json.loads(FORMAL_SUMMARY.read_text(encoding="utf-8"))
    if len(bbox) != 539 or bbox.patient_id.nunique() != 121 or len(excluded) != 3:
        raise ValueError("bbox v2计数不符合全量肉眼验收结果")
    if len(formal) != 1941 or formal.patient_id.nunique() != 1329:
        raise ValueError("既有外部逐图预测计数异常")

    bbox["image_path"] = bbox.source_keep_path.map(
        lambda value: str((PROJECT_ROOT / value).resolve())
    )
    missing = [path for path in bbox.image_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"bbox队列Keep图缺失{len(missing)}张")
    probability_columns = ["M0-F", "A-long", "C-long"]
    attention_columns = [
        f"{name}_attention_{cell}"
        for name in ("A-long", "C-long")
        for cell in range(GRID_SIZE * GRID_SIZE)
    ]
    joined = bbox.merge(
        formal[["image_path", "label", *probability_columns, *attention_columns]],
        on="image_path",
        how="left",
        validate="one_to_one",
        suffixes=("", "_formal"),
    )
    if joined[probability_columns].isna().any().any() or not joined.label_formal.eq(1).all():
        raise ValueError("bbox队列未能与既有外部癌图预测逐张连接")
    joined = joined.sort_values("annotation_index").reset_index(drop=True)
    joined = pd.concat([
        joined,
        pd.DataFrame({
            "image_index": np.arange(len(joined), dtype=int),
            "relative_path": joined.source_relative_path.to_numpy(),
        }),
    ], axis=1)
    return joined, excluded, summary


def collect_m0f_cam(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """提取M0-F后验CAM，对无正向癌证据图显式记为不可定义。"""
    loader = DataLoader(
        ProjectionDataset(frame), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    probabilities = np.zeros(len(frame), dtype=np.float64)
    attentions = np.full((len(frame), GRID_SIZE * GRID_SIZE), np.nan, dtype=np.float64)
    unavailable = []
    classifier = model.classifier[1]
    margin_weight = (classifier.weight[1] - classifier.weight[0]).detach()
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            feature_map = model.features(images)
            pooled = model.avgpool(feature_map)
            logits = model.classifier(torch.flatten(pooled, 1))
            positive_cam = torch.relu(
                torch.einsum("bchw,c->bhw", feature_map, margin_weight)
            )
            mass = positive_cam.flatten(1).sum(dim=1)
            valid = mass > 1e-12
            row_indices = indices.numpy()
            probabilities[row_indices] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            if valid.any():
                normalized = positive_cam[valid] / mass[valid, None, None]
                attentions[row_indices[valid.cpu().numpy()]] = (
                    normalized.flatten(1).double().cpu().numpy()
                )
            unavailable.extend(row_indices[~valid.cpu().numpy()].tolist())
            if batch_index == len(loader):
                print(
                    f"[CAM] batch {batch_index}/{len(loader)}; "
                    f"累计不可定义={len(unavailable)}", flush=True,
                )
    return probabilities, attentions, unavailable


def build_v2_classification_frame(excluded: pd.DataFrame) -> pd.DataFrame:
    """从既有1941张预测中排除三张病灶不清图。"""
    formal = pd.read_csv(
        FORMAL_IMAGE_PREDICTIONS,
        encoding="utf-8-sig",
        dtype={"patient_id": "string"},
    )
    excluded_paths = {
        str((PROJECT_ROOT / value).resolve()) for value in excluded.source_keep_path
    }
    frame = formal.loc[~formal.image_path.isin(excluded_paths)].copy()
    if len(frame) != 1938 or frame.patient_id.nunique() != 1329:
        raise ValueError(f"v2分类队列计数异常: {len(frame)}张/{frame.patient_id.nunique()}人")
    return frame


def classification_v2_metrics(
    frame: pd.DataFrame, formal_summary: dict
) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    """使用原val冻结阈值复算v2图像级和患者级分类指标。"""
    metrics, patients = {}, {}
    for name in MODEL_SPECS:
        patient = patient_mean(frame[["patient_id", "label", name]], name)
        patients[name] = patient
        model_info = formal_summary["models"][name]
        metrics[name] = {
            "image": threshold_metrics(
                frame.label, frame[name], float(model_info["image_threshold"])
            ),
            "patient": threshold_metrics(
                patient.label,
                patient.probability,
                float(model_info["patient_threshold"]),
            ),
        }
    bootstrap = shared_patient_bootstrap(patients, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED)
    return metrics, patients, bootstrap


def select_qualitative_cases(spatial: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """固定选取C改善、C失败和C/M0分歧三类诊断样本。"""
    merged = spatial["C-long"].merge(
        spatial["A-long"][["image_index", "pga", "normalized_aib"]],
        on="image_index", suffixes=("_C", "_A"), validate="one_to_one",
    ).merge(
        spatial["M0-F"][["image_index", "pga", "normalized_aib"]],
        on="image_index", validate="one_to_one",
    ).rename(columns={"pga": "pga_M0", "normalized_aib": "normalized_aib_M0"})
    groups = []
    improved = merged.loc[merged.pga_C.eq(1) & merged.pga_A.eq(0)].copy()
    improved["selection_group"] = "C_peak_correct_A_peak_wrong"
    groups.append(improved.sort_values("normalized_aib_C", ascending=False).head(QUALITATIVE_PER_GROUP))
    failed = merged.loc[merged.pga_C.eq(0)].copy()
    failed["selection_group"] = "C_peak_outside_bbox"
    groups.append(failed.sort_values("normalized_aib_C").head(QUALITATIVE_PER_GROUP))
    disagreement = merged.loc[merged.pga_C.ne(merged.pga_M0)].copy()
    disagreement["selection_group"] = "C_M0_peak_disagreement"
    disagreement["disagreement_strength"] = (
        disagreement.normalized_aib_C - disagreement.normalized_aib_M0
    ).abs()
    groups.append(disagreement.sort_values("disagreement_strength", ascending=False).head(QUALITATIVE_PER_GROUP))
    selected = pd.concat(groups, ignore_index=True).drop_duplicates("image_index")
    return selected.reset_index(drop=True)


def render_four_panel_subset(
    frame: pd.DataFrame,
    attentions: dict[str, np.ndarray],
    selection: pd.DataFrame,
    output: Path,
) -> None:
    """为固定诊断子集导出原图框和三模型空间图四联图。"""
    output.mkdir(parents=True)
    for order, selected in enumerate(selection.itertuples(index=False), start=1):
        row = frame.iloc[int(selected.image_index)]
        with Image.open(row.image_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        maps = {name: values[int(selected.image_index)] for name, values in attentions.items()}
        vmax = max(float(values.max()) for values in maps.values())
        figure, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(image)
        axes[0].set_title("Keep image + lesion bbox")
        for axis, (name, values) in zip(axes[1:], maps.items()):
            axis.imshow(image)
            axis.imshow(
                values.reshape(GRID_SIZE, GRID_SIZE), cmap="jet", alpha=0.45,
                vmin=0.0, vmax=vmax,
                extent=(0, image.shape[1], image.shape[0], 0),
                interpolation="bilinear",
            )
            axis.set_title(name)
        for axis in axes:
            axis.add_patch(Rectangle(
                (row.bbox_x1, row.bbox_y1),
                row.bbox_x2 - row.bbox_x1,
                row.bbox_y2 - row.bbox_y1,
                fill=False, edgecolor="lime", linewidth=2,
            ))
            axis.axis("off")
        figure.suptitle(
            f"{selected.selection_group} | {row.annotation_image_name} | "
            f"patient={row.patient_id}"
        )
        figure.tight_layout()
        figure.savefig(output / f"{order:03d}_{row.annotation_image_name}.png", dpi=110)
        plt.close(figure)


def main() -> None:
    """执行清单预检、M0-F CAM、三模型空间统计与v2分类复算。"""
    args = parse_args()
    frame, excluded, formal_summary = build_spatial_frame()
    classification_frame = build_v2_classification_frame(excluded)
    print(
        f"[preflight] bbox={len(frame)}张/{frame.patient_id.nunique()}人; "
        f"v2分类={len(classification_frame)}张/{classification_frame.patient_id.nunique()}人",
        flush=True,
    )
    if args.preflight_only:
        return
    if args.output.exists():
        raise FileExistsError(f"正式输出目录已存在，拒绝覆盖: {args.output}")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定cuda但当前不可用")

    loaded_m0 = load_model_spec("M0-F", MODEL_SPECS["M0-F"], device)
    val_self_test = run_val_self_test(
        "M0-F", loaded_m0, args.batch_size, args.num_workers, device
    )
    m0_probability, m0_attention, m0_unavailable = collect_m0f_cam(
        loaded_m0["model"], frame, args.batch_size, args.num_workers, device
    )
    m0_probability_diff = float(np.max(np.abs(m0_probability - frame["M0-F"].to_numpy())))

    attentions = {
        "M0-F post-hoc CAM": m0_attention,
        "A-long built-in attention": frame[[
            f"A-long_attention_{cell}" for cell in range(GRID_SIZE * GRID_SIZE)
        ]].to_numpy(dtype=np.float64),
        "C-long built-in attention": frame[[
            f"C-long_attention_{cell}" for cell in range(GRID_SIZE * GRID_SIZE)
        ]].to_numpy(dtype=np.float64),
    }
    m0_valid = ~np.isnan(m0_attention).any(axis=1)
    m0_frame = frame.loc[m0_valid].reset_index(drop=True)
    spatial = {
        "M0-F": spatial_predictions(
            m0_frame, m0_attention[m0_valid], "M0-F"
        ),
        "A-long": spatial_predictions(frame, attentions["A-long built-in attention"], "A-long"),
        "C-long": spatial_predictions(frame, attentions["C-long built-in attention"], "C-long"),
    }
    summaries = {name: summarize_spatial(values) for name, values in spatial.items()}
    comparisons = {
        "C-long_minus_A-long": paired_spatial_bootstrap(
            spatial["A-long"], spatial["C-long"], SPATIAL_BOOTSTRAP_ITERATIONS
        ),
        "A-long_minus_M0-F": paired_patient_spatial_difference(
            spatial["M0-F"], spatial["A-long"], SPATIAL_BOOTSTRAP_ITERATIONS
        ),
        "C-long_minus_M0-F": paired_patient_spatial_difference(
            spatial["M0-F"], spatial["C-long"], SPATIAL_BOOTSTRAP_ITERATIONS
        ),
    }
    classification_metrics, patients, classification_bootstrap = classification_v2_metrics(
        classification_frame, formal_summary
    )
    selection = select_qualitative_cases(spatial)

    args.output.mkdir(parents=True)
    pd.concat(spatial.values(), ignore_index=True).to_csv(
        args.output / "external_spatial_predictions_v2.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "model": name,
            "attention_type": "post-hoc CAM" if name == "M0-F" else "built-in attention",
            "mean_aib": values["mean_aib"],
            "mean_normalized_aib": values["mean_normalized_aib"],
            "pga": values["pga"],
            "peak_in_bbox_count": int(round(values["pga"] * values["cancer_images"])),
            "cancer_images": values["cancer_images"],
            "cancer_patients": values["cancer_patients"],
        }
        for name, values in summaries.items()
    ]).to_csv(
        args.output / "external_spatial_summary_v2.csv",
        index=False, encoding="utf-8-sig",
    )
    classification_rows = []
    for name in MODEL_SPECS:
        for level in ("image", "patient"):
            classification_rows.append({
                "model": name, "level": level, **classification_metrics[name][level]
            })
    pd.DataFrame(classification_rows).to_csv(
        args.output / "external_classification_metrics_v2.csv",
        index=False, encoding="utf-8-sig",
    )
    patient_output = patients["M0-F"][["patient_id", "label"]].copy()
    for name in MODEL_SPECS:
        patient_output[f"{name}_probability"] = patients[name].probability
    patient_output.to_csv(
        args.output / "external_patient_predictions_v2.csv",
        index=False, encoding="utf-8-sig",
    )
    selection.to_csv(
        args.output / "qualitative_subset.csv", index=False, encoding="utf-8-sig"
    )
    frame.loc[m0_unavailable, [
        "image_index", "annotation_index", "annotation_image_name", "patient_id",
        "source_relative_path", "M0-F",
    ]].assign(
        cam_status="unavailable_no_positive_cancer_evidence_after_relu"
    ).to_csv(
        args.output / "m0f_cam_unavailable.csv", index=False, encoding="utf-8-sig"
    )
    render_four_panel_subset(frame, attentions, selection, args.output / "four_panel_subset")
    summary = {
        "stage": "MAGE_external_multicenter_bbox_spatial_descriptive_evaluation",
        "cohort": {
            "spatial_images": len(frame),
            "spatial_patients": int(frame.patient_id.nunique()),
            "excluded_unclear_lesion_images": len(excluded),
            "classification_v2_images": len(classification_frame),
            "classification_v2_patients": int(classification_frame.patient_id.nunique()),
        },
        "m0f_val_self_test": val_self_test,
        "m0f_cam_probability_max_abs_difference_vs_20260819": m0_probability_diff,
        "m0f_cam_unavailable_images": int(len(m0_unavailable)),
        "m0f_cam_unavailable_indices": [int(value) for value in m0_unavailable],
        "spatial": summaries,
        "spatial_paired_patient_bootstrap": comparisons,
        "classification_v2": classification_metrics,
        "classification_v2_patient_bootstrap": classification_bootstrap,
        "boundaries": [
            "本外部队列已参与旧工程决策，本轮只作描述性空间评价",
            "三张病灶不清图按临床反馈排除，未损失患者",
            "A/C为前向分类内置注意力；M0-F为线性分类权重后验CAM，机制不完全同构",
            "本轮不重扫分类阈值，不调整checkpoint，不回灌训练",
        ],
    }
    (args.output / "external_spatial_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {args.output}", flush=True)
    for name in ("M0-F", "A-long", "C-long"):
        values = summaries[name]
        print(
            f"  {name}: AiB={values['mean_aib']:.4f} "
            f"nAiB={values['mean_normalized_aib']:.4f} "
            f"PGA={values['pga']:.4f}", flush=True,
        )


if __name__ == "__main__":
    main()
