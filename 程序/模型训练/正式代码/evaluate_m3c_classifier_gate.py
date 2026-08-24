#!/usr/bin/env python3
"""M3c-A：用冻结分类概率决定是否展示冻结M1预测框。

本脚本只读取M1 warmup-only产品已经保存的val预测，不训练、不重新前向推理、
不读取internal test或external。每个seed在val癌图分类概率上锁定召回>=0.90的
最高阈值，再报告非癌FP、分辨率子组召回、患者级结果和QC。
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
M1_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3c分类门控_0804/正式验证集筛选"
SEEDS = (42, 202, 503)
MIN_GROUP_CANCER = 15
GROUP_GAP_TOLERANCE = 0.05
MIN_FP_DROP = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,202,503")
    parser.add_argument("--m1-root", type=Path, default=M1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--qc-count", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path, chunk_size=1024 * 1024):
    """流式计算文件SHA-256，记录冻结输入来源。"""
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def lock_recall_threshold(values, target=0.90):
    """返回满足实际召回>=target的最高候选值；判定始终使用>=。"""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("锁阈值至少需要一张癌图")
    for threshold in np.sort(np.unique(values))[::-1]:
        if float(np.mean(values >= threshold)) >= target:
            return float(threshold)
    return float(values.min())


def rate_metrics(labels, decisions):
    """计算图像或患者层面的癌召回、非癌FP和计数。"""
    labels = np.asarray(labels, dtype=int)
    decisions = np.asarray(decisions, dtype=bool)
    cancer = labels == 1
    noncancer = ~cancer
    return {
        "count": int(len(labels)),
        "cancer_count": int(cancer.sum()),
        "noncancer_count": int(noncancer.sum()),
        "cancer_detected": int((decisions & cancer).sum()),
        "noncancer_false_positive": int((decisions & noncancer).sum()),
        "cancer_recall": float(decisions[cancer].mean()) if cancer.any() else None,
        "noncancer_fp": float(decisions[noncancer].mean()) if noncancer.any() else None,
    }


def add_resolution_gate_group(frame):
    """按预注册把原始size_group冻结合并为large与non_large。"""
    result = frame.copy()
    result["resolution_gate_group"] = result.size_group.map({
        "large_1025_1600": "large",
        "medium_641_1024": "non_large",
        "small_le640": "non_large",
    })
    return result


def resolution_diagnostics(frame, overall_recall):
    """输出分辨率子组门控率；癌图不足15的组只报告、不参与验收。"""
    rows = []
    for group_name, group in frame.groupby("resolution_gate_group", dropna=False):
        metrics = rate_metrics(group.label, group.gate_display)
        eligible = metrics["cancer_count"] >= MIN_GROUP_CANCER
        recall = metrics["cancer_recall"]
        passed = bool(
            not eligible
            or (recall is not None and recall >= overall_recall - GROUP_GAP_TOLERANCE)
        )
        rows.append({
            "group": str(group_name),
            **metrics,
            "minimum_cancer_for_gate": MIN_GROUP_CANCER,
            "participates_in_gate": eligible,
            "required_cancer_recall": overall_recall - GROUP_GAP_TOLERANCE,
            "passed": passed,
        })
    return pd.DataFrame(rows)


def box_iou(frame):
    """计算冻结M1框与真值框IoU，仅用于癌图几何核对。"""
    x1 = np.maximum(frame.pred_x1, frame.gt_x1)
    y1 = np.maximum(frame.pred_y1, frame.gt_y1)
    x2 = np.minimum(frame.pred_x2, frame.gt_x2)
    y2 = np.minimum(frame.pred_y2, frame.gt_y2)
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    pred_area = np.maximum(0, frame.pred_x2 - frame.pred_x1) * np.maximum(
        0, frame.pred_y2 - frame.pred_y1
    )
    gt_area = np.maximum(0, frame.gt_x2 - frame.gt_x1) * np.maximum(
        0, frame.gt_y2 - frame.gt_y1
    )
    union = pred_area + gt_area - intersection
    return np.divide(intersection, union, out=np.zeros(len(frame)), where=union > 0)


def geometry_metrics(frame):
    """报告有效癌框的IoU与中心命中；门控不允许修改任何坐标。"""
    valid = frame.loc[frame.valid_box.astype(bool)].copy()
    if valid.empty:
        return {"count": 0, "mean_iou": None, "center_hit": None}
    iou = box_iou(valid)
    center_x = (valid.pred_x1 + valid.pred_x2) / 2
    center_y = (valid.pred_y1 + valid.pred_y2) / 2
    center_hit = (
        center_x.between(valid.gt_x1, valid.gt_x2)
        & center_y.between(valid.gt_y1, valid.gt_y2)
    )
    return {
        "count": int(len(valid)),
        "mean_iou": float(iou.mean()),
        "center_hit": float(center_hit.mean()),
    }


def patient_table(frame):
    """患者任一图被门控展示即视为患者级阳性。"""
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        cancer_probability=("cancer_probability", "max"),
        gate_display=("gate_display", "max"),
        m1_localization_display=("m1_localization_display", "max"),
    )


def patient_bootstrap(patient, iterations, seed):
    """按标签分层对患者有放回抽样，计算门控癌召回与非癌FP的95%CI。"""
    rng = np.random.default_rng(seed)
    groups = {
        label: patient.loc[patient.label.eq(label)].reset_index(drop=True)
        for label in (0, 1)
    }
    if any(group.empty for group in groups.values()):
        raise ValueError("患者bootstrap要求癌与非癌患者均非空")
    recall_values = np.empty(iterations)
    fp_values = np.empty(iterations)
    for index in range(iterations):
        sampled = {
            label: group.iloc[rng.integers(0, len(group), size=len(group))]
            for label, group in groups.items()
        }
        recall_values[index] = sampled[1].gate_display.mean()
        fp_values[index] = sampled[0].gate_display.mean()
    return {
        "iterations": iterations,
        "cancer_recall_ci95": np.quantile(recall_values, [0.025, 0.975]).tolist(),
        "noncancer_fp_ci95": np.quantile(fp_values, [0.025, 0.975]).tolist(),
    }


def draw_box(draw, row, color="red", width=3):
    """在224x224 QC图上绘制冻结M1归一化预测框。"""
    coordinates = tuple(
        int(round(float(value) * 223))
        for value in (row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2)
    )
    draw.rectangle(coordinates, outline=color, width=width)


def save_contact_sheet(rows, image_root, path, count):
    """保存简单4列QC联系表，标签写入图像顶部，避免依赖中文字体。"""
    selected = list(rows)[:count]
    if not selected:
        return
    tile_width, image_size, header = 224, 224, 24
    columns = 4
    rows_count = int(np.ceil(len(selected) / columns))
    canvas = Image.new("RGB", (columns * tile_width, rows_count * (image_size + header)), "white")
    for index, row in enumerate(selected):
        with Image.open(image_root / row.image_relpath) as source:
            image = source.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        draw_box(draw, row)
        tile = Image.new("RGB", (tile_width, image_size + header), "white")
        tile.paste(image, (0, header))
        label = f"y={row.label} gate={int(row.gate_display)} p={row.cancer_probability:.3f}"
        ImageDraw.Draw(tile).text((4, 5), label, fill="black")
        x = (index % columns) * tile_width
        y = (index // columns) * (image_size + header)
        canvas.paste(tile, (x, y))
    canvas.save(path, quality=95)


def export_qc(frame, image_root, output, count):
    """导出癌图展示/拒绝和非癌误展示三类可审查样例。"""
    cancer = frame.loc[frame.label.eq(1)]
    noncancer = frame.loc[frame.label.eq(0)]
    save_contact_sheet(
        cancer.loc[cancer.gate_display].sort_values("cancer_probability").itertuples(),
        image_root, output / "gated_cancer_qc.jpg", count,
    )
    save_contact_sheet(
        cancer.loc[~cancer.gate_display].sort_values("cancer_probability", ascending=False).itertuples(),
        image_root, output / "rejected_cancer_qc.jpg", count,
    )
    save_contact_sheet(
        noncancer.loc[noncancer.gate_display].sort_values("cancer_probability", ascending=False).itertuples(),
        image_root, output / "gated_noncancer_qc.jpg", count,
    )


def validate_inputs(frame, config, seed):
    """拒绝错seed、非val预测或曾评估test的M1输入。"""
    required = {
        "image_relpath", "patient_id", "label", "split", "size_group",
        "cancer_probability", "localization_confidence", "valid_box",
        "gt_x1", "gt_y1", "gt_x2", "gt_y2",
        "pred_x1", "pred_y1", "pred_x2", "pred_y2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"M1 val预测缺少字段: {sorted(missing)}")
    if int(config["seed"]) != seed:
        raise ValueError(f"M1 config seed错配: {config['seed']} != {seed}")
    if set(frame.split.astype(str)) != {"val"}:
        raise ValueError("M3c-A输入必须只包含val")
    if bool(config.get("test_evaluated", config.get("evaluate_test", False))):
        raise ValueError("M1产品记录为已评估test，拒绝作为锁定输入")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("val必须同时包含癌与非癌")


def evaluate_seed(seed, args):
    """完成单个seed的冻结门控、验收、分层、bootstrap和QC。"""
    source = args.m1_root / f"m1_balanced_keep_efficientnet_b0_seed{seed}_warmup_product"
    prediction_path = source / "val_image_predictions.csv"
    config_path = source / "config.json"
    checkpoint_path = source / "m1_best_warmup_localization.pth"
    for path in (prediction_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_root / f"m3c_balanced_keep_efficientnet_b0_seed{seed}"
    if output.exists():
        raise FileExistsError(f"输出已存在，请更换输出根目录: {output}")
    output.mkdir(parents=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(prediction_path, encoding="utf-8-sig")
    validate_inputs(frame, config, seed)
    frame = add_resolution_gate_group(frame)

    cancer_probability = frame.loc[frame.label.eq(1), "cancer_probability"]
    gate_threshold = lock_recall_threshold(cancer_probability, target=0.90)
    localization_threshold = lock_recall_threshold(
        frame.loc[frame.label.eq(1), "localization_confidence"], target=0.90
    )
    frame["gate_display"] = frame.cancer_probability >= gate_threshold
    frame["m1_localization_display"] = (
        frame.localization_confidence >= localization_threshold
    )

    image_metrics = rate_metrics(frame.label, frame.gate_display)
    baseline_metrics = rate_metrics(frame.label, frame.m1_localization_display)
    fp_drop = baseline_metrics["noncancer_fp"] - image_metrics["noncancer_fp"]
    groups = resolution_diagnostics(frame, image_metrics["cancer_recall"])
    group_gate_passed = bool(groups.loc[groups.participates_in_gate, "passed"].all())
    operation_point_passed = image_metrics["cancer_recall"] >= 0.90
    fp_gate_passed = fp_drop >= MIN_FP_DROP
    seed_passed = operation_point_passed and fp_gate_passed and group_gate_passed

    patient = patient_table(frame)
    patient_gate_metrics = rate_metrics(patient.label, patient.gate_display)
    patient_baseline_metrics = rate_metrics(patient.label, patient.m1_localization_display)
    bootstrap = patient_bootstrap(patient, args.bootstrap, seed)
    geometry_all = geometry_metrics(frame)
    geometry_displayed = geometry_metrics(frame.loc[frame.gate_display])

    coordinate_columns = [
        "pred_x1", "pred_y1", "pred_x2", "pred_y2",
    ]
    coordinate_checksum_before = file_sha256(prediction_path)
    coordinate_max_abs_diff = float(np.max(np.abs(
        frame[coordinate_columns].to_numpy()
        - pd.read_csv(prediction_path, encoding="utf-8-sig")[coordinate_columns].to_numpy()
    )))
    if coordinate_max_abs_diff != 0:
        raise RuntimeError("M3c门控意外修改了M1预测框坐标")

    image_root = Path(config["image_root"])
    if not frame.image_relpath.map(lambda value: (image_root / value).is_file()).all():
        raise FileNotFoundError("M1 val预测存在缺失的image_relpath")
    export_qc(frame, image_root, output, args.qc_count)
    frame.to_csv(output / "val_gated_predictions.csv", index=False, encoding="utf-8-sig")
    patient.to_csv(output / "val_patient_gated_predictions.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output / "stratified_diagnostics.csv", index=False, encoding="utf-8-sig")

    result = {
        "protocol": "M3c-A冻结分类概率门控；仅内部val；零训练；不读取test/external。",
        "seed": seed,
        "gate_threshold": gate_threshold,
        "m1_localization_threshold": localization_threshold,
        "image_gate_metrics": image_metrics,
        "image_m1_localization_baseline": baseline_metrics,
        "fp_drop_vs_m1": fp_drop,
        "patient_gate_metrics": patient_gate_metrics,
        "patient_m1_localization_baseline": patient_baseline_metrics,
        "patient_bootstrap": bootstrap,
        "geometry_all_valid_cancer": geometry_all,
        "geometry_displayed_valid_cancer": geometry_displayed,
        "coordinate_max_abs_diff": coordinate_max_abs_diff,
        "gates": {
            "operation_point_recall_at_least_0_90": operation_point_passed,
            "fp_drop_at_least_0_10": fp_gate_passed,
            "resolution_groups_within_overall_minus_0_05": group_gate_passed,
            "passed_seed_success": seed_passed,
        },
        "inputs": {
            "m1_product_dir": str(source.resolve()),
            "val_predictions": str(prediction_path.resolve()),
            "val_predictions_sha256": coordinate_checksum_before,
            "m1_config": str(config_path.resolve()),
            "m1_config_sha256": file_sha256(config_path),
            "m1_checkpoint": str(checkpoint_path.resolve()),
            "m1_checkpoint_sha256": file_sha256(checkpoint_path),
            "manifest": config["manifest"],
            "manifest_sha256": config["manifest_sha256"],
        },
        "test_evaluated": False,
        "external_evaluated": False,
    }
    (output / "config.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(
        f"seed{seed}: t={gate_threshold:.4f} 癌召回={image_metrics['cancer_recall']:.4f} "
        f"非癌FP={image_metrics['noncancer_fp']:.4f} FP下降={fp_drop:.4f} "
        f"分组通过={group_gate_passed} seed成功={seed_passed}"
    )
    return {
        "seed": seed,
        "gate_threshold": gate_threshold,
        "cancer_recall": image_metrics["cancer_recall"],
        "noncancer_fp": image_metrics["noncancer_fp"],
        "m1_noncancer_fp": baseline_metrics["noncancer_fp"],
        "fp_drop": fp_drop,
        "group_gate_passed": group_gate_passed,
        "passed_seed_success": seed_passed,
        "output_dir": str(output.resolve()),
    }


def self_test():
    """覆盖阈值重复值、分辨率合并和离散分组门槛。"""
    values = np.arange(10, dtype=float)
    threshold = lock_recall_threshold(values, 0.90)
    if float(np.mean(values >= threshold)) != 0.90:
        raise AssertionError("10例阈值召回不是0.90")
    frame = pd.DataFrame({
        "label": [1] * 30,
        "size_group": ["large_1025_1600"] * 15
        + ["medium_641_1024"] * 4 + ["small_le640"] * 11,
        "gate_display": [True] * 13 + [False] * 2 + [True] * 13 + [False] * 2,
    })
    frame = add_resolution_gate_group(frame)
    groups = resolution_diagnostics(frame, overall_recall=26 / 30)
    if set(groups.group) != {"large", "non_large"}:
        raise AssertionError("分辨率合并错误")
    if not groups.passed.all():
        raise AssertionError("13/15不应低于总体26/30减0.05")
    print("M3c-A自测通过：阈值、分辨率合并和离散门槛正确")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"seeds只能来自{SEEDS}")
    if args.bootstrap <= 0:
        raise ValueError("bootstrap必须为正整数")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_seed(seed, args) for seed in seeds]
    summary = pd.DataFrame(rows)
    summary.to_csv(
        args.output_root / "m3c_balanced_summary.csv", index=False, encoding="utf-8-sig"
    )
    matrix_passed = bool(len(rows) == len(SEEDS) and summary.passed_seed_success.all())
    summary_json = {
        "protocol": "M3c-A balanced三种子冻结val门控汇总",
        "requested_seeds": list(seeds),
        "required_seeds": list(SEEDS),
        "passed_seed_count": int(summary.passed_seed_success.sum()),
        "required_seed_count": len(SEEDS),
        "passed_matrix_success": matrix_passed,
        "test_evaluated": False,
        "external_evaluated": False,
        "rows": rows,
    }
    (args.output_root / "m3c_balanced_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), args.output_root / "source_entry.py")
    print(
        f"M3c-A balanced汇总: {summary.passed_seed_success.sum()}/{len(summary)} seed通过; "
        f"矩阵成功={matrix_passed}; 输出={args.output_root}"
    )


if __name__ == "__main__":
    main()
