#!/usr/bin/env python3
"""按冻结协议一次性评价M1+M3c-B内部test，不允许从test重选阈值。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from train_utils import file_sha256, git_snapshot, json_ready  # noqa: E402

DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "结果/M3c区域门控_0804/冻结内部test协议"
    / "m3c_internal_test_protocol.json"
)
PREDICTION_ROOT = PROJECT_ROOT / "结果/M3c区域门控_0804/冻结内部test预测"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3c区域门控_0804/锁定内部test评估"
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def rate_metrics(labels, decisions):
    """计算癌召回和非癌FP的离散计数。"""
    labels = np.asarray(labels, dtype=int)
    decisions = np.asarray(decisions, dtype=bool)
    cancer = labels == 1
    noncancer = labels == 0
    return {
        "cancer_count": int(cancer.sum()),
        "noncancer_count": int(noncancer.sum()),
        "cancer_detected": int((cancer & decisions).sum()),
        "noncancer_false_positive": int((noncancer & decisions).sum()),
        "cancer_recall": float(decisions[cancer].mean()),
        "noncancer_fp": float(decisions[noncancer].mean()),
    }


def geometry_metrics(frame):
    """在有真值框的癌图上计算M1框IoU、中心命中和病灶覆盖率。"""
    cancer = frame.loc[frame.label.eq(1) & frame.localization_supervision.eq(1)].copy()
    px1, py1 = cancer.m1_pred_bbox_x1, cancer.m1_pred_bbox_y1
    px2, py2 = cancer.m1_pred_bbox_x2, cancer.m1_pred_bbox_y2
    gx1, gy1 = cancer.bbox_x1_norm, cancer.bbox_y1_norm
    gx2, gy2 = cancer.bbox_x2_norm, cancer.bbox_y2_norm
    ix1, iy1 = np.maximum(px1, gx1), np.maximum(py1, gy1)
    ix2, iy2 = np.minimum(px2, gx2), np.minimum(py2, gy2)
    intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    pred_area = np.maximum(0, px2 - px1) * np.maximum(0, py2 - py1)
    gt_area = np.maximum(0, gx2 - gx1) * np.maximum(0, gy2 - gy1)
    union = pred_area + gt_area - intersection
    iou = np.divide(intersection, union, out=np.zeros(len(cancer)), where=union > 0)
    coverage = np.divide(
        intersection, gt_area, out=np.zeros(len(cancer)), where=gt_area > 0
    )
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    center_hit = cx.between(gx1, gx2) & cy.between(gy1, gy2)
    return {
        "count": int(len(cancer)),
        "mean_iou": float(iou.mean()),
        "median_iou": float(np.median(iou)),
        "iou_ge_0_3": float((iou >= 0.3).mean()),
        "iou_ge_0_5": float((iou >= 0.5).mean()),
        "center_hit_rate": float(center_hit.mean()),
        "mean_lesion_coverage": float(coverage.mean()),
    }


def patient_metrics(frame, decision_column):
    """任一图展示框即记为患者级展示阳性。"""
    patient = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), decision=(decision_column, "max")
    )
    return rate_metrics(patient.label, patient.decision), patient


def patient_bootstrap(frame, iterations, seed):
    """按患者标签分层有放回估计M3c癌召回、非癌FP与FP下降区间。"""
    patients = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), m1_display=("m1_display", "max"),
        m3c_display=("m3c_display", "max"),
    )
    groups = {
        label: patients.loc[patients.label.eq(label)].reset_index(drop=True)
        for label in (0, 1)
    }
    rng = np.random.default_rng(seed)
    recall, fp, fp_drop = [], [], []
    for _ in range(iterations):
        sampled = {
            label: group.iloc[rng.integers(0, len(group), size=len(group))]
            for label, group in groups.items()
        }
        recall.append(float(sampled[1].m3c_display.mean()))
        fp.append(float(sampled[0].m3c_display.mean()))
        fp_drop.append(float(sampled[0].m1_display.mean() - sampled[0].m3c_display.mean()))
    return {
        "iterations": iterations,
        "cancer_recall_ci95": np.quantile(recall, [0.025, 0.975]).tolist(),
        "noncancer_fp_ci95": np.quantile(fp, [0.025, 0.975]).tolist(),
        "fp_drop_ci95": np.quantile(fp_drop, [0.025, 0.975]).tolist(),
    }


def resolution_report(frame):
    """按冻结large/non_large定义比较M1与M3c癌检出计数。"""
    cancer = frame.loc[frame.label.eq(1)].copy()
    cancer["resolution_group"] = cancer.size_group.map({
        "large_1025_1600": "large",
        "medium_641_1024": "non_large",
        "small_le640": "non_large",
    })
    report = {}
    for name, group in cancer.dropna(subset=["resolution_group"]).groupby("resolution_group"):
        n = int(len(group))
        m1_detected = int(group.m1_display.sum())
        m3c_detected = int(group.m3c_display.sum())
        participates = n >= 15
        report[str(name)] = {
            "n_cancer": n, "m1_detected": m1_detected,
            "m3c_detected": m3c_detected, "participates_in_gate": participates,
            "passed": bool(not participates or m3c_detected >= m1_detected - 1),
        }
    return report


def self_test():
    metrics = rate_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    if metrics["cancer_recall"] != 0.5 or metrics["noncancer_fp"] != 0.5:
        raise AssertionError("M3c test计数指标错误")


def main():
    """合并冻结预测与真值，计算单seed锁定test结论并保存审计产物。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M3c-B锁定test评价自测通过")
        return
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    record = protocol["seed_records"][str(args.seed)]
    prediction_path = args.predictions or (
        PREDICTION_ROOT / f"m3c_internal_test_predictions_seed{args.seed}.csv"
    )
    prediction_config_path = prediction_path.with_suffix(".json")
    for path in (prediction_path, prediction_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    prediction_config = json.loads(prediction_config_path.read_text(encoding="utf-8"))
    if prediction_config["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("M3c test预测与冻结协议不同源")
    if prediction_config["prediction_csv_sha256"] != file_sha256(prediction_path):
        raise ValueError("M3c test预测CSV SHA不一致")

    predictions = pd.read_csv(
        prediction_path, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    queue_path = Path(protocol["test_queue"])
    queue = pd.read_csv(queue_path, encoding="utf-8-sig", dtype={"patient_id": str})
    join_keys = ["split", "image_relpath", "patient_id", "width", "height"]
    truth_columns = join_keys + [
        "label", "localization_supervision", "bbox_x1_norm", "bbox_y1_norm",
        "bbox_x2_norm", "bbox_y2_norm",
    ]
    frame = predictions.merge(
        queue[truth_columns], on=join_keys, how="inner", validate="one_to_one"
    )
    if len(frame) != protocol["test_cohort"]["n_images"]:
        raise ValueError("预测与冻结test队列未一一匹配")

    image_m1 = rate_metrics(frame.label, frame.m1_display)
    image_m3c = rate_metrics(frame.label, frame.m3c_display)
    patient_m1, _ = patient_metrics(frame, "m1_display")
    patient_m3c, _ = patient_metrics(frame, "m3c_display")
    fp_drop = image_m1["noncancer_fp"] - image_m3c["noncancer_fp"]
    recall_floor = max(0.85, image_m1["cancer_recall"] - 0.05)
    groups = resolution_report(frame)
    checks = {
        "cancer_recall_passed": image_m3c["cancer_recall"] >= recall_floor,
        "fp_drop_passed": fp_drop >= 0.10,
        "resolution_groups_passed": bool(groups) and all(
            item["passed"] for item in groups.values()
        ),
    }
    checks["passed_seed_confirmation"] = all(checks.values())
    summary = {
        "seed": args.seed,
        "m1_localization_threshold": float(record["m1_localization_threshold"]),
        "m3c_gate_threshold": float(record["m3c_gate_threshold"]),
        "q_region_image_auc": float(roc_auc_score(frame.label, frame.q_region)),
        "image_m1": image_m1,
        "image_m3c": image_m3c,
        "image_fp_drop": fp_drop,
        "required_cancer_recall": recall_floor,
        "patient_m1": patient_m1,
        "patient_m3c": patient_m3c,
        "geometry": geometry_metrics(frame),
        "resolution_groups": groups,
        "checks": checks,
        "patient_bootstrap": patient_bootstrap(frame, args.bootstrap, args.seed),
        "test_evaluated": True,
        "external_evaluated": False,
        **git_snapshot(),
    }
    output = args.output_root / f"m3c_locked_internal_test_seed{args.seed}"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"M3c test结果已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "test_predictions_with_truth.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"seed{args.seed}: M1癌召回={image_m1['cancer_recall']:.4f}, "
        f"M3c癌召回={image_m3c['cancer_recall']:.4f}; "
        f"M1非癌FP={image_m1['noncancer_fp']:.4f}, "
        f"M3c非癌FP={image_m3c['noncancer_fp']:.4f}, 下降={fp_drop:.4f}"
    )
    print(f"锁定test单seed通过={checks['passed_seed_confirmation']}; 输出={output}")


if __name__ == "__main__":
    main()
