#!/usr/bin/env python3
"""冻结M5c-A内部test协议：公式、阈值、产物SHA与确认判据。

该脚本只读取已完成的train/val M5c/M5b产物和原始split清单，不读取test预测结果。
输出JSON一旦生成默认拒绝覆盖，作为后续test ROI构建和评价的唯一协议来源。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from train_utils import file_sha256, git_snapshot, json_ready  # noqa: E402

M5C_ROOT = PROJECT_ROOT / "结果/M5c概率融合_0804/正式验证集筛选"
M5B_ROOT = PROJECT_ROOT / "结果/M5b残差融合_0804/正式验证集筛选"
M5_ROI_ROOT = PROJECT_ROOT / "结果/M5预测ROI融合_0804/冻结ROI清单"
MANIFEST = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "08_M1辅助定位清单_20260810"
    / "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "结果/M5c概率融合_0804/冻结内部test协议"
    / "m5c_internal_test_protocol.json"
)
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def lock_recall_threshold(values, labels, recall=0.90):
    """返回实际候选概率中满足Sensitivity>=recall的最高阈值。"""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positive = values[labels == 1]
    if len(positive) == 0:
        raise ValueError("阈值冻结数据没有癌患者")
    for threshold in np.sort(np.unique(positive))[::-1]:
        if float(np.mean(positive >= threshold)) >= recall:
            return float(threshold)
    return float(positive.min())


def self_test():
    values = np.arange(10, dtype=float) / 10
    labels = np.ones(10, dtype=int)
    threshold = lock_recall_threshold(values, labels, 0.90)
    if float(np.mean(values >= threshold)) != 0.90:
        raise AssertionError("Sensitivity>=0.90阈值冻结错误")


def main():
    """校验三seed正式val产物并写出不可变内部test协议。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M5c test协议自测通过: Sensitivity阈值精确冻结")
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"冻结协议已存在，拒绝覆盖: {args.output}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    manifest = pd.read_csv(
        args.manifest, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    test = manifest.loc[manifest.split.eq("test")]
    if len(test) != 198 or test.patient_id.nunique() != 153:
        raise ValueError(
            f"内部test规模异常: {len(test)}张/{test.patient_id.nunique()}人"
        )

    seed_records = {}
    for seed in SEEDS:
        m5c_run = M5C_ROOT / f"m5c_balanced_keep_efficientnet_b0_seed{seed}"
        m5b_run = M5B_ROOT / f"m5b_gate_local_balanced_keep_efficientnet_b0_seed{seed}"
        m5_roi_config = M5_ROI_ROOT / f"m5_roi_config_seed{seed}.json"
        required = [
            m5c_run / "config.json", m5c_run / "val_patient_predictions.csv",
            m5b_run / "config.json", m5b_run / "m5b_best.pth",
            m5b_run / "val_image_predictions.csv", m5_roi_config,
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        m5c_config = json.loads(required[0].read_text(encoding="utf-8"))
        m5b_config = json.loads(required[2].read_text(encoding="utf-8"))
        roi_config = json.loads(m5_roi_config.read_text(encoding="utf-8"))
        if m5c_config.get("primary_method") != "A_fixed_mean_primary":
            raise ValueError(f"seed{seed} M5c主方法不是固定均值")
        if m5b_config.get("alpha_mode") != "gate_local":
            raise ValueError(f"seed{seed}冻结局部头不是M5b-B")
        if m5c_config.get("test_evaluated") or m5c_config.get("external_evaluated"):
            raise ValueError(f"seed{seed} M5c来源已触碰test/external")

        patients = pd.read_csv(
            required[1], encoding="utf-8-sig", dtype={"patient_id": str}
        )
        images = pd.read_csv(
            required[4], encoding="utf-8-sig", dtype={"patient_id": str}
        )
        images["p_fixed_mean_image"] = 0.5 * (images.p_global + images.p_local)
        seed_records[str(seed)] = {
            "m5c_run": str(m5c_run.resolve()),
            "m5c_config_sha256": file_sha256(m5c_run / "config.json"),
            "m5b_run": str(m5b_run.resolve()),
            "m5b_checkpoint": str((m5b_run / "m5b_best.pth").resolve()),
            "m5b_checkpoint_sha256": file_sha256(m5b_run / "m5b_best.pth"),
            "m1_checkpoint": roi_config["m1_checkpoint"],
            "m1_checkpoint_sha256": roi_config["m1_checkpoint_sha256"],
            "frozen_train_val_roi_config": str(m5_roi_config.resolve()),
            "frozen_train_val_roi_config_sha256": file_sha256(m5_roi_config),
            "fallback_width": float(roi_config["fallback_width"]),
            "fallback_height": float(roi_config["fallback_height"]),
            "patient_threshold_sens90": lock_recall_threshold(
                patients.p_fixed_mean, patients.label,
            ),
            "global_patient_threshold_sens90": lock_recall_threshold(
                patients.p_global, patients.label,
            ),
            "image_threshold_sens90_auxiliary": lock_recall_threshold(
                images.p_fixed_mean_image, images.label,
            ),
            "val_global_patient_auc": float(m5c_config["global_patient_auc"]),
            "val_m5c_patient_auc": float(
                m5c_config["methods"]["A_fixed_mean_primary"]["patient_auc"]
            ),
        }

    protocol = {
        "protocol_name": "M5c-A locked internal test projection",
        "origin": "post_hoc_hypothesis_frozen_after_m5c_val_success",
        "primary_method": "fixed_patient_probability_mean",
        "branch_weights": {"global": 0.5, "local": 0.5},
        "patient_aggregation": "top2_mean_each_branch_before_fusion",
        "roi_source": "frozen_M1_top1_predicted_box_only",
        "roi_margin": 0.20,
        "roi_resize": [224, 224],
        "inference_batch_size": 32,
        "uses_gate_or_q_region": False,
        "uses_test_gt_for_roi": False,
        "seeds": list(SEEDS),
        "seed_records": seed_records,
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": file_sha256(args.manifest),
        "test_cohort": {
            "n_images": int(len(test)),
            "n_patients": int(test.patient_id.nunique()),
            "image_labels": {
                str(k): int(v) for k, v in test.label.value_counts().sort_index().items()
            },
            "patient_labels": {
                str(k): int(v) for k, v in
                test[["patient_id", "label"]].drop_duplicates().label
                .value_counts().sort_index().items()
            },
        },
        "test_confirmation_rule": {
            "confirmed": "3/3 seed delta_auc>0 and mean_delta_auc>=0.01",
            "suggestive": "3/3 seed delta_auc>0 and 0.005<=mean_delta_auc<0.01",
            "not_confirmed": "otherwise",
        },
        "forbidden_after_freeze": [
            "change_branch_weights", "change_patient_aggregation",
            "change_thresholds_from_test", "use_test_gt_box_for_roi",
            "select_seed_or_method_by_test", "replace_failed_roi_manually",
        ],
        "test_evaluated_at_freeze": False,
        "external_evaluated_at_freeze": False,
        **git_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_ready(protocol), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"M5c内部test协议已冻结: {args.output}")
    for seed, record in seed_records.items():
        print(
            f"seed{seed}: patient阈值={record['patient_threshold_sens90']:.6f}, "
            f"global阈值={record['global_patient_threshold_sens90']:.6f}"
        )


if __name__ == "__main__":
    main()
