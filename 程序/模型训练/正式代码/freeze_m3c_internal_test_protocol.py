#!/usr/bin/env python3
"""冻结M3c-B内部test协议、输入队列、模型SHA与确认判据。

本脚本只读取正式val配置和split清单，不读取或生成test模型预测。冻结后，test推理与
评价必须引用本协议；门控阈值、M1定位阈值和成功判据不得根据test结果调整。
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from train_utils import file_sha256, git_snapshot, json_ready  # noqa: E402

M3C_ROOT = PROJECT_ROOT / "结果/M3c区域门控_0804/正式验证集筛选"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "08_M1辅助定位清单_20260810"
    / "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3c区域门控_0804/冻结内部test协议"
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """校验三seed正式产品，冻结test队列和唯一评价口径。"""
    args = parse_args()
    protocol_path = args.output_root / "m3c_internal_test_protocol.json"
    queue_path = args.output_root / "m3c_internal_test_queue.csv"
    for target in (protocol_path, queue_path):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"冻结产物已存在，拒绝覆盖: {target}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    manifest = pd.read_csv(
        args.manifest, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    required = {
        "split", "image_relpath", "patient_id", "label", "size_group",
        "localization_supervision", "bbox_x1_norm", "bbox_y1_norm",
        "bbox_x2_norm", "bbox_y2_norm",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"M3c test队列缺少字段: {sorted(missing)}")
    test = manifest.loc[manifest.split.eq("test")].copy()
    if len(test) != 198 or test.patient_id.nunique() != 153:
        raise ValueError(f"内部test规模异常: {len(test)}张/{test.patient_id.nunique()}人")
    if test.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("内部test患者跨标签")
    cancer = test.label.eq(1)
    if not test.loc[cancer, "localization_supervision"].eq(1).all():
        raise ValueError("内部test癌图bbox监督不完整")

    queue_columns = [
        "split", "image_relpath", "patient_id", "label", "width", "height",
        "size_group", "localization_supervision", "bbox_x1_norm",
        "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
    ]
    queue_columns += [
        column for column in ("source", "center", "aspect_group", "frame_profile")
        if column in test.columns
    ]
    queue = test[queue_columns].reset_index(drop=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    queue.to_csv(queue_path, index=False, encoding="utf-8-sig")

    records = {}
    for seed in SEEDS:
        run = M3C_ROOT / f"m3c_balanced_keep_efficientnet_b0_seed{seed}"
        config_path = run / "config.json"
        checkpoint = run / "m3c_gate_best.pth"
        for path in (config_path, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        selection = config.get("m3c_selection", {})
        if config.get("status") != "success" or not selection.get("passed_seed_success"):
            raise ValueError(f"seed{seed}不是正式成功的M3c-B产品")
        if config.get("debug") or config.get("test_evaluated") \
                or config.get("external_evaluated"):
            raise ValueError(f"seed{seed}来源不是未触碰test/external的正式产品")
        if file_sha256(args.manifest) != config["manifest_sha256"]:
            raise ValueError(f"seed{seed} M3c-B训练清单与test队列不同源")
        summary = selection["final_summary"]
        m1_checkpoint = Path(config["m1_checkpoint"])
        if file_sha256(m1_checkpoint) != config["m1_checkpoint_sha256"]:
            raise ValueError(f"seed{seed} M1 checkpoint SHA不一致")
        records[str(seed)] = {
            "m1_checkpoint": str(m1_checkpoint.resolve()),
            "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
            "m3c_checkpoint": str(checkpoint.resolve()),
            "m3c_checkpoint_sha256": file_sha256(checkpoint),
            "m3c_config": str(config_path.resolve()),
            "m3c_config_sha256": file_sha256(config_path),
            "m1_localization_threshold": float(config["baseline"]["localization_threshold"]),
            "m3c_gate_threshold": float(summary["gate_threshold"]),
            "val_m1_noncancer_fp": float(config["baseline"]["image_metrics"]["noncancer_fp"]),
            "val_m3c_noncancer_fp": float(summary["noncancer_fp"]),
            "val_m3c_cancer_recall": float(summary["cancer_recall"]),
        }

    protocol = {
        "protocol_name": "M3c-B locked internal test projection",
        "role": "frozen_M1_box_plus_frozen_M3c_region_display_gate",
        "seeds": list(SEEDS),
        "inference_batch_size": 32,
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": file_sha256(args.manifest),
        "test_queue": str(queue_path.resolve()),
        "test_queue_sha256": file_sha256(queue_path),
        "test_cohort": {
            "n_images": int(len(queue)),
            "n_patients": int(queue.patient_id.nunique()),
            "image_labels": {
                str(k): int(v) for k, v in queue.label.value_counts().sort_index().items()
            },
            "patient_labels": {
                str(k): int(v) for k, v in
                queue[["patient_id", "label"]].drop_duplicates().label
                .value_counts().sort_index().items()
            },
            "cancer_bbox_images": int(
                (queue.label.eq(1) & queue.localization_supervision.eq(1)).sum()
            ),
        },
        "seed_records": records,
        "locked_confirmation_rule": {
            "per_seed_cancer_recall": "M3c recall >= max(0.85, M1 recall - 0.05)",
            "per_seed_fp_drop": "M1 noncancer FP - M3c noncancer FP >= 0.10",
            "resolution_groups": "group cancer n>=15: M3c detected >= M1 detected - 1",
            "overall": "all three seeds pass every applicable rule",
            "geometry": "M3c cannot alter M1 boxes; IoU/center-hit are diagnostic only",
        },
        "forbidden_after_freeze": [
            "change_m1_or_m3c_checkpoint", "change_gate_or_localization_threshold",
            "change_test_queue", "select_seed_by_test", "tune_from_test",
            "use_test_bbox_during_prediction", "replace_prediction_manually",
        ],
        "test_predictions_generated_at_freeze": False,
        "test_metrics_evaluated_at_freeze": False,
        "external_evaluated_at_freeze": False,
        **git_snapshot(),
    }
    protocol_path.write_text(
        json.dumps(json_ready(protocol), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"M3c-B内部test协议已冻结: {protocol_path}")
    print(f"test队列: {len(queue)}张/{queue.patient_id.nunique()}人; SHA={file_sha256(queue_path)}")
    for seed, record in records.items():
        print(
            f"seed{seed}: M1阈值={record['m1_localization_threshold']:.6f}, "
            f"M3c阈值={record['m3c_gate_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
