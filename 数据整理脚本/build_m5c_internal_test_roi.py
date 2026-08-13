#!/usr/bin/env python3
"""用冻结M1 Top-1预测框构建M5c内部test ROI清单。

ROI几何仅由图像、冻结M1输出和协议中的train回退尺寸决定。test标签只原样携带供后续
评价，不参与预测框、回退、扩边或裁剪；医生真值框不会写入输出清单。
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(MODEL_DIR))

from efficientnet_m1_localization import (  # noqa: E402
    M1Dataset, EfficientNetM1, decode_boxes,
)
from train_utils import file_sha256, json_ready  # noqa: E402
from build_m5_roi_manifest import box_area, expand_and_square  # noqa: E402

DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "结果/M5c概率融合_0804/冻结内部test协议"
    / "m5c_internal_test_protocol.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M5c概率融合_0804/冻结内部test_ROI清单"
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test():
    crop = expand_and_square((10, 20, 30, 40), 0.20, 100, 80)
    if not (0 <= crop[0] < crop[2] <= 100 and 0 <= crop[1] < crop[3] <= 80):
        raise AssertionError("test ROI扩边框越界")


def main():
    """加载协议与M1，对test确定性推理并保存无真值几何的ROI清单。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M5c test ROI自测通过: 扩边框有效")
        return
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.batch_size != int(protocol["inference_batch_size"]):
        raise ValueError("test ROI batch size必须与冻结协议一致")
    record = protocol["seed_records"][str(args.seed)]
    manifest_path = Path(protocol["input_manifest"])
    m1_checkpoint = Path(record["m1_checkpoint"])
    if file_sha256(manifest_path) != protocol["input_manifest_sha256"]:
        raise ValueError("输入split清单SHA与冻结协议不一致")
    if file_sha256(m1_checkpoint) != record["m1_checkpoint_sha256"]:
        raise ValueError("M1 checkpoint SHA与冻结协议不一致")

    suffix = "_debug" if args.debug else ""
    target_csv = args.output_root / f"m5c_internal_test_roi_seed{args.seed}{suffix}.csv"
    target_config = args.output_root / f"m5c_internal_test_roi_seed{args.seed}{suffix}.json"
    for target in (target_csv, target_config):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"test ROI产物已存在，拒绝覆盖: {target}")

    frame = pd.read_csv(
        manifest_path, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    frame = frame.loc[frame.split.eq("test")].reset_index(drop=True)
    if args.debug:
        patients = frame[["patient_id", "label"]].drop_duplicates()
        selected = pd.concat([
            group.sample(min(args.debug_units, len(group)), random_state=args.seed)
            for _, group in patients.groupby("label", sort=True)
        ])
        frame = frame.loc[frame.patient_id.isin(selected.patient_id)].reset_index(drop=True)
    elif len(frame) != protocol["test_cohort"]["n_images"] \
            or frame.patient_id.nunique() != protocol["test_cohort"]["n_patients"]:
        raise ValueError("内部test规模与冻结协议不一致")

    payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    model = EfficientNetM1()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 推理副本显式屏蔽定位监督，使Dataset根本不读取或变换test真值框。
    inference_frame = frame.copy()
    inference_frame["localization_supervision"] = 0
    dataset = M1Dataset(inference_frame, args.image_root, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    rows = []
    repeat_box_max_abs_diff = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            repeated = model(images)
            boxes, confidence = decode_boxes(outputs)
            boxes_repeat, _ = decode_boxes(repeated)
            repeat_box_max_abs_diff = max(
                repeat_box_max_abs_diff,
                float((boxes - boxes_repeat).abs().max().item()),
            )
            boxes = boxes.cpu().tolist()
            confidence = confidence.cpu().tolist()
            for local, row_index in enumerate(batch["row_index"].tolist()):
                meta = frame.iloc[row_index]
                base = tuple(float(value) for value in boxes[local])
                source = "predicted"
                if box_area(base) <= 1e-8:
                    width = float(record["fallback_width"])
                    height = float(record["fallback_height"])
                    base = (0.5 - width / 2, 0.5 - height / 2,
                            0.5 + width / 2, 0.5 + height / 2)
                    source = "fallback"
                crop_px = expand_and_square(
                    (base[0] * meta.width, base[1] * meta.height,
                     base[2] * meta.width, base[3] * meta.height),
                    float(protocol["roi_margin"]), int(meta.width), int(meta.height),
                )
                crop = (
                    crop_px[0] / meta.width, crop_px[1] / meta.height,
                    crop_px[2] / meta.width, crop_px[3] / meta.height,
                )
                if box_area(crop) <= 1e-8:
                    raise ValueError(f"test ROI裁剪框退化: {meta.image_relpath}")
                row = {
                    "split": "test", "image_relpath": meta.image_relpath,
                    "patient_id": meta.patient_id, "label": int(meta.label),
                    "width": int(meta.width), "height": int(meta.height),
                    "roi_source": source,
                    "localization_confidence": float(confidence[local]),
                    "roi_label": 2, "gate_pass": 1,
                    "noncancer_stratum": "", "q_region": float("nan"),
                    "bbox_area_fraction": float("nan"),
                }
                for prefix, values in (("m1_pred_bbox", boxes[local]),
                                       ("roi_crop_bbox", crop)):
                    for name, value in zip(("x1", "y1", "x2", "y2"), values):
                        row[f"{prefix}_{name}"] = float(value)
                for column in ("source", "center", "size_group", "aspect_group",
                               "frame_profile"):
                    if column in frame.columns:
                        row[column] = meta[column]
                rows.append(row)
    if repeat_box_max_abs_diff > 1e-6:
        raise RuntimeError(f"M1重复推理框差异过大: {repeat_box_max_abs_diff}")

    predictions = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(target_csv, index=False, encoding="utf-8-sig")
    config = {
        "seed": args.seed, "debug": bool(args.debug),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": file_sha256(args.protocol),
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": file_sha256(manifest_path),
        "m1_checkpoint": str(m1_checkpoint.resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "roi_csv_sha256": file_sha256(target_csv),
        "n_images": int(len(predictions)),
        "n_patients": int(predictions.patient_id.nunique()),
        "roi_source_counts": predictions.roi_source.value_counts().to_dict(),
        "margin": float(protocol["roi_margin"]),
        "inference_batch_size": args.batch_size,
        "repeat_box_max_abs_diff": repeat_box_max_abs_diff,
        "labels_used_for_roi_generation": False,
        "test_gt_used_for_roi_generation": False,
        "test_metrics_evaluated": False,
        "external_evaluated": False,
    }
    target_config.write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"seed={args.seed} test ROI: {len(predictions)}张/"
        f"{predictions.patient_id.nunique()}人; 来源={config['roi_source_counts']}"
    )
    print(f"输出: {target_csv}")


if __name__ == "__main__":
    main()
