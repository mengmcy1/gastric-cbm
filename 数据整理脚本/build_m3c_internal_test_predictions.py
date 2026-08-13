#!/usr/bin/env python3
"""以batch32确定性前向生成M3c-B内部test的M1框与q_region。

推理只读取冻结test队列中的图像和元数据。真值bbox在送入Dataset前被屏蔽，不参与M1框、
q_region或门控决策；正式指标由后续锁定评价脚本单独计算。
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

from efficientnet_m1_localization import M1Dataset, EfficientNetM1  # noqa: E402
from efficientnet_m3c_region_gate import FrozenM1RegionGate  # noqa: E402
from train_utils import file_sha256, json_ready  # noqa: E402

DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "结果/M3c区域门控_0804/冻结内部test协议"
    / "m3c_internal_test_protocol.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3c区域门控_0804/冻结内部test预测"
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test():
    probabilities = torch.sigmoid(torch.tensor([-2.0, 0.0, 2.0]))
    if not torch.all((probabilities > 0) & (probabilities < 1)):
        raise AssertionError("q_region概率范围错误")


def main():
    """校验冻结链，加载M1+M3c-B并按队列顺序保存test预测。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M3c-B test预测构建器自测通过")
        return
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.batch_size != int(protocol["inference_batch_size"]):
        raise ValueError("M3c-B test推理batch size必须等于冻结协议的32")
    queue_path = Path(protocol["test_queue"])
    if file_sha256(queue_path) != protocol["test_queue_sha256"]:
        raise ValueError("冻结test队列SHA不一致")
    record = protocol["seed_records"][str(args.seed)]
    m1_checkpoint = Path(record["m1_checkpoint"])
    m3c_checkpoint = Path(record["m3c_checkpoint"])
    if file_sha256(m1_checkpoint) != record["m1_checkpoint_sha256"]:
        raise ValueError("冻结M1 checkpoint SHA不一致")
    if file_sha256(m3c_checkpoint) != record["m3c_checkpoint_sha256"]:
        raise ValueError("冻结M3c-B checkpoint SHA不一致")

    target_csv = args.output_root / f"m3c_internal_test_predictions_seed{args.seed}.csv"
    target_config = args.output_root / f"m3c_internal_test_predictions_seed{args.seed}.json"
    for target in (target_csv, target_config):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"M3c-B test预测已存在，拒绝覆盖: {target}")

    queue = pd.read_csv(queue_path, encoding="utf-8-sig", dtype={"patient_id": str})
    cohort = protocol["test_cohort"]
    if len(queue) != cohort["n_images"] or queue.patient_id.nunique() != cohort["n_patients"]:
        raise ValueError("test队列规模与冻结协议不一致")
    missing_paths = [
        value for value in queue.image_relpath.astype(str)
        if not (args.image_root / value).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"test缺失{len(missing_paths)}张图: {missing_paths[0]}")

    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    m3c_payload = torch.load(m3c_checkpoint, map_location="cpu", weights_only=False)
    m1 = EfficientNetM1()
    m1.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model = FrozenM1RegionGate(m1)
    model.gate.load_state_dict(m3c_payload["gate_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 屏蔽test真值框，确保预测路径无法读取定位监督。
    inference = queue.copy()
    inference["localization_supervision"] = 0
    for column in ("bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"):
        inference[column] = 0.0
    dataset = M1Dataset(inference, args.image_root, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    rows = []
    box_repeat_diff = 0.0
    q_repeat_diff = 0.0
    m1_threshold = float(record["m1_localization_threshold"])
    gate_threshold = float(record["m3c_gate_threshold"])
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            repeated = model(images)
            q_region = torch.sigmoid(outputs["gate_logits"])
            q_repeat = torch.sigmoid(repeated["gate_logits"])
            box_repeat_diff = max(
                box_repeat_diff,
                float((outputs["boxes"] - repeated["boxes"]).abs().max().item()),
            )
            q_repeat_diff = max(
                q_repeat_diff, float((q_region - q_repeat).abs().max().item())
            )
            for local, row_index in enumerate(batch["row_index"].tolist()):
                meta = queue.iloc[row_index]
                box = outputs["boxes"][local].cpu().tolist()
                loc = float(outputs["localization_confidence"][local].cpu())
                q_value = float(q_region[local].cpu())
                row = {
                    "split": "test", "image_relpath": meta.image_relpath,
                    "patient_id": meta.patient_id,
                    "width": int(meta.width), "height": int(meta.height),
                    "localization_confidence": loc, "q_region": q_value,
                    "m1_display": int(loc >= m1_threshold),
                    "m3c_display": int(q_value >= gate_threshold),
                    "cancer_probability": float(outputs["cancer_probability"][local].cpu()),
                }
                for name, value in zip(("x1", "y1", "x2", "y2"), box):
                    row[f"m1_pred_bbox_{name}"] = float(value)
                for column in ("source", "center", "size_group", "aspect_group", "frame_profile"):
                    if column in queue.columns:
                        row[column] = meta[column]
                rows.append(row)
    if box_repeat_diff > 1e-6 or q_repeat_diff > 1e-6:
        raise RuntimeError(
            f"确定性重复推理失败: box={box_repeat_diff:.3e}, q_region={q_repeat_diff:.3e}"
        )

    predictions = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(target_csv, index=False, encoding="utf-8-sig")
    config = {
        "seed": args.seed,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": file_sha256(args.protocol),
        "test_queue": str(queue_path.resolve()),
        "test_queue_sha256": file_sha256(queue_path),
        "m1_checkpoint": str(m1_checkpoint.resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "m3c_checkpoint": str(m3c_checkpoint.resolve()),
        "m3c_checkpoint_sha256": file_sha256(m3c_checkpoint),
        "m1_localization_threshold": m1_threshold,
        "m3c_gate_threshold": gate_threshold,
        "inference_batch_size": args.batch_size,
        "prediction_csv_sha256": file_sha256(target_csv),
        "n_images": int(len(predictions)),
        "n_patients": int(predictions.patient_id.nunique()),
        "repeat_box_max_abs_diff": box_repeat_diff,
        "repeat_q_region_max_abs_diff": q_repeat_diff,
        "test_gt_used_for_prediction": False,
        "test_labels_exported_with_predictions": False,
        "test_metrics_evaluated": False,
        "external_evaluated": False,
    }
    target_config.write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"seed={args.seed}: {len(predictions)}张/{predictions.patient_id.nunique()}人; "
        f"M1框+q_region已生成（未评价test指标）"
    )
    print(f"输出: {target_csv}")


if __name__ == "__main__":
    main()
