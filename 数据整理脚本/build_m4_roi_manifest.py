#!/usr/bin/env python3
"""M4冻结ROI清单构建：用冻结M1+M3c-B对train/val确定性推理，输出三套坐标与hard/easy。

按M4修订预注册（2026-08-11）：
- 只为train/val生成；internal test/external不读取图像、不生成预测框、不计算q_region；
- M1与M3c-B使用确定性评估变换（不随机增强）；
- 每seed独立生成，三套坐标分开保存（m1_pred_bbox / roi_base_bbox / roi_crop_bbox）；
- 回退框尺寸=train癌图中M1框命中病灶的预测框宽高中位数，只用train计算并冻结。

输出：`m4_roi_manifest_seed{seed}.csv` 与 `config.json`。
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # 数据整理脚本的上级即项目根
sys.path.insert(0, str(PROJECT_ROOT / "程序/模型训练/正式代码"))

from efficientnet_m1_localization import (  # noqa: E402
    M1Dataset,
    EfficientNetM1,
    IMAGE_SIZE,
)
from efficientnet_m3c_region_gate import FrozenM1RegionGate  # noqa: E402

MANIFEST = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "08_M1辅助定位清单_20260810"
    / "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
)
M1_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
M3C_ROOT = PROJECT_ROOT / "结果/M3c区域门控_0804/正式验证集筛选"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M4真值ROI融合_0804/冻结ROI清单"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--m1-checkpoint", type=Path, default=None)
    parser.add_argument("--m3c-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="运行expand_and_square边界回归自测后退出。")
    return parser.parse_args()


def default_m1_checkpoint(seed):
    return (
        M1_ROOT / f"m1_balanced_keep_efficientnet_b0_seed{seed}_warmup_product"
        / "m1_best_warmup_localization.pth"
    )


def default_m3c_checkpoint(seed):
    return (
        M3C_ROOT / f"m3c_balanced_keep_efficientnet_b0_seed{seed}"
        / "m3c_gate_best.pth"
    )


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def expand_and_square(base_px, margin, img_w, img_h):
    """扩边0.20+正方形化+尽量平移回界，框过大时截断到图像内。

    base_px为像素坐标(x1,y1,x2,y2)。返回在[0,img_w]x[0,img_h]内的像素框；
    当正方形边超过图像尺寸时接受截断，保证框始终有效。
    """
    x1, y1, x2, y2 = base_px
    width, height = x2 - x1, y2 - y1
    ex1, ex2 = x1 - margin * width, x2 + margin * width
    ey1, ey2 = y1 - margin * height, y2 + margin * height
    side = max(ex2 - ex1, ey2 - ey1)
    center_x, center_y = (ex1 + ex2) / 2, (ey1 + ey2) / 2
    half = side / 2
    cx1, cx2 = center_x - half, center_x + half
    cy1, cy2 = center_y - half, center_y + half
    # 尽量平移回界（按像素坐标）。
    if cx1 < 0:
        shift = -cx1; cx1 += shift; cx2 += shift
    if cy1 < 0:
        shift = -cy1; cy1 += shift; cy2 += shift
    if cx2 > img_w:
        shift = cx2 - img_w; cx1 -= shift; cx2 -= shift
    if cy2 > img_h:
        shift = cy2 - img_h; cy1 -= shift; cy2 -= shift
    # 框过大时截断到图像内，保证不越界。
    cx1, cy1 = max(0.0, cx1), max(0.0, cy1)
    cx2, cy2 = min(img_w, cx2), min(img_h, cy2)
    return (cx1, cy1, cx2, cy2)


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center_inside(pred, target):
    pcx = (pred[0] + pred[2]) / 2
    pcy = (pred[1] + pred[3]) / 2
    return (
        target[0] <= pcx <= target[2] and target[1] <= pcy <= target[3]
    )


def self_test():
    """expand_and_square边界回归：角落/触边/超大框都必须返回[0,w]x[0,h]内非退化框。"""
    for base, width, height in [
        ((0.0, 0.0, 0.1, 0.1), 100, 100),
        ((0.9, 0.9, 1.0, 1.0), 100, 100),
        ((0.3, 0.3, 0.7, 0.7), 100, 100),
        ((0.0, 0.0, 1.0, 1.0), 100, 100),
    ]:
        px = (base[0] * width, base[1] * height, base[2] * width, base[3] * height)
        crop = expand_and_square(px, 0.20, width, height)
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise AssertionError(f"expand_and_square越界或退化: {crop}")
    return True


def main():
    """编排ROI清单构建：只对train/val确定性推理，输出三套坐标/hard-easy/病灶分位。

    输入: --seed对应的M1 warmup产品、M3c-B门控、冻结M0 manifest；
    输出: m4_roi_manifest_seed{seed}{_debug}.csv 与对应config（含门控阈值、回退框、
    病灶三分位、SHA）；
    防泄漏边界: 只读train/val；不读取internal test/external；M1/M3c-B用确定性评估变换。
    返回: 无；产物写入--output-root，任一同名CSV/config存在时默认快速失败。
    """
    args = parse_args()
    if args.self_test:
        self_test()
        print("ROI构建自测通过: expand_and_square四类边界均在界且非退化")
        return
    if args.seed is None:
        raise ValueError("非自测模式必须提供--seed")
    m1_checkpoint = args.m1_checkpoint or default_m1_checkpoint(args.seed)
    m3c_checkpoint = args.m3c_checkpoint or default_m3c_checkpoint(args.seed)
    for path in (args.manifest, m1_checkpoint, m3c_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    output = args.output_root
    # debug输出与正式输出严格隔离：debug追加_debug后缀，config记录debug=true。
    # 同时检查目标CSV与config：任一个存在都快速失败，避免config丢失时覆盖旧CSV。
    suffix = "_debug" if args.debug else ""
    target_manifest = output / f"m4_roi_manifest_seed{args.seed}{suffix}.csv"
    target_config = output / f"m4_roi_config_seed{args.seed}{suffix}.json"
    for target in (target_manifest, target_config):
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"该seed目标已存在: {target}；显式--overwrite才允许覆盖"
            )
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"patient_id": str})
    if args.debug:
        selected = []
        for split, split_frame in frame.groupby("split", sort=False):
            patients = split_frame[["patient_id", "label"]].drop_duplicates()
            sampled = pd.concat([
                group.sample(min(args.debug_units, len(group)), random_state=args.seed)
                for _, group in patients.groupby("label", sort=True)
            ])
            selected.append(split_frame.loc[split_frame.patient_id.isin(sampled.patient_id)])
        frame = pd.concat(selected, ignore_index=True)
    frame = frame.loc[frame.split.isin(["train", "val"])].copy()
    print(f"train/val: {frame.patient_id.nunique()}人/{len(frame)}张 "
          f"(debug={args.debug})")

    # 冻结M1 + M3c-B门控；门控阈值从"所用m3c checkpoint同目录"的run config读取，
    # 保证checkpoint与阈值同源，避免"checkpoint A + threshold B"混用。
    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    m3c_payload = torch.load(m3c_checkpoint, map_location="cpu", weights_only=False)
    m3c_config_path = Path(m3c_checkpoint).parent / "config.json"
    if not m3c_config_path.is_file():
        raise FileNotFoundError(m3c_config_path)
    m3c_run_config = json.loads(m3c_config_path.read_text(encoding="utf-8"))
    gate_threshold = float(
        m3c_run_config["m3c_selection"]["final_summary"]["gate_threshold"]
    )
    m1 = EfficientNetM1()
    m1.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model = FrozenM1RegionGate(m1)
    model.gate.load_state_dict(m3c_payload["gate_state_dict"], strict=True)
    model.eval()
    print(f"seed={args.seed} 门控阈值={gate_threshold:.4f}（来源{Path(m3c_checkpoint).parent.name}）")

    # 确定性评估变换推理（train/val均用eval变换，不随机增强）。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = []
    with torch.no_grad():
        for split in ("train", "val"):
            split_frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
            dataset = M1Dataset(split_frame, args.image_root, training=False)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            for batch in loader:
                images = batch["image"].to(device)
                outputs = model(images)
                boxes = outputs["boxes"].cpu().tolist()
                loc_conf = outputs["localization_confidence"].cpu().tolist()
                q_region = torch.sigmoid(outputs["gate_logits"]).cpu().tolist()
                for local, row_index in enumerate(batch["row_index"].tolist()):
                    meta = split_frame.iloc[row_index]
                    row_dict = {
                        "row_index": row_index,
                        "split": split,
                        "image_relpath": meta["image_relpath"],
                        "patient_id": meta["patient_id"],
                        "label": int(meta["label"]),
                        "width": int(meta["width"]),
                        "height": int(meta["height"]),
                        "gt_x1": float(meta["bbox_x1_norm"]),
                        "gt_y1": float(meta["bbox_y1_norm"]),
                        "gt_x2": float(meta["bbox_x2_norm"]),
                        "gt_y2": float(meta["bbox_y2_norm"]),
                        "m1_x1": float(boxes[local][0]),
                        "m1_y1": float(boxes[local][1]),
                        "m1_x2": float(boxes[local][2]),
                        "m1_y2": float(boxes[local][3]),
                        "localization_confidence": float(loc_conf[local]),
                        "q_region": float(q_region[local]),
                    }
                    # 保留来源/中心/分辨率/病灶尺寸字段，供M4分层分析使用。
                    for column in ["source", "center", "size_group", "bbox_area_fraction"]:
                        if column in split_frame.columns:
                            row_dict[column] = meta[column]
                    rows.append(row_dict)

    predictions = pd.DataFrame(rows)
    # 回退框尺寸：只用train癌图中M1框命中病灶的预测框宽高中位数（不得混入val）。
    hit = predictions.loc[
        predictions.split.eq("train") & predictions.label.eq(1) & predictions.apply(
            lambda row: box_center_inside(
                (row.m1_x1, row.m1_y1, row.m1_x2, row.m1_y2),
                (row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2),
            ), axis=1,
        )
    ]
    if len(hit):
        fallback_w = float(np.median(hit.m1_x2 - hit.m1_x1))
        fallback_h = float(np.median(hit.m1_y2 - hit.m1_y1))
    else:
        fallback_w = fallback_h = 0.1
    print(f"回退框尺寸(train命中癌图): {fallback_w:.4f}x{fallback_h:.4f}")

    # 病灶大小三分位边界：只用train癌图bbox_area_fraction，冻结进config供M4分层。
    train_cancer_area = predictions.loc[
        predictions.split.eq("train") & predictions.label.eq(1),
        "bbox_area_fraction",
    ]
    lesion_tercile_bounds = (
        train_cancer_area.quantile([0.33, 0.66]).tolist()
        if len(train_cancer_area) else None
    )
    print(f"病灶大小三分位边界(train癌图): {lesion_tercile_bounds}")

    # 三套坐标与non-cancer分层（hard/easy只对非癌有意义）。
    m1_boxes = []
    base_boxes = []
    crop_boxes = []
    strata = []
    sources = []
    for row in predictions.itertuples(index=False):
        m1_box = (row.m1_x1, row.m1_y1, row.m1_x2, row.m1_y2)
        m1_boxes.append(m1_box)
        if row.label == 1:
            base = (row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2)
            source = "gt"
            stratum = ""
        else:
            base = m1_box
            source = "predicted"
            stratum = "hard" if row.q_region >= gate_threshold else "easy"
        if box_area(base) <= 1e-8:
            # 回退：以base中心（无中心则图中心）放fallback尺寸框。
            cx = (base[0] + base[2]) / 2 if box_area(base) > 0 else 0.5
            cy = (base[1] + base[3]) / 2 if box_area(base) > 0 else 0.5
            base = (cx - fallback_w / 2, cy - fallback_h / 2,
                    cx + fallback_w / 2, cy + fallback_h / 2)
            source = "fallback"
        base_boxes.append(base)
        # 像素坐标扩边+正方形+回界，再归一化。
        crop_px = expand_and_square(
            (base[0] * row.width, base[1] * row.height,
             base[2] * row.width, base[3] * row.height),
            args.margin, row.width, row.height,
        )
        crop_boxes.append((crop_px[0] / row.width, crop_px[1] / row.height,
                           crop_px[2] / row.width, crop_px[3] / row.height))
        strata.append(stratum)
        sources.append(source)

    predictions["roi_source"] = sources
    predictions["noncancer_stratum"] = strata
    predictions["gate_threshold"] = gate_threshold
    predictions["fallback_width"] = fallback_w
    predictions["fallback_height"] = fallback_h
    for prefix, boxes in [("m1_pred_bbox", m1_boxes),
                          ("roi_base_bbox", base_boxes),
                          ("roi_crop_bbox", crop_boxes)]:
        for index, name in enumerate(["x1", "y1", "x2", "y2"]):
            predictions[f"{prefix}_{name}"] = [box[index] for box in boxes]

    # 校验：所有crop框在[0,1]内且非退化。
    for row in predictions.itertuples(index=False):
        crop = (row.roi_crop_bbox_x1, row.roi_crop_bbox_y1,
                row.roi_crop_bbox_x2, row.roi_crop_bbox_y2)
        if box_area(crop) <= 1e-8:
            raise ValueError(f"裁剪框退化: {row.image_relpath}")
        if not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError(f"裁剪框越界: {row.image_relpath} {crop}")

    manifest_name = f"m4_roi_manifest_seed{args.seed}{suffix}.csv"
    config_name = f"m4_roi_config_seed{args.seed}{suffix}.json"
    predictions.drop(columns=["row_index"]).to_csv(
        output / manifest_name, index=False, encoding="utf-8-sig",
    )
    config = {
        "seed": args.seed,
        "roi_csv_sha256": file_sha256(output / manifest_name),
        "debug": bool(args.debug),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "m1_checkpoint": str(m1_checkpoint.resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "m3c_checkpoint": str(m3c_checkpoint.resolve()),
        "m3c_checkpoint_sha256": file_sha256(m3c_checkpoint),
        "m3c_config_source": str(Path(m3c_checkpoint).parent / "config.json"),
        "gate_threshold": gate_threshold,
        "margin": args.margin,
        "fallback_width": fallback_w,
        "fallback_height": fallback_h,
        "lesion_tercile_bounds": lesion_tercile_bounds,
        "n_train": int(predictions.split.eq("train").sum()),
        "n_val": int(predictions.split.eq("val").sum()),
        "test_evaluated": False,
        "external_evaluated": False,
        "roi_source_counts": predictions.roi_source.value_counts().to_dict(),
    }
    (output / config_name).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"输出: {output / manifest_name}")
    print(f"  ROI来源: {config['roi_source_counts']}")


if __name__ == "__main__":
    main()
