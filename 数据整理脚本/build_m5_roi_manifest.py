#!/usr/bin/env python3
"""M5冻结ROI清单构建：冻结M1+M3c-B对train/val确定性推理，两类都用预测框ROI。

按M5预注册（2026-08-12）：
- 癌图与非癌图**全部**使用冻结M1 Top-1预测框作为ROI基准框，不提供真值框；
- M3c-B门控只控制是否向医生展示框，不控制分类分支；gate_pass另存为分层字段；
- 只为train/val生成；internal test/external不读取图像、不生成预测框、不计算q_region；
- M1与M3c-B使用确定性评估变换（不随机增强）；
- roi_label以**扩边后roi_crop_bbox**与医生GT框的病灶覆盖率计算（0/1/2，2=ignore）：
    lesion_coverage = intersection(crop, GT) / area(GT)
    任一GT中心在crop内 且 coverage>=0.50 → 1；与全部GT均无重叠 → 0；其余 → 2；
- 门控阈值与M3c-B checkpoint同源读取；回退框尺寸=train癌图M1命中中位数（只用train）；
- 病灶大小三分位边界只用train癌图bbox_area_fraction冻结。

输出：`m5_roi_manifest_seed{seed}.csv` 与 `config.json`（含sanity验收报告）。
"""

import argparse
import hashlib
import json
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
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M5预测ROI融合_0804/冻结ROI清单"

# roi_label约定（与efficientnet_m5_fusion.py保持一致）：
# 0=ROI不含病灶（非癌恒0；癌图预测框未拍到病灶）；1=ROI明确含病灶；2=ignore（不确定）。
IGNORE = 2
# 冻结：正例需病灶中心在crop内且覆盖至少50%。
POSITIVE_COVERAGE = 0.50
# 与全部GT的覆盖率都小于此值才视为"无重叠"。
ZERO_EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--m1-checkpoint", type=Path, default=None)
    parser.add_argument("--m3c-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="运行expand_and_square边界与roi_label匹配回归自测后退出。")
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
    """扩边margin+正方形化+尽量平移回界，框过大时截断到图像内。

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


def box_intersection_area(a, b):
    """两个归一化框的交叠面积；无重叠时为0。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def gt_center_in_box(gt, crop):
    gcx = (gt[0] + gt[2]) / 2
    gcy = (gt[1] + gt[3]) / 2
    return crop[0] <= gcx <= crop[2] and crop[1] <= gcy <= crop[3]


def lesion_coverage(crop, gt):
    return box_intersection_area(crop, gt) / box_area(gt)


def compute_roi_label(crop, gt_boxes):
    """以扩边后crop与全部有效GT框匹配，返回roi_label（0/1/2，2=ignore）。

    规则（M5预注册冻结）：
    - 任一GT中心在crop内 且 该GT的coverage>=0.50 → 1；
    - 与全部GT均无重叠（coverage<ZERO_EPS）→ 0；
    - 无有效GT框 / 部分覆盖 / 不确定 → 2。
    """
    valid = [box for box in gt_boxes if box_area(box) > 1e-12]
    if not valid:
        return IGNORE
    any_positive = False
    all_zero = True
    for gt in valid:
        cov = lesion_coverage(crop, gt)
        if cov >= POSITIVE_COVERAGE and gt_center_in_box(gt, crop):
            any_positive = True
        if cov >= ZERO_EPS:
            all_zero = False
    if any_positive:
        return 1
    if all_zero:
        return 0
    return IGNORE


def self_test():
    """边界回归：expand_and_square四类边界 + roi_label匹配规则全分支。"""
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

    crop = (0.2, 0.2, 0.8, 0.8)
    cases = [
        # (crop, gt_boxes, 期望roi_label, 说明)
        (crop, [(0.3, 0.3, 0.5, 0.5)], 1, "GT完整在crop内"),
        (crop, [(0.9, 0.9, 1.0, 1.0)], 0, "GT与crop无重叠"),
        (crop, [(0.1, 0.1, 0.25, 0.25)], 2, "部分覆盖且GT中心在crop外"),
        (crop, [(0.7, 0.7, 0.9, 0.9)], 2, "GT中心在内但coverage<0.5"),
        (crop, [], 2, "无有效GT框"),
        ((0.0, 0.0, 1.0, 1.0), [(0.4, 0.4, 0.6, 0.6)], 1, "大ROI完整含小病灶(IoU低也标1)"),
        (crop, [(0.3, 0.3, 0.5, 0.5), (0.9, 0.9, 1.0, 1.0)], 1, "多病灶任一正例→1"),
        (crop, [(0.9, 0.9, 1.0, 1.0), (0.0, 0.0, 0.05, 0.05)], 0, "多病灶全部无重叠→0"),
        (crop, [(0.1, 0.1, 0.25, 0.25), (0.9, 0.9, 1.0, 1.0)], 2, "多病灶混合→ignore"),
    ]
    for crop_box, gt_boxes, expected, note in cases:
        got = compute_roi_label(crop_box, gt_boxes)
        if got != expected:
            raise AssertionError(
                f"roi_label匹配失败[{note}]: 期望{expected} 实际{got}"
            )
    return True


def main():
    """编排M5 ROI清单构建：两类都用预测框，输出roi_label/gate_pass与sanity报告。

    输入: --seed对应的M1 warmup产品、M3c-B门控、冻结M0 manifest；
    输出: m5_roi_manifest_seed{seed}{_debug}.csv 与对应config（含门控阈值、回退框、
    病灶三分位、SHA、sanity验收）；
    防泄漏边界: 只读train/val；不读取internal test/external；M1/M3c-B用确定性评估变换。
    返回: 无；产物写入--output-root，任一同名CSV/config存在时默认快速失败。
    """
    args = parse_args()
    if args.self_test:
        self_test()
        print("M5 ROI构建自测通过: expand_and_square边界 + roi_label匹配全分支")
        return
    if args.seed is None:
        raise ValueError("非自测模式必须提供--seed")
    m1_checkpoint = args.m1_checkpoint or default_m1_checkpoint(args.seed)
    m3c_checkpoint = args.m3c_checkpoint or default_m3c_checkpoint(args.seed)
    for path in (args.manifest, m1_checkpoint, m3c_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    output = args.output_root
    suffix = "_debug" if args.debug else ""
    target_manifest = output / f"m5_roi_manifest_seed{args.seed}{suffix}.csv"
    target_config = output / f"m5_roi_config_seed{args.seed}{suffix}.json"
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

    # 冻结M1 + M3c-B门控；门控阈值与checkpoint同源读取。
    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    m3c_payload = torch.load(m3c_checkpoint, map_location="cpu", weights_only=False)
    m3c_config_path = Path(m3c_checkpoint).parent / "config.json"
    if not m3c_config_path.is_file():
        raise FileNotFoundError(m3c_config_path)
    m3c_run_config = json.loads(m3c_config_path.read_text(encoding="utf-8"))
    m1_sha = file_sha256(m1_checkpoint)
    if int(m3c_run_config.get("seed", -1)) != args.seed:
        raise ValueError("M3c-B config的seed与当前ROI清单seed不一致")
    if m3c_run_config.get("debug", False):
        raise ValueError("禁止使用debug M3c-B产物生成正式ROI清单")
    if m3c_run_config.get("m1_checkpoint_sha256") != m1_sha:
        raise ValueError("M3c-B与当前M1 checkpoint不同源")
    embedded_m3c_config = m3c_payload.get("config", {})
    if int(embedded_m3c_config.get("seed", -1)) != args.seed:
        raise ValueError("M3c-B checkpoint内嵌seed与当前seed不一致")
    if "batch_size" not in m3c_run_config:
        raise ValueError(f"M3c-B配置缺少batch_size，无法校验推理一致性: {m3c_config_path}")
    expected_batch_size = int(m3c_run_config["batch_size"])
    if args.batch_size != expected_batch_size:
        raise ValueError(
            f"M5 ROI推理batch_size={args.batch_size}，"
            f"与M3c-B正式评估batch_size={expected_batch_size}不一致"
        )
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
    repeat_box_max_abs_diff = 0.0
    repeat_q_region_max_abs_diff = 0.0
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
                repeated = model(images)
                repeat_box_max_abs_diff = max(
                    repeat_box_max_abs_diff,
                    float((outputs["boxes"] - repeated["boxes"]).abs().max().item()),
                )
                repeat_q_region_max_abs_diff = max(
                    repeat_q_region_max_abs_diff,
                    float((outputs["gate_logits"] - repeated["gate_logits"]).abs().max().item()),
                )
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
                        "gate_pass": int(q_region[local] >= gate_threshold),
                    }
                    for column in ["source", "center", "size_group", "bbox_area_fraction"]:
                        if column in split_frame.columns:
                            row_dict[column] = meta[column]
                    rows.append(row_dict)

    repeat_tolerance = 1e-6
    if (repeat_box_max_abs_diff > repeat_tolerance
            or repeat_q_region_max_abs_diff > repeat_tolerance):
        raise RuntimeError(
            "确定性重复推理校验失败: "
            f"box diff={repeat_box_max_abs_diff:.3e}, "
            f"q_region diff={repeat_q_region_max_abs_diff:.3e}"
        )

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

    # 病灶大小三分位边界：只用train癌图bbox_area_fraction，冻结进config供M5分层。
    train_cancer_area = predictions.loc[
        predictions.split.eq("train") & predictions.label.eq(1),
        "bbox_area_fraction",
    ]
    lesion_tercile_bounds = (
        train_cancer_area.quantile([0.33, 0.66]).tolist()
        if len(train_cancer_area) else None
    )
    print(f"病灶大小三分位边界(train癌图): {lesion_tercile_bounds}")

    # 两类全部用预测框ROI；roi_label按扩边后crop与GT匹配；gate_pass已存。
    m1_boxes = []
    base_boxes = []
    crop_boxes = []
    roi_labels = []
    max_coverage = []
    sources = []
    strata = []
    for row in predictions.itertuples(index=False):
        m1_box = (row.m1_x1, row.m1_y1, row.m1_x2, row.m1_y2)
        m1_boxes.append(m1_box)
        base = m1_box
        source = "predicted"
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
        crop = (crop_px[0] / row.width, crop_px[1] / row.height,
                crop_px[2] / row.width, crop_px[3] / row.height)
        crop_boxes.append(crop)
        if int(row.label) == 1:
            gt_boxes = [(row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2)]
            roi_labels.append(compute_roi_label(crop, gt_boxes))
            max_cov = max(
                (lesion_coverage(crop, gt) for gt in gt_boxes
                 if box_area(gt) > 1e-12),
                default=0.0,
            )
            max_coverage.append(float(max_cov))
            stratum = ""
        else:
            roi_labels.append(0)
            max_coverage.append(float("nan"))
            stratum = "hard" if row.q_region >= gate_threshold else "easy"
        sources.append(source)
        strata.append(stratum)

    predictions["roi_source"] = sources
    predictions["noncancer_stratum"] = strata
    predictions["roi_label"] = roi_labels
    predictions["max_lesion_coverage"] = max_coverage
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

    # sanity验收报告（只检查实现是否合理，不用于调阈值）。
    sanity = build_sanity_report(predictions, gate_threshold)

    manifest_name = f"m5_roi_manifest_seed{args.seed}{suffix}.csv"
    config_name = f"m5_roi_config_seed{args.seed}{suffix}.json"
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
        "m1_checkpoint_sha256": m1_sha,
        "m3c_checkpoint": str(m3c_checkpoint.resolve()),
        "m3c_checkpoint_sha256": file_sha256(m3c_checkpoint),
        "m3c_config_source": str(Path(m3c_checkpoint).parent / "config.json"),
        "gate_threshold": gate_threshold,
        "inference_batch_size": args.batch_size,
        "reference_m3c_batch_size": expected_batch_size,
        "margin": args.margin,
        "roi_positive_coverage": POSITIVE_COVERAGE,
        "roi_ignore_label": IGNORE,
        "roi_zero_eps": ZERO_EPS,
        "repeat_inference_check": {
            "tolerance": repeat_tolerance,
            "box_max_abs_diff": repeat_box_max_abs_diff,
            "q_region_max_abs_diff": repeat_q_region_max_abs_diff,
            "passed": True,
        },
        "fallback_width": fallback_w,
        "fallback_height": fallback_h,
        "lesion_tercile_bounds": lesion_tercile_bounds,
        "n_train": int(predictions.split.eq("train").sum()),
        "n_val": int(predictions.split.eq("val").sum()),
        "test_evaluated": False,
        "external_evaluated": False,
        "roi_source_counts": predictions.roi_source.value_counts().to_dict(),
        "sanity": sanity,
    }
    (output / config_name).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"输出: {output / manifest_name}")
    print(f"  ROI来源: {config['roi_source_counts']}")
    print(f"  sanity: roi_label计数(val)={config['sanity']['roi_label_counts_val']}")


def build_sanity_report(predictions, gate_threshold):
    """预注册要求的sanity验收：roi_label计数、coverage分布、GT中心命中、gate分层。"""
    report = {}
    for split in ("train", "val"):
        sub = predictions.loc[predictions.split.eq(split)]
        report[f"roi_label_counts_{split}"] = {
            str(k): int((sub.roi_label == k).sum())
            for k in (0, 1, 2)
        }
        report[f"roi_label_patients_{split}"] = {
            str(k): int(sub.loc[sub.roi_label.eq(k), "patient_id"].nunique())
            for k in (0, 1, 2)
        }
    cancer = predictions.loc[predictions.label.eq(1)]
    report["cancer_coverage_percentiles"] = [
        float(v) for v in np.nanpercentile(
            cancer.max_lesion_coverage.to_numpy(), [0, 25, 50, 75, 100]
        )
    ]
    # 癌图GT中心命中crop比例：只在癌图内统计（GT框在癌图才有效）。
    cancer_center_hit = cancer.apply(
        lambda row: box_center_inside(
            (row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2),
            (row.roi_crop_bbox_x1, row.roi_crop_bbox_y1,
             row.roi_crop_bbox_x2, row.roi_crop_bbox_y2),
        ), axis=1,
    )
    report["cancer_gt_center_hit_rate"] = float(
        cancer_center_hit.mean()
    ) if len(cancer_center_hit) else 0.0
    report["cancer_gate_pass"] = {
        "pass": int((cancer.gate_pass == 1).sum()),
        "fail": int((cancer.gate_pass == 0).sum()),
    }
    report["cancer_roi_label_x_gate_cross"] = {}
    for label in (0, 1, 2):
        sub = cancer.loc[cancer.roi_label.eq(label)]
        report["cancer_roi_label_x_gate_cross"][str(label)] = {
            "gate_pass": int((sub.gate_pass == 1).sum()),
            "gate_fail": int((sub.gate_pass == 0).sum()),
        }
    report["fallback_count"] = int((predictions.roi_source == "fallback").sum())
    return report


if __name__ == "__main__":
    main()
