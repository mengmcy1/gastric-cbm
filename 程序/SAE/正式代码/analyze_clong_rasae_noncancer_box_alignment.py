#!/usr/bin/env python3
"""评估医学生非癌框与C-long注意力/RA-SAE Feature响应的空间关系。

输入为CVAT导出的训练集非癌矩形框。脚本只读现有train/val缓存和固定
RA-SAE 100轮权重，不重训、不删除Feature、不读取test/external。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from clong_rasae_core import ArchetypalMatryoshkaSAE
from run_clong_rasae_pilot import read_subset


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
ANNOTATION_ROOT = ROOT / "数据/标注与汇总/训练集非癌病灶标注/训练集非癌病灶标注"
K = 256
GRID = 7
FEATURES = 2560
STRONG_Q99 = 0.5
SOURCE_MAP = {
    "非癌-省人民.zip": "武大省人民",
    "非癌-第一届早癌大赛.zip": "第一届早癌大赛",
    "非癌-第二届早癌大赛.zip": "第二届早癌大赛",
}


def parse_cvat_boxes(root: Path) -> pd.DataFrame:
    """读取CVAT zip中的矩形框，返回逐图标注表。"""
    rows = []
    for archive in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            xml = handle.read("annotations.xml")
        tree = ET.fromstring(xml)
        for image in tree.findall(".//image"):
            boxes = image.findall("box")
            if len(boxes) != 1:
                raise RuntimeError(f"{archive.name}:{image.attrib.get('name')} 不是单框标注")
            box = boxes[0]
            if box.attrib.get("label") != "非癌":
                raise RuntimeError(f"{archive.name}:{image.attrib.get('name')} 标签不是非癌")
            width = float(image.attrib["width"])
            height = float(image.attrib["height"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            rows.append({
                "annotation_source": archive.name,
                "source": SOURCE_MAP.get(archive.name, ""),
                "image_name": image.attrib["name"],
                "annotation_width": width,
                "annotation_height": height,
                "box_x1": xtl,
                "box_y1": ytl,
                "box_x2": xbr,
                "box_y2": ybr,
                "box_x1_norm": xtl / width,
                "box_y1_norm": ytl / height,
                "box_x2_norm": xbr / width,
                "box_y2_norm": ybr / height,
                "box_area_fraction": (xbr - xtl) * (ybr - ytl) / (width * height),
            })
    frame = pd.DataFrame(rows)
    if frame.image_name.duplicated().any():
        raise RuntimeError("非癌框标注存在重复文件名")
    return frame


def load_split_metadata(split: str) -> pd.DataFrame:
    """读取当前缓存metadata并加入source_row和basename。"""
    frame = pd.read_csv(CACHE / f"{split}_metadata.csv")
    frame = frame.reset_index().rename(columns={"index": "source_row"})
    frame["image_name"] = frame.relative_path.astype(str).map(lambda value: Path(value).name)
    return frame


def attach_annotations(metadata: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """将标注按文件名接到当前缓存metadata；只保留非癌图。"""
    eligible = boxes[boxes.source.ne("")].copy()
    noncancer = metadata[metadata.label.eq(0)].copy()
    merged = noncancer.merge(eligible, on=["source", "image_name"], how="inner", validate="one_to_one")
    merged["width_matches"] = merged.width.astype(float).eq(merged.annotation_width.astype(float))
    merged["height_matches"] = merged.height.astype(float).eq(merged.annotation_height.astype(float))
    return merged.sort_values("source_row").reset_index(drop=True)


def cell_overlap_map(box: np.ndarray) -> np.ndarray:
    """返回7x7每个cell被归一化矩形覆盖的比例。"""
    x1, y1, x2, y2 = box
    values = np.zeros((GRID, GRID), dtype=np.float32)
    for y in range(GRID):
        cy1, cy2 = y / GRID, (y + 1) / GRID
        oy = max(0.0, min(y2, cy2) - max(y1, cy1))
        for x in range(GRID):
            cx1, cx2 = x / GRID, (x + 1) / GRID
            ox = max(0.0, min(x2, cx2) - max(x1, cx1))
            values[y, x] = ox * oy * GRID * GRID
    return values.reshape(-1)


def overlap_matrix(frame: pd.DataFrame) -> np.ndarray:
    """逐图生成7x7框覆盖矩阵。"""
    boxes = frame[["box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm"]].to_numpy(float)
    return np.stack([cell_overlap_map(box) for box in boxes]).astype(np.float32)


def load_sae(device: torch.device) -> ArchetypalMatryoshkaSAE:
    """读取固定RA-SAE duration100权重。"""
    checkpoint = torch.load(BASE / "duration100/ra_final.pth", map_location=device, weights_only=False)
    state, config = checkpoint["state_dict"], checkpoint["config"]
    if config["hidden_dim"] != FEATURES or K not in config["k_list"]:
        raise RuntimeError("RA-SAE权重不是预期的2560宽度/K=256")
    model = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], config["hidden_dim"], tuple(config["k_list"]),
        config["delta"], True, config["seed"], config["initialization"],
    ).to(device)
    model.load_state_dict(state)
    model.requires_grad_(False).eval()
    return model


def attention_alignment(split: str, frame: pd.DataFrame, overlap: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """统计冻结C-long attention在非癌框内的质量占比。"""
    attention = np.load(CACHE / f"{split}_attention.npy", mmap_mode="r")
    selected = np.asarray(attention[frame.source_row.to_numpy(int)]).astype(np.float32)
    inside = (selected * overlap).sum(1)
    peak_overlap = overlap[np.arange(len(frame)), selected.argmax(1)]
    rows = frame[["source_row", "patient_id", "image_name", "annotation_source", "box_area_fraction"]].copy()
    rows["attention_box_mass_fraction"] = inside
    rows["attention_peak_box_overlap"] = peak_overlap
    patient = rows.groupby("patient_id", sort=True).agg(
        attention_box_mass_fraction=("attention_box_mass_fraction", "mean"),
        attention_peak_box_overlap=("attention_peak_box_overlap", "mean"),
    )
    summary = {
        "images": int(len(rows)),
        "patients": int(rows.patient_id.nunique()),
        "mean_attention_box_mass_fraction": float(rows.attention_box_mass_fraction.mean()),
        "patient_mean_attention_box_mass_fraction": float(patient.attention_box_mass_fraction.mean()),
        "mean_attention_peak_box_overlap": float(rows.attention_peak_box_overlap.mean()),
        "patient_mean_attention_peak_box_overlap": float(patient.attention_peak_box_overlap.mean()),
    }
    return rows, summary


@torch.no_grad()
def feature_alignment(
    split: str, frame: pd.DataFrame, overlap: np.ndarray, sae: ArchetypalMatryoshkaSAE,
    q99: np.ndarray, device: torch.device, batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按Feature汇总RA-SAE响应在非癌框内的比例。"""
    total = np.zeros(FEATURES, dtype=np.float64)
    inside = np.zeros(FEATURES, dtype=np.float64)
    strong_total = np.zeros(FEATURES, dtype=np.float64)
    strong_inside = np.zeros(FEATURES, dtype=np.float64)
    positive_image = np.zeros(FEATURES, dtype=np.int64)
    strong_image = np.zeros(FEATURES, dtype=np.int64)
    peak_overlap_sum = np.zeros(FEATURES, dtype=np.float64)
    peak_positive_count = np.zeros(FEATURES, dtype=np.int64)
    peak_inside_image = np.zeros(FEATURES, dtype=np.int64)
    active_patients = [set() for _ in range(FEATURES)]
    strong_patients = [set() for _ in range(FEATURES)]
    image_rows = []
    scale = np.asarray(q99, dtype=np.float32).reshape(1, 1, -1)

    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        batch_frame = frame.iloc[start:stop]
        data = read_subset(batch_frame, split, device)
        hidden = sae.encode(data["spatial"], K).cpu().numpy().astype(np.float32)
        current_overlap = overlap[start:stop]
        total += hidden.sum(axis=(0, 1), dtype=np.float64)
        inside += (hidden * current_overlap[:, :, None]).sum(axis=(0, 1), dtype=np.float64)
        strong = hidden >= (STRONG_Q99 * scale)
        strong_total += (hidden * strong).sum(axis=(0, 1), dtype=np.float64)
        strong_inside += (hidden * strong * current_overlap[:, :, None]).sum(axis=(0, 1), dtype=np.float64)

        peaks = hidden.max(axis=1)
        peak_position = hidden.argmax(axis=1)
        image_positive = peaks > 0
        image_strong = peaks >= (STRONG_Q99 * q99.reshape(1, -1))
        positive_image += image_positive.sum(axis=0)
        strong_image += image_strong.sum(axis=0)
        peak_overlap = np.take_along_axis(current_overlap, peak_position, axis=1)
        peak_overlap_sum += (peak_overlap * image_positive).sum(axis=0)
        peak_positive_count += image_positive.sum(axis=0)
        peak_inside_image += ((peak_overlap >= 0.5) & image_positive).sum(axis=0)

        for local, row in enumerate(batch_frame.itertuples(index=False)):
            active_ids = np.flatnonzero(image_positive[local])
            strong_ids = np.flatnonzero(image_strong[local])
            patient = str(row.patient_id)
            for feature_id in active_ids:
                active_patients[int(feature_id)].add(patient)
            for feature_id in strong_ids:
                strong_patients[int(feature_id)].add(patient)
            image_rows.append({
                "split": split,
                "source_row": int(row.source_row),
                "patient_id": patient,
                "image_name": row.image_name,
                "annotation_source": row.annotation_source,
                "box_area_fraction": float(row.box_area_fraction),
                "active_feature_count": int(image_positive[local].sum()),
                "strong_feature_count": int(image_strong[local].sum()),
            })

    feature = pd.DataFrame({
        "feature_id": np.arange(FEATURES, dtype=int),
        f"{split}_box_activation_fraction": np.divide(
            inside, total, out=np.full(FEATURES, np.nan), where=total > 0,
        ),
        f"{split}_strong_box_activation_fraction": np.divide(
            strong_inside, strong_total, out=np.full(FEATURES, np.nan), where=strong_total > 0,
        ),
        f"{split}_positive_image_count": positive_image,
        f"{split}_positive_patient_count": [len(value) for value in active_patients],
        f"{split}_strong_image_count": strong_image,
        f"{split}_strong_patient_count": [len(value) for value in strong_patients],
        f"{split}_peak_box_overlap_mean_active": np.divide(
            peak_overlap_sum, peak_positive_count, out=np.full(FEATURES, np.nan), where=peak_positive_count > 0,
        ),
        f"{split}_peak_inside_image_fraction": np.divide(
            peak_inside_image, peak_positive_count, out=np.full(FEATURES, np.nan), where=peak_positive_count > 0,
        ),
    })
    image = pd.DataFrame(image_rows)
    summary = {
        "images": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "mean_active_features_per_image": float(image.active_feature_count.mean()),
        "mean_strong_features_per_image": float(image.strong_feature_count.mean()),
        "features_with_strong_response": int((strong_image > 0).sum()),
    }
    return feature, image, summary


def write_report(output: Path, summary: dict, feature: pd.DataFrame) -> None:
    """写入简短Markdown说明。"""
    train = summary["splits"]["train"]
    val = summary["splits"]["val"]
    functional = feature.sort_values(
        ["val98_patient_mean_image_abs_delta_margin", "feature_id"],
        ascending=[False, True],
        kind="stable",
    ).head(20)
    high_box = functional.train_box_activation_fraction.ge(0.5).sum()
    low_box = functional.train_box_activation_fraction.lt(0.25).sum()
    text = f"""# 非癌框空间对齐诊断

## 数据

- CVAT非癌矩形框共{summary["annotations"]["total_boxes"]}张图，唯一文件名{summary["annotations"]["unique_images"]}个。
- 当前缓存中匹配train非癌{train["annotated_images"]}张、{train["annotated_patients"]}位患者；匹配val非癌{val["annotated_images"]}张、{val["annotated_patients"]}位患者。
- 外院zip当前没有匹配到train/val缓存，本轮未参与空间诊断。

## C-long注意力

- train非癌框内attention质量均值为{train["attention"]["mean_attention_box_mass_fraction"]:.4f}，患者等权均值为{train["attention"]["patient_mean_attention_box_mass_fraction"]:.4f}。
- val非癌框内attention质量均值为{val["attention"]["mean_attention_box_mass_fraction"]:.4f}，患者等权均值为{val["attention"]["patient_mean_attention_box_mass_fraction"]:.4f}。

## RA-SAE Feature

- 本轮对匹配到的非癌框图投影固定RA-SAE K=256，不重训、不改Feature编号。
- train中出现强响应的Feature为{train["feature"]["features_with_strong_response"]}项；val中为{val["feature"]["features_with_strong_response"]}项。
- val98功能影响Top20 Feature里，train非癌框内响应占比>=0.5的有{int(high_box)}项，<0.25的有{int(low_box)}项。

## 解释边界

- 框是粗矩形，只能作为“是否落在医生标出的非癌病灶区域附近”的空间参照，不是精细分割。
- box内响应不等于医学概念已确认，box外响应也不自动等于伪特征。
- val匹配结果只作描述性诊断；正式筛选仍应优先看train-only统计，再把val作为稳定性补充。
"""
    (output / "结果说明.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    boxes = parse_cvat_boxes(ANNOTATION_ROOT)
    boxes.to_csv(args.output / "noncancer_cvat_boxes.csv", index=False)
    metadata = {split: load_split_metadata(split) for split in ("train", "val")}
    annotated = {split: attach_annotations(metadata[split], boxes) for split in ("train", "val")}
    for split, frame in annotated.items():
        frame.to_csv(args.output / f"{split}_matched_noncancer_boxes.csv", index=False)
        if frame.empty:
            raise RuntimeError(f"{split}没有匹配到非癌框")
        if not frame.width_matches.all() or not frame.height_matches.all():
            raise RuntimeError(f"{split}匹配图像尺寸与CVAT标注尺寸不一致")

    matched_names = set(annotated["train"].image_name) | set(annotated["val"].image_name)
    unmatched = boxes[~boxes.image_name.isin(matched_names)].copy()
    unmatched.to_csv(args.output / "unmatched_annotation_images.csv", index=False)

    overlaps = {split: overlap_matrix(frame) for split, frame in annotated.items()}
    attention_rows, attention_summary = {}, {}
    for split in ("train", "val"):
        attention_rows[split], attention_summary[split] = attention_alignment(split, annotated[split], overlaps[split])
        attention_rows[split].to_csv(args.output / f"{split}_attention_box_alignment.csv", index=False)

    device = torch.device(args.device)
    sae = load_sae(device)
    q99 = np.load(BASE / "decision_alignment/ra_train_q99.npy").astype(np.float32)
    feature_tables, image_tables, feature_summary = {}, {}, {}
    for split in ("train", "val"):
        feature_tables[split], image_tables[split], feature_summary[split] = feature_alignment(
            split, annotated[split], overlaps[split], sae, q99, device, args.batch_size,
        )
        image_tables[split].to_csv(args.output / f"{split}_image_feature_box_summary.csv", index=False)

    feature = feature_tables["train"].merge(feature_tables["val"], on="feature_id", validate="one_to_one")
    usage_path = BASE / "feature_usage_audit_20260915/feature_usage_statistics.csv"
    if usage_path.exists():
        usage = pd.read_csv(usage_path)
        keep = [
            "feature_id", "group", "representative",
            "val98_patient_mean_image_abs_delta_margin",
            "val98_patient_mean_signed_delta_margin",
            "val98_abs_effect_rank_desc",
        ]
        feature = feature.merge(usage[keep], on="feature_id", how="left", validate="one_to_one")
    feature.to_csv(args.output / "feature_box_alignment.csv", index=False)
    feature.sort_values(
        ["val98_patient_mean_image_abs_delta_margin", "feature_id"],
        ascending=[False, True],
        kind="stable",
    ).head(100).to_csv(args.output / "top_functional_feature_box_alignment.csv", index=False)

    summary = {
        "annotations": {
            "total_boxes": int(len(boxes)),
            "unique_images": int(boxes.image_name.nunique()),
            "by_source": boxes.groupby("annotation_source").size().astype(int).to_dict(),
            "unmatched_images": int(len(unmatched)),
        },
        "splits": {},
        "definition": {
            "box_mass_fraction": "sum(response * 7x7_cell_box_overlap_fraction) / sum(response)",
            "strong_response": f"patch activation >= {STRONG_Q99} * train_positive_Q99(feature)",
            "scope": "matched non-cancer images in existing train/val caches",
            "test_read": False,
            "external_read": False,
            "sae_checkpoint": str(BASE / "duration100/ra_final.pth"),
        },
    }
    for split in ("train", "val"):
        summary["splits"][split] = {
            "annotated_images": int(len(annotated[split])),
            "annotated_patients": int(annotated[split].patient_id.nunique()),
            "annotation_sources": annotated[split].groupby("annotation_source").size().astype(int).to_dict(),
            "attention": attention_summary[split],
            "feature": feature_summary[split],
        }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output, summary, feature)


if __name__ == "__main__":
    main()
