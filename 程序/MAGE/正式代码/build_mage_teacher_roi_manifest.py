#!/usr/bin/env python3
"""Build and audit the locked MG0b teacher-ROI manifest.

Train cancer images use GT boxes; train non-cancer images use patient-level
OOF YOLO Top-1 boxes. Validation cancer images use GT boxes and validation
non-cancer images use the frozen Y3-F seed-42 Top-1 predictions. Missing
candidates receive a deterministic source/size-matched fallback centered on
valid image content. Internal test and external data are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
G0_MANIFEST = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0a_患者级OOF分折_20260817/mage_g0_manifest.csv"
)
OOF_ROOT = PROJECT_ROOT / "结果/MAGE/MG0b_OOF_YOLO_20260817"
Y3F_RUN = PROJECT_ROOT / (
    "结果/YOLO26定位_0804/Y3固定640三种子/y3_full_yolo26s_640_seed42"
)
Y3F_PREDICTIONS = Y3F_RUN / "y3_val_top1_predictions.csv"
Y3F_CONFIG = Y3F_RUN / "y3_geometry_config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_教师ROI清单与审计_20260817"
)

N_FOLDS = 5
CANDIDATE_CONFIDENCE = 0.001
ROI_MARGIN = 0.20
QC_SEED = 42
QC_PER_GROUP = 20
QC_TILE = 320
GEOMETRY_FEATURES = [
    "base_center_x", "base_center_y", "base_width", "base_height",
    "base_area", "base_aspect_ratio",
]


def parse_args() -> argparse.Namespace:
    """Parse frozen inputs, output location and self-test mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-manifest", type=Path, default=G0_MANIFEST)
    parser.add_argument("--oof-root", type=Path, default=OOF_ROOT)
    parser.add_argument("--y3f-predictions", type=Path, default=Y3F_PREDICTIONS)
    parser.add_argument("--y3f-config", type=Path, default=Y3F_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_square_box(box: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    """Expand a normalized box, make it square in pixels and clamp it in-frame."""
    scale = np.array([width, height, width, height], dtype=float)
    x1, y1, x2, y2 = np.asarray(box, dtype=float) * scale
    if not np.all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"无效ROI源框: {box}")
    side = min(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin), width, height)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx = min(max(cx, side / 2.0), width - side / 2.0)
    cy = min(max(cy, side / 2.0), height - side / 2.0)
    pixels = np.array([cx - side / 2.0, cy - side / 2.0,
                       cx + side / 2.0, cy + side / 2.0])
    return np.clip(pixels / scale, 0.0, 1.0)


def content_center(image_path: Path) -> tuple[float, float]:
    """Return the center of the non-black content bounds in normalized coordinates."""
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"))
    ys, xs = np.where(gray > 8)
    if not len(xs):
        return 0.5, 0.5
    return float((xs.min() + xs.max() + 1) / (2 * gray.shape[1])), float(
        (ys.min() + ys.max() + 1) / (2 * gray.shape[0])
    )


def centered_box(cx: float, cy: float, width: float, height: float) -> np.ndarray:
    """Place one normalized box at a center while keeping it inside the image."""
    width, height = min(max(width, 1e-4), 1.0), min(max(height, 1e-4), 1.0)
    cx = min(max(cx, width / 2.0), 1.0 - width / 2.0)
    cy = min(max(cy, height / 2.0), 1.0 - height / 2.0)
    return np.array([cx - width / 2.0, cy - height / 2.0,
                     cx + width / 2.0, cy + height / 2.0])


def validate_g0(frame: pd.DataFrame) -> None:
    """Validate the immutable train/val queue and reject locked-data contamination."""
    expected = {("train", 0): 1169, ("train", 1): 1181,
                ("val", 0): 257, ("val", 1): 240}
    if len(frame) != 2847 or frame.groupby(["split", "label"]).size().to_dict() != expected:
        raise ValueError("MG0a清单规模与冻结记录不一致")
    if frame.relative_path.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError("MG0a清单存在重复图像")
    if not set(frame.split).issubset({"train", "val"}):
        raise ValueError("MG0a清单混入test数据")
    if frame[["internal_test_read", "external_read"]].astype(bool).any().any():
        raise ValueError("MG0a清单的数据边界标记异常")
    missing_images = [rel for rel in frame.image_relpath if not (PROJECT_ROOT / rel).is_file()]
    if missing_images:
        raise FileNotFoundError(f"MG0a原图缺失，首例: {missing_images[0]}")


def load_oof_predictions(oof_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Merge five formal OOF holdout predictions and verify unique coverage."""
    frames, hashes = [], {}
    for fold in range(N_FOLDS):
        run = oof_root / f"mg0b_oof_yolo26s_fold{fold}"
        prediction_path = run / "holdout_top1_predictions.csv"
        config_path = run / "mg0b_config.json"
        if not prediction_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"MG0b fold{fold}正式产物不完整")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("debug") or config.get("stage") != "MG0b_oof_yolo":
            raise ValueError(f"MG0b fold{fold}不是正式产物")
        frame = pd.read_csv(
            prediction_path, encoding="utf-8-sig", dtype={"patient_id": str},
            float_precision="round_trip",
        )
        if not frame.prediction_provenance.eq(f"oof_fold_{fold}").all():
            raise ValueError(f"MG0b fold{fold}预测来源异常")
        frames.append(frame)
        hashes[f"fold_{fold}_predictions"] = file_sha256(prediction_path)
        hashes[f"fold_{fold}_checkpoint"] = config["products"]["best_checkpoint_sha256"]
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != 2350 or merged.relative_path.nunique() != 2350:
        raise ValueError("五折OOF预测未唯一覆盖2350张train图")
    if merged.patient_id.nunique() != 1212 or not merged.split.eq("train").all():
        raise ValueError("五折OOF患者或split边界异常")
    return merged, hashes


def merge_predictions(g0: pd.DataFrame, oof: pd.DataFrame, y3f_path: Path) -> pd.DataFrame:
    """Attach role-correct Top-1 predictions to the frozen train/val manifest."""
    prediction_columns = [
        "relative_path", "top1_confidence", "candidate_count_at_0p001",
        "top1_x1", "top1_y1", "top1_x2", "top1_y2", "prediction_provenance",
    ]
    train = g0.loc[g0.split.eq("train")].merge(
        oof[prediction_columns], on="relative_path", how="left", validate="one_to_one"
    )
    if train.top1_confidence.isna().any():
        raise ValueError("train存在未绑定OOF预测的图像")

    val_predictions = pd.read_csv(
        y3f_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    if len(val_predictions) != 497 or val_predictions.relative_path.nunique() != 497:
        raise ValueError("冻结Y3-F val预测规模异常")
    val_predictions = val_predictions[prediction_columns[:-1]].copy()
    val_predictions["prediction_provenance"] = "frozen_y3f_full_seed42"
    val = g0.loc[g0.split.eq("val")].merge(
        val_predictions, on="relative_path", how="left", validate="one_to_one"
    )
    if val.top1_confidence.isna().any():
        raise ValueError("val存在未绑定冻结Y3-F预测的图像")
    return pd.concat([train, val], ignore_index=True)


def fallback_size_pool(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """Build deterministic raw lesion-size medians from train cancer GT boxes."""
    cancer = frame.loc[frame.split.eq("train") & frame.label.eq(1)].copy()
    cancer["raw_width"] = cancer.bbox_x2_norm - cancer.bbox_x1_norm
    cancer["raw_height"] = cancer.bbox_y2_norm - cancer.bbox_y1_norm
    pools: dict[tuple[str, str], tuple[float, float]] = {}
    for keys, subset in cancer.groupby(["source", "size_group"], dropna=False):
        pools[(str(keys[0]), str(keys[1]))] = (
            float(subset.raw_width.median()), float(subset.raw_height.median())
        )
    for source, subset in cancer.groupby("source", dropna=False):
        pools[(str(source), "*")] = (
            float(subset.raw_width.median()), float(subset.raw_height.median())
        )
    pools[("*", "*")] = (float(cancer.raw_width.median()), float(cancer.raw_height.median()))
    return pools


def choose_fallback_size(
    pools: dict[tuple[str, str], tuple[float, float]], source: str, size_group: str
) -> tuple[float, float, str]:
    """Resolve source+size, source-only, then global fallback dimensions."""
    for key, level in (((source, size_group), "source_size"),
                       ((source, "*"), "source"), (("*", "*"), "global")):
        if key in pools:
            return *pools[key], level
    raise RuntimeError("无法解析fallback尺寸")


def assign_teacher_boxes(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign GT, predicted or deterministic fallback source and base crop boxes."""
    pools = fallback_size_pool(frame)
    records = []
    for row in frame.itertuples(index=False):
        cancer = int(row.label) == 1
        has_candidate = float(row.top1_confidence) >= CANDIDATE_CONFIDENCE
        if cancer:
            source_box = np.array([
                row.bbox_x1_norm, row.bbox_y1_norm, row.bbox_x2_norm, row.bbox_y2_norm
            ], dtype=float)
            roi_source, fallback_level = "gt_bbox", "not_applicable"
        elif has_candidate:
            source_box = np.array([
                row.top1_x1, row.top1_y1, row.top1_x2, row.top1_y2
            ], dtype=float)
            roi_source = "oof_yolo_top1" if row.split == "train" else "frozen_y3f_seed42_top1"
            fallback_level = "not_applicable"
        else:
            fallback_width, fallback_height, fallback_level = choose_fallback_size(
                pools, str(row.source), str(row.size_group)
            )
            cx, cy = content_center(PROJECT_ROOT / row.image_relpath)
            source_box = centered_box(cx, cy, fallback_width, fallback_height)
            roi_source = "deterministic_fallback"
        base_box = normalized_square_box(source_box, int(row.width), int(row.height), ROI_MARGIN)
        record = row._asdict()
        record.update({
            "roi_source": roi_source,
            "fallback_level": fallback_level,
            "source_box_x1": float(source_box[0]), "source_box_y1": float(source_box[1]),
            "source_box_x2": float(source_box[2]), "source_box_y2": float(source_box[3]),
            "base_crop_x1": float(base_box[0]), "base_crop_y1": float(base_box[1]),
            "base_crop_x2": float(base_box[2]), "base_crop_y2": float(base_box[3]),
            "base_center_x": float((base_box[0] + base_box[2]) / 2),
            "base_center_y": float((base_box[1] + base_box[3]) / 2),
            "base_width": float(base_box[2] - base_box[0]),
            "base_height": float(base_box[3] - base_box[1]),
            "base_area": float((base_box[2] - base_box[0]) * (base_box[3] - base_box[1])),
            "base_aspect_ratio": float((base_box[2] - base_box[0]) / (base_box[3] - base_box[1])),
            "candidate_available_at_0p001": bool(has_candidate),
            "lesion_bbox_available": bool(cancer),
            "internal_test_read": False,
            "external_read": False,
        })
        records.append(record)
    output = pd.DataFrame(records)
    boxes = output[["base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"]].to_numpy()
    if not np.all(np.isfinite(boxes)) or not np.all((boxes >= 0) & (boxes <= 1)):
        raise ValueError("教师base_crop_box存在非有限值或越界")
    if not np.all((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])):
        raise ValueError("教师base_crop_box宽高非正")
    return output


def patient_mean_auc(frame: pd.DataFrame, probability_column: str) -> float:
    """Return patient-level AUC after mean aggregation of image probabilities."""
    patient = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), probability=(probability_column, "mean")
    )
    return float(roc_auc_score(patient.label, patient.probability))


def geometry_only_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit geometry-only LR on train and evaluate image/patient AUC on val."""
    train = frame.loc[frame.split.eq("train")].copy()
    val = frame.loc[frame.split.eq("val")].copy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(train[GEOMETRY_FEATURES], train.label)
    val["geometry_probability"] = model.predict_proba(val[GEOMETRY_FEATURES])[:, 1]
    metrics = {
        "features": GEOMETRY_FEATURES,
        "fit_split": "train",
        "evaluation_split": "val",
        "patient_aggregation": "mean image probability",
        "val_image_auc": float(roc_auc_score(val.label, val.geometry_probability)),
        "val_patient_auc": patient_mean_auc(val, "geometry_probability"),
    }
    return metrics, val[[
        "relative_path", "patient_id", "label", "roi_source", "geometry_probability"
    ]]


def describe_group(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return grouped image/patient counts and ROI geometry summaries."""
    return (
        frame.groupby(columns, dropna=False)
        .agg(
            images=("relative_path", "size"), patients=("patient_id", "nunique"),
            confidence_median=("top1_confidence", "median"),
            area_median=("base_area", "median"),
            center_x_median=("base_center_x", "median"),
            center_y_median=("base_center_y", "median"),
        )
        .reset_index()
    )


def crop_image(image: Image.Image, box: np.ndarray) -> Image.Image:
    """Crop one normalized box and return a 224-square luma RGB teacher view."""
    width, height = image.size
    pixels = (box * np.array([width, height, width, height])).round().astype(int)
    pixels[[0, 2]] = np.clip(pixels[[0, 2]], 0, width)
    pixels[[1, 3]] = np.clip(pixels[[1, 3]], 0, height)
    return image.crop(tuple(pixels)).resize((224, 224), Image.Resampling.BILINEAR).convert("L").convert("RGB")


def fit_panel(image: Image.Image, size: int) -> Image.Image:
    """Letterbox an image into a square white QC panel without distortion."""
    panel = Image.new("RGB", (size, size), "white")
    fitted = image.copy()
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return panel


def qc_triptych(row: pd.Series) -> Image.Image:
    """Render original, annotated source/base boxes and clean grayscale crop."""
    with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
        original = handle.convert("RGB")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    source = np.array([row.source_box_x1, row.source_box_y1, row.source_box_x2, row.source_box_y2])
    base = np.array([row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2])
    source_pixels = source * np.array([w, h, w, h])
    base_pixels = base * np.array([w, h, w, h])
    draw.rectangle(tuple(source_pixels), outline=(0, 255, 0), width=max(2, w // 300))
    draw.rectangle(tuple(base_pixels), outline=(255, 215, 0), width=max(2, w // 300))
    crop = crop_image(original, base)
    panels = [fit_panel(original, QC_TILE), fit_panel(annotated, QC_TILE), fit_panel(crop, QC_TILE)]
    caption_h = 52
    output = Image.new("RGB", (QC_TILE * 3, QC_TILE + caption_h), "white")
    for index, panel in enumerate(panels):
        output.paste(panel, (index * QC_TILE, 0))
    caption = (
        f"{row.split} label={int(row.label)} source={row.roi_source} "
        f"conf={float(row.top1_confidence):.4f} file={Path(row.relative_path).name[-72:]}"
    )
    ImageDraw.Draw(output).text((6, QC_TILE + 5), caption, fill="black", font=ImageFont.load_default())
    return output


def save_qc_sheet(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Save vertically stacked triptychs for one deterministic QC stratum."""
    if frame.empty:
        return
    triptychs = [qc_triptych(row) for _, row in frame.iterrows()]
    header_h = 30
    canvas = Image.new("RGB", (QC_TILE * 3, header_h + sum(x.height for x in triptychs)), "white")
    ImageDraw.Draw(canvas).text((6, 8), title, fill="black", font=ImageFont.load_default())
    y = header_h
    for triptych in triptychs:
        canvas.paste(triptych, (0, y))
        y += triptych.height
    canvas.save(path, quality=92)


def stratified_qc(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    """Create deterministic QC sheets for GT, confidence bands and fallback ROIs."""
    rng = np.random.default_rng(QC_SEED)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir()
    noncancer_train = frame.loc[frame.split.eq("train") & frame.label.eq(0)].copy()
    candidates = noncancer_train.loc[noncancer_train.candidate_available_at_0p001].copy()
    candidates["confidence_band"] = pd.qcut(
        candidates.top1_confidence.rank(method="first"), 3,
        labels=["low", "mid", "high"],
    )
    groups: list[tuple[str, pd.DataFrame]] = [
        ("train_cancer_gt", frame.loc[frame.split.eq("train") & frame.label.eq(1)]),
        ("train_noncancer_low_conf", candidates.loc[candidates.confidence_band.eq("low")]),
        ("train_noncancer_mid_conf", candidates.loc[candidates.confidence_band.eq("mid")]),
        ("train_noncancer_high_conf", candidates.loc[candidates.confidence_band.eq("high")]),
        ("train_noncancer_fallback", noncancer_train.loc[noncancer_train.roi_source.eq("deterministic_fallback")]),
        ("val_cancer_gt", frame.loc[frame.split.eq("val") & frame.label.eq(1)]),
        ("val_noncancer_predicted", frame.loc[frame.split.eq("val") & frame.label.eq(0) & ~frame.roi_source.eq("deterministic_fallback")]),
        ("val_noncancer_fallback", frame.loc[frame.split.eq("val") & frame.label.eq(0) & frame.roi_source.eq("deterministic_fallback")]),
    ]
    outputs = []
    for name, subset in groups:
        if len(subset) > QC_PER_GROUP:
            indices = rng.choice(subset.index.to_numpy(), size=QC_PER_GROUP, replace=False)
            subset = subset.loc[np.sort(indices)]
        path = qc_dir / f"{name}.jpg"
        save_qc_sheet(subset.reset_index(drop=True), path, f"{name}: original | source/base boxes | luma crop")
        if path.is_file():
            outputs.append(str(path))
    return outputs


def run_self_test() -> None:
    """Exercise square clamping, deterministic placement and geometry audit."""
    square = normalized_square_box(np.array([0.1, 0.2, 0.3, 0.4]), 800, 600, 0.2)
    assert np.all((square >= 0) & (square <= 1))
    pixels = square * np.array([800, 600, 800, 600])
    assert abs((pixels[2] - pixels[0]) - (pixels[3] - pixels[1])) < 1e-6
    placed = centered_box(0.0, 1.0, 0.4, 0.2)
    assert np.allclose(placed, [0.0, 0.8, 0.4, 1.0])
    rows = []
    for split, count in (("train", 20), ("val", 10)):
        for label in (0, 1):
            for index in range(count):
                value = 0.2 + 0.5 * label + index / 1000
                rows.append({
                    "split": split, "label": label, "patient_id": f"{split}-{label}-{index}",
                    "relative_path": f"{split}-{label}-{index}.jpg", "roi_source": "synthetic",
                    **{feature: value for feature in GEOMETRY_FEATURES},
                })
    metrics, predictions = geometry_only_audit(pd.DataFrame(rows))
    assert metrics["val_image_auc"] > 0.99 and len(predictions) == 20
    print("MG0b教师ROI self-test通过: 方框、回退放置与geometry-only审计有效")


def main() -> None:
    """Build the formal teacher manifest, audits, geometry baseline and QC sheets."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    g0_path = args.g0_manifest.resolve()
    oof_root = args.oof_root.resolve()
    y3f_predictions = args.y3f_predictions.resolve()
    y3f_config_path = args.y3f_config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"MG0b教师ROI输出已存在: {output_dir}")

    g0 = pd.read_csv(g0_path, encoding="utf-8-sig", dtype={"patient_id": str})
    validate_g0(g0)
    oof, lineage_hashes = load_oof_predictions(oof_root)
    y3f_config = json.loads(y3f_config_path.read_text(encoding="utf-8"))
    if y3f_config.get("stage") != "Y3-F" or y3f_config.get("role") != "full" or y3f_config.get("seed") != 42:
        raise ValueError("val非癌预测不是冻结Y3-F full seed42产物")
    frame = assign_teacher_boxes(merge_predictions(g0, oof, y3f_predictions))

    if len(frame) != 2847 or frame.relative_path.nunique() != 2847:
        raise ValueError("最终教师ROI清单未唯一覆盖train/val")
    expected_sources = {
        ("train", 0): {"oof_yolo_top1", "deterministic_fallback"},
        ("train", 1): {"gt_bbox"},
        ("val", 0): {"frozen_y3f_seed42_top1", "deterministic_fallback"},
        ("val", 1): {"gt_bbox"},
    }
    for key, expected in expected_sources.items():
        actual = set(frame.loc[frame.split.eq(key[0]) & frame.label.eq(key[1]), "roi_source"])
        if not actual.issubset(expected) or not actual:
            raise ValueError(f"{key} ROI来源异常: {actual}")

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "mage_teacher_roi_manifest.csv"
    frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    geometry_metrics, geometry_predictions = geometry_only_audit(frame)
    geometry_predictions.to_csv(
        output_dir / "geometry_only_val_predictions.csv", index=False, encoding="utf-8-sig"
    )
    split_source = describe_group(frame, ["split", "label", "roi_source"])
    split_source.to_csv(output_dir / "split_label_roi_source_summary.csv", index=False, encoding="utf-8-sig")
    source_size = describe_group(frame, ["split", "label", "source", "size_group", "roi_source"])
    source_size.to_csv(output_dir / "source_size_roi_audit.csv", index=False, encoding="utf-8-sig")

    train_noncancer = frame.loc[frame.split.eq("train") & frame.label.eq(0)]
    candidate_confidence = train_noncancer.loc[
        train_noncancer.candidate_available_at_0p001, "top1_confidence"
    ]
    quantiles = {
        f"p{int(q * 100)}": float(candidate_confidence.quantile(q))
        for q in (0.25, 0.50, 0.75, 0.90)
    }
    qc_paths = stratified_qc(frame, output_dir)
    config = {
        "stage": "MG0b_teacher_roi_manifest_and_audit",
        "created_on": "2026-08-17",
        "candidate_confidence": CANDIDATE_CONFIDENCE,
        "roi_margin": ROI_MARGIN,
        "g0_manifest": str(g0_path),
        "g0_manifest_sha256": file_sha256(g0_path),
        "oof_root": str(oof_root),
        "oof_lineage": lineage_hashes,
        "y3f_predictions": str(y3f_predictions),
        "y3f_predictions_sha256": file_sha256(y3f_predictions),
        "y3f_config": str(y3f_config_path),
        "y3f_config_sha256": file_sha256(y3f_config_path),
        "y3f_checkpoint_sha256": y3f_config["checkpoint_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "counts": {
            "images": int(len(frame)),
            "patients": int(frame.patient_id.nunique()),
            "train_images": int(frame.split.eq("train").sum()),
            "val_images": int(frame.split.eq("val").sum()),
            "train_noncancer_candidates": int(train_noncancer.candidate_available_at_0p001.sum()),
            "train_noncancer_fallback": int(train_noncancer.roi_source.eq("deterministic_fallback").sum()),
            "val_noncancer_fallback": int(
                (frame.split.eq("val") & frame.label.eq(0) & frame.roi_source.eq("deterministic_fallback")).sum()
            ),
        },
        "train_noncancer_candidate_confidence_quantiles": quantiles,
        "geometry_only": geometry_metrics,
        "qc_sheets": qc_paths,
        "teacher_static_coordinates": {
            "lesion_bbox": "original GT bbox; cancer only; evaluation target",
            "source_box": "GT, Top-1 prediction or deterministic fallback before margin",
            "base_crop_box": "source box after margin=0.20, pixel-square conversion and clamping",
            "runtime_crop_box": "base crop after train-time jitter; Dataset must return it",
        },
        "patient_overlap_across_split": bool(frame.groupby("patient_id").split.nunique().gt(1).any()),
        "test_rows_read": 0,
        "internal_test_read": False,
        "external_read": False,
    }
    config_path = output_dir / "mg0b_teacher_roi_audit.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"教师ROI清单: {manifest_path}")
    print(split_source.to_string(index=False))
    print(f"train非癌候选置信度分位数: {quantiles}")
    print(f"Geometry-only val image AUC={geometry_metrics['val_image_auc']:.4f}, "
          f"patient AUC={geometry_metrics['val_patient_auc']:.4f}")
    print(f"质控图: {len(qc_paths)}份; 输出目录: {output_dir}")
    print("未读取Y0-F test、internal test或external。")


if __name__ == "__main__":
    main()
