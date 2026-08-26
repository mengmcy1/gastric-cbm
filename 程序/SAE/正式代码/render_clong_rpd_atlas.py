#!/usr/bin/env python3
"""按冻结RP-D病例清单渲染技术Atlas、盲审包和资产审计。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from clong_rpd_core import blind_hash
from clong_rpd_render_core import placeholder, positive_q99, save_case_assets, thumbnail


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
SELECTION_ROOT = (
    PROJECT_ROOT / "结果/SAE/RP_D_Technical_Atlas_20260825/manifest_dry_run_v1_retry1"
)
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_D_Technical_Atlas_20260825/render_v1"
RENDER_PROTOCOL = CODE_ROOT / "rpd_render_protocol_v1.json"
SELECTION_PROTOCOL = CODE_ROOT / "rpd_atlas_protocol_v1.json"
RPA_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824/analysis_cache"
RPC2_EVIDENCE = (
    PROJECT_ROOT
    / "结果/SAE/RP_C2_Intervention_20260825/formal_retry1/rpc2_target_seed_evidence.csv"
)
SEEDS = (42, 43, 44)
STANDARD_ROLES = ("high", "mid", "low", "zero")
TECHNICAL_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TECHNICAL_FONT_SIZE = 12


def file_sha256(path: Path) -> str:
    """计算单文件SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render frozen RP-D v1 Atlas")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--debug-anchor-limit", type=int, default=0)
    return parser.parse_args()


def verify_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """核验选择协议、冻结记录和四张manifest SHA。"""
    protocol = json.loads(RENDER_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_image_rendering_2026-08-25":
        raise RuntimeError("RP-D render协议未冻结")
    if not TECHNICAL_FONT_PATH.is_file():
        raise FileNotFoundError(f"RP-D技术面板字体不存在: {TECHNICAL_FONT_PATH}")
    if file_sha256(SELECTION_PROTOCOL) != protocol["selection_protocol_sha256"]:
        raise RuntimeError("RP-D selection protocol SHA不一致")
    freeze_path = SELECTION_ROOT / "selection_freeze_v1.json"
    if file_sha256(freeze_path) != protocol["selection_freeze_sha256"]:
        raise RuntimeError("RP-D selection freeze SHA不一致")
    for name, expected in protocol["selection_manifest_sha256"].items():
        if file_sha256(SELECTION_ROOT / name) != expected:
            raise RuntimeError(f"RP-D冻结manifest SHA不一致: {name}")
    anchors = pd.read_csv(SELECTION_ROOT / "rpd_anchor_manifest.csv", dtype={"anchor_id": str})
    cases = pd.read_csv(
        SELECTION_ROOT / "rpd_case_manifest.csv",
        dtype={"anchor_id": str, "patient_id": str, "case_id": str},
        low_memory=False,
    )
    if len(anchors) != 149 or anchors.anchor_id.nunique() != 149 or cases.case_id.duplicated().any():
        raise RuntimeError("RP-D冻结Anchor或case主键异常")
    return protocol, anchors, cases


def select_debug_anchors(anchors: pd.DataFrame, limit: int) -> pd.DataFrame:
    """debug固定优先覆盖Heavy，再补非Heavy。"""
    if limit <= 0:
        return anchors
    if limit >= 2:
        first = pd.concat([
            anchors[anchors.heavy_atlas].sort_values("anchor_id").head(1),
            anchors[~anchors.heavy_atlas].sort_values("anchor_id").head(1),
        ])
    else:
        first = anchors[anchors.heavy_atlas].sort_values("anchor_id").head(1)
    ordered = pd.concat([
        first,
        anchors[anchors.heavy_atlas].sort_values("anchor_id"),
        anchors[~anchors.heavy_atlas].sort_values("anchor_id"),
    ]).drop_duplicates("anchor_id")
    return ordered.head(limit).reset_index(drop=True)


def add_asset_record(
    records: list[dict], output_root: Path, anchor_id: str, case_id: str,
    seed: int | None, feature_id: int | None, asset_type: str, path: Path,
    q99: float | None = None, status: str = "ok", reason: str = "",
) -> None:
    """向资产审计表追加一条记录。"""
    records.append({
        "anchor_id": anchor_id,
        "case_id": case_id,
        "seed": seed,
        "feature_id": feature_id,
        "asset_type": asset_type,
        "relative_output_path": str(path.relative_to(output_root)) if path else "",
        "sha256": file_sha256(path) if path and path.is_file() else "",
        "q99_scale": q99,
        "asset_status": status,
        "failure_reason": reason,
    })


def render_seed_assets(
    seed: int,
    anchors: pd.DataFrame,
    cases: pd.DataFrame,
    protocol: dict,
    output_root: Path,
    asset_records: list[dict],
) -> pd.DataFrame:
    """一次载入一个seed的冻结激活，渲染该seed全部所需病例。"""
    seed_cases = cases if seed == 42 else cases[cases.in_heavy_atlas.astype(bool)]
    feature_column = f"feature_{seed}"
    feature_ids = anchors[feature_column].to_numpy(dtype=np.int64)
    activation_path = RPA_ROOT / f"seed{seed}/train_image_activations.npy"
    activation = np.load(activation_path, mmap_mode="r")
    selected = np.take(activation, feature_ids, axis=2)
    scales = []
    for column, item in enumerate(anchors.itertuples(index=False)):
        q99, status = positive_q99(selected[:, :, column])
        scales.append({
            "anchor_id": item.anchor_id, "seed": seed,
            "feature_id": int(getattr(item, feature_column)),
            "q99_scale": q99, "visualization_scale_status": status,
        })
    scale_frame = pd.DataFrame(scales)
    scale_lookup = scale_frame.set_index("anchor_id").q99_scale.to_dict()
    anchor_column = {anchor_id: index for index, anchor_id in enumerate(anchors.anchor_id)}
    config = protocol["independent_assets"]
    for case in seed_cases.itertuples(index=False):
        anchor_id = str(case.anchor_id)
        feature_id = int(anchors.loc[anchors.anchor_id.eq(anchor_id), feature_column].item())
        case_root = output_root / "assets" / anchor_id / "cases" / str(case.case_id)
        seed_root = case_root / f"seed{seed}"
        image_path = PROJECT_ROOT / str(case.image_relpath)
        raw = np.asarray(selected[int(case.image_index), :, anchor_column[anchor_id]]).reshape(7, 7)
        try:
            paths = save_case_assets(
                image_path, raw, float(scale_lookup[anchor_id]), seed_root,
                int(config["display_max_side_pixels"]), float(config["overlay_max_alpha"]),
                save_original=seed == 42,
            )
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
            add_asset_record(
                asset_records, output_root, anchor_id, str(case.case_id), seed, feature_id,
                "case_render", seed_root, float(scale_lookup[anchor_id]), "render_failure", str(error),
            )
            continue
        for asset_type, path in paths.items():
            add_asset_record(
                asset_records, output_root, anchor_id, str(case.case_id), seed, feature_id,
                asset_type, path, float(scale_lookup[anchor_id]),
            )
    del selected, activation
    gc.collect()
    return scale_frame


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(
        xy, text, fill=(20, 20, 20),
        font=ImageFont.truetype(str(TECHNICAL_FONT_PATH), TECHNICAL_FONT_SIZE),
    )


def technical_grid(
    anchor_id: str,
    cases: pd.DataFrame,
    output_root: Path,
    count: int,
    output_path: Path,
) -> None:
    """渲染Light/Heavy activation固定网格，缺失Zero使用占位块。"""
    tile = (220, 170)
    row_height = tile[1] + 26
    columns = count * 2
    canvas = Image.new("RGB", (150 + columns * tile[0], 45 + 8 * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (10, 10), anchor_id)
    row_index = 0
    for label in (1, 0):
        for bin_name in ("high", "mid", "low", "zero"):
            subset = cases[
                cases.label.eq(label) & cases.case_role.eq(bin_name)
            ].sort_values("selection_rank")
            draw_text(draw, (8, 52 + row_index * row_height), f"label={label} {bin_name}")
            for slot in range(count):
                x = 150 + slot * 2 * tile[0]
                y = 45 + row_index * row_height
                if slot < len(subset):
                    case = subset.iloc[slot]
                    case_root = output_root / "assets" / anchor_id / "cases" / case.case_id
                    original = case_root / "original.png"
                    overlay = case_root / "seed42/overlay.png"
                    if original.exists() and overlay.exists():
                        canvas.paste(thumbnail(original, tile), (x, y))
                        canvas.paste(thumbnail(overlay, tile), (x + tile[0], y))
                    else:
                        canvas.paste(placeholder((tile[0] * 2, tile[1]), "Render failure"), (x, y))
                else:
                    text = "No true-zero patient available\nFrozen selection shortfall"
                    canvas.paste(placeholder((tile[0] * 2, tile[1]), text), (x, y))
            row_index += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def cross_seed_panel(anchor_id: str, cases: pd.DataFrame, output_root: Path, output_path: Path) -> None:
    """对同一冻结病例并排显示原图和三个seed overlay。"""
    subset = cases[cases.case_role.isin(STANDARD_ROLES)].sort_values(
        ["label", "case_role", "selection_rank"], kind="stable",
    )
    tile = (180, 135)
    canvas = Image.new("RGB", (120 + 4 * tile[0], 35 + len(subset) * (tile[1] + 4)), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 10), f"{anchor_id} | original | seed42 | seed43 | seed44")
    for row_index, case in enumerate(subset.itertuples(index=False)):
        y = 35 + row_index * (tile[1] + 4)
        draw_text(draw, (5, y + 5), f"L{int(case.label)} {case.case_role} {int(case.selection_rank)}")
        root = output_root / "assets" / anchor_id / "cases" / str(case.case_id)
        paths = [root / "original.png", *(root / f"seed{s}/overlay.png" for s in SEEDS)]
        for column, path in enumerate(paths):
            image = thumbnail(path, tile) if path.exists() else placeholder(tile, "Render failure")
            canvas.paste(image, (120 + column * tile[0], y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def functional_profile(anchor_id: str, evidence: pd.DataFrame, output_path: Path) -> None:
    """绘制三seed五剂量及matched percentile技术图。"""
    alpha_labels = ["1p00", "0p75", "0p50", "0p25", "0p00"]
    alpha = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    metrics = [("cancer", "Cancer"), ("noncancer", "Non-cancer"), ("separation", "Separation")]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        for row in evidence.sort_values("seed").itertuples(index=False):
            values = [float(getattr(row, f"{metric}_alpha_{label}")) for label in alpha_labels]
            axis.plot(alpha, values, marker="o", label=f"seed{int(row.seed)}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("alpha retained")
        axis.set_ylabel("delta margin")
        axis.invert_xaxis()
        axis.legend(fontsize=7)
    percentiles = ", ".join(
        f"s{int(row.seed)}={float(row.matched_midrank_percentile):.2f}"
        for row in evidence.sort_values("seed").itertuples(index=False)
    )
    figure.suptitle(f"{anchor_id} | matched percentiles: {percentiles}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def pair_panel(anchor_id: str, cases: pd.DataFrame, output_root: Path, output_path: Path) -> None:
    """绘制Heavy high exemplar与冻结hard negative对照。"""
    negatives = cases[cases.case_role.eq("hard_negative")].sort_values(
        ["label", "selection_rank"], kind="stable",
    )
    tile = (180, 135)
    canvas = Image.new("RGB", (150 + 4 * tile[0], 35 + len(negatives) * (tile[1] + 5)), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 10), f"{anchor_id} | query original/overlay | comparison original/overlay")
    lookup = cases.set_index("case_id")
    for row_index, negative in enumerate(negatives.itertuples(index=False)):
        y = 35 + row_index * (tile[1] + 5)
        query = lookup.loc[str(negative.hard_negative_query_case_id)]
        draw_text(draw, (5, y + 5), f"L{int(negative.label)} {negative.hard_negative_pool}")
        query_root = output_root / "assets" / anchor_id / "cases" / str(query.name)
        neg_root = output_root / "assets" / anchor_id / "cases" / str(negative.case_id)
        paths = [
            query_root / "original.png", query_root / "seed42/overlay.png",
            neg_root / "original.png", neg_root / "seed42/overlay.png",
        ]
        for column, path in enumerate(paths):
            image = thumbnail(path, tile) if path.exists() else placeholder(tile, "Render failure")
            canvas.paste(image, (150 + column * tile[0], y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def source_panel(anchor_id: str, cases: pd.DataFrame, output_root: Path, output_path: Path) -> None:
    """绘制冻结label x source病例，不改变source-risk判定。"""
    subset = cases[cases.case_role.eq("source_panel")].sort_values(
        ["label", "source", "selection_rank"], kind="stable",
    )
    tile = (180, 135)
    canvas = Image.new("RGB", (220 + 2 * tile[0], 35 + len(subset) * (tile[1] + 4)), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 10), f"{anchor_id} source descriptive panel")
    for row_index, case in enumerate(subset.itertuples(index=False)):
        y = 35 + row_index * (tile[1] + 4)
        draw_text(draw, (5, y + 5), f"L{int(case.label)} {case.source} #{int(case.selection_rank)}")
        root = output_root / "assets" / anchor_id / "cases" / str(case.case_id)
        for column, path in enumerate([root / "original.png", root / "seed42/overlay.png"]):
            image = thumbnail(path, tile) if path.exists() else placeholder(tile, "Render failure")
            canvas.paste(image, (220 + column * tile[0], y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_blind_package(
    anchors: pd.DataFrame, cases: pd.DataFrame, output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建立不含label/source/技术字段的盲包及独立Reveal映射。"""
    anchor_order = sorted(
        anchors.anchor_id.astype(str), key=lambda value: blind_hash(value, "anchor"),
    )
    blind_anchor = {anchor_id: f"B{index:04d}" for index, anchor_id in enumerate(anchor_order, 1)}
    blind_rows, reveal_rows = [], []
    query_ids = set(
        cases.loc[cases.case_role.eq("hard_negative"), "hard_negative_query_case_id"]
        .dropna().astype(str)
    )
    included = cases[
        cases.in_light_atlas.astype(bool)
        | cases.case_role.eq("hard_negative")
        | cases.case_id.astype(str).isin(query_ids)
    ]
    for anchor_id, group in included.groupby("anchor_id", sort=True):
        ordered = group.assign(
            blind_order_hash=[blind_hash(anchor_id, str(case_id)) for case_id in group.case_id],
        ).sort_values("blind_order_hash", kind="stable")
        case_map = {
            case_id: f"C{index:04d}" for index, case_id in enumerate(ordered.case_id.astype(str), 1)
        }
        for row in ordered.itertuples(index=False):
            blind_id = case_map[str(row.case_id)]
            blind_root = output_root / "blind_review/cases" / blind_anchor[anchor_id] / blind_id
            blind_root.mkdir(parents=True, exist_ok=False)
            technical_root = output_root / "assets" / anchor_id / "cases" / str(row.case_id)
            paths = {}
            for name, source in {
                "original": technical_root / "original.png",
                "heatmap": technical_root / "seed42/heatmap.png",
                "overlay": technical_root / "seed42/overlay.png",
            }.items():
                target = blind_root / f"{name}.png"
                if source.exists():
                    os.link(source, target)
                    paths[name] = str(target.relative_to(output_root))
                else:
                    paths[name] = ""
            pair_id = ""
            if row.case_role == "hard_negative":
                pair_id = f"P-{case_map.get(str(row.hard_negative_query_case_id), 'UNMAPPED')}-{blind_id}"
            blind_rows.append({
                "blind_anchor_id": blind_anchor[anchor_id],
                "blind_case_id": blind_id,
                "activation_bin": "comparison" if row.case_role == "hard_negative" else row.activation_bin,
                **paths,
                "hard_negative_pair_id": pair_id,
                "blind_order_hash": blind_hash(anchor_id, str(row.case_id)),
            })
            reveal_rows.append({
                "blind_anchor_id": blind_anchor[anchor_id], "blind_case_id": blind_id,
                "anchor_id": anchor_id, "case_id": row.case_id,
                "label": int(row.label), "source": row.source,
                "case_role": row.case_role, "hard_negative_pool": row.hard_negative_pool,
                "in_light_atlas": bool(row.in_light_atlas),
            })
    return pd.DataFrame(blind_rows), pd.DataFrame(reveal_rows)


def blind_grid(
    blind_anchor_id: str,
    blind: pd.DataFrame,
    reveal: pd.DataFrame,
    output_root: Path,
    output_path: Path,
) -> None:
    """按冻结盲序混合标签渲染Light页，页面不读取诊断或source文本。"""
    light_ids = reveal[
        reveal.blind_anchor_id.eq(blind_anchor_id) & reveal.in_light_atlas.astype(bool)
    ].blind_case_id
    rows = blind[
        blind.blind_anchor_id.eq(blind_anchor_id) & blind.blind_case_id.isin(light_ids)
    ]
    tile = (220, 170)
    row_height = tile[1] + 26
    canvas = Image.new("RGB", (150 + 6 * 2 * tile[0], 45 + 4 * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (10, 10), f"{blind_anchor_id} | blind activation atlas")
    for row_index, bin_name in enumerate(("high", "mid", "low", "zero")):
        subset = rows[rows.activation_bin.eq(bin_name)].sort_values(
            "blind_order_hash", kind="stable",
        )
        draw_text(draw, (8, 52 + row_index * row_height), bin_name)
        for slot in range(6):
            x = 150 + slot * 2 * tile[0]
            y = 45 + row_index * row_height
            if slot < len(subset):
                case = subset.iloc[slot]
                original = output_root / str(case.original)
                overlay = output_root / str(case.overlay)
                if original.exists() and overlay.exists():
                    canvas.paste(thumbnail(original, tile), (x, y))
                    canvas.paste(thumbnail(overlay, tile), (x + tile[0], y))
                else:
                    canvas.paste(placeholder((tile[0] * 2, tile[1]), "Render failure"), (x, y))
            else:
                canvas.paste(placeholder((tile[0] * 2, tile[1]), "No eligible case"), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    """执行冻结RP-D v1渲染；任何失败均记录且不重选病例。"""
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"RP-D render输出已存在: {args.output_root}")
    protocol, anchors, cases = verify_inputs()
    anchors = select_debug_anchors(anchors, int(args.debug_anchor_limit))
    cases = cases[cases.anchor_id.isin(anchors.anchor_id)].copy()
    args.output_root.mkdir(parents=True)
    asset_records: list[dict] = []
    scale_frames = []
    for seed in SEEDS:
        scale_frames.append(render_seed_assets(
            seed, anchors, cases, protocol, args.output_root, asset_records,
        ))
    scales = pd.concat(scale_frames, ignore_index=True)
    scales.to_csv(args.output_root / "q99_scales.csv", index=False)

    evidence = pd.read_csv(RPC2_EVIDENCE, dtype={"anchor_id": str})
    for anchor in anchors.itertuples(index=False):
        anchor_cases = cases[cases.anchor_id.eq(anchor.anchor_id)]
        light_path = args.output_root / "light_atlas" / f"{anchor.anchor_id}.png"
        technical_grid(anchor.anchor_id, anchor_cases[anchor_cases.in_light_atlas], args.output_root, 3, light_path)
        add_asset_record(asset_records, args.output_root, anchor.anchor_id, "", None, None, "light_atlas", light_path)
        if bool(anchor.heavy_atlas):
            heavy_root = args.output_root / "heavy_atlas" / anchor.anchor_id
            paths = {
                "heavy_activation_atlas": heavy_root / "activation_atlas.png",
                "cross_seed_panel": heavy_root / "cross_seed.png",
                "functional_profile": heavy_root / "dose_curve.png",
                "hard_negative_panel": heavy_root / "hard_negative.png",
            }
            technical_grid(anchor.anchor_id, anchor_cases, args.output_root, 5, paths["heavy_activation_atlas"])
            cross_seed_panel(anchor.anchor_id, anchor_cases, args.output_root, paths["cross_seed_panel"])
            functional_profile(
                anchor.anchor_id, evidence[evidence.anchor_id.eq(anchor.anchor_id)],
                paths["functional_profile"],
            )
            pair_panel(anchor.anchor_id, anchor_cases, args.output_root, paths["hard_negative_panel"])
            if bool(anchor.heavy_source_risk):
                paths["source_panel"] = heavy_root / "source_panel.png"
                source_panel(anchor.anchor_id, anchor_cases, args.output_root, paths["source_panel"])
            for asset_type, path in paths.items():
                add_asset_record(asset_records, args.output_root, anchor.anchor_id, "", None, None, asset_type, path)

    blind, reveal = build_blind_package(anchors, cases, args.output_root)
    blind.to_csv(args.output_root / "blind_review/blind_manifest.csv", index=False)
    reveal_root = args.output_root / "technical_reveal"
    reveal_root.mkdir(parents=True)
    reveal.to_csv(reveal_root / "blind_to_technical_mapping.csv", index=False)
    technical_by_blind = reveal.set_index(
        ["blind_anchor_id", "blind_case_id"]
    ).anchor_id.to_dict()
    for row in blind.itertuples(index=False):
        technical_anchor = technical_by_blind[(row.blind_anchor_id, row.blind_case_id)]
        for asset_type in ("original", "heatmap", "overlay"):
            relative = getattr(row, asset_type)
            if relative:
                add_asset_record(
                    asset_records, args.output_root, technical_anchor, row.blind_case_id,
                    42, None, f"blind_{asset_type}", args.output_root / relative,
                )
    for blind_anchor_id in sorted(blind.blind_anchor_id.unique()):
        technical_anchor = reveal.loc[
            reveal.blind_anchor_id.eq(blind_anchor_id), "anchor_id"
        ].iloc[0]
        path = args.output_root / "blind_review/panels" / f"{blind_anchor_id}.png"
        blind_grid(blind_anchor_id, blind, reveal, args.output_root, path)
        add_asset_record(
            asset_records, args.output_root, technical_anchor, "", None, None,
            "blind_activation_atlas", path,
        )

    assets = pd.DataFrame(asset_records)
    assets.to_csv(args.output_root / "rpd_asset_manifest.csv", index=False)
    config = {
        "status": "rpd_render_debug_complete" if args.debug_anchor_limit else "rpd_render_complete",
        "debug_anchor_limit": int(args.debug_anchor_limit),
        "anchor_count": int(len(anchors)),
        "case_count": int(len(cases)),
        "asset_rows": int(len(assets)),
        "render_failure_rows": int(assets.asset_status.ne("ok").sum()),
        "blind_case_rows": int(len(blind)),
        "render_protocol_sha256": file_sha256(RENDER_PROTOCOL),
        "selection_freeze_sha256": file_sha256(SELECTION_ROOT / "selection_freeze_v1.json"),
        "asset_manifest_sha256": file_sha256(args.output_root / "rpd_asset_manifest.csv"),
        "blind_manifest_sha256": file_sha256(args.output_root / "blind_review/blind_manifest.csv"),
        "technical_font_path": str(TECHNICAL_FONT_PATH),
        "technical_font_sha256": file_sha256(TECHNICAL_FONT_PATH),
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
