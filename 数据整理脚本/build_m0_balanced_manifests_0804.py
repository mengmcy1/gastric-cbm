#!/usr/bin/env python3
"""从冻结M0全量Keep清单构建来源内风格匹配的1:1.3与1:1清单。"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "06_M0全量诊断清单_20260805/m0_full_keep_split_seed42.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "07_M0来源内平衡清单_20260805"
)
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
SIZE_ORDER = {
    "small_le640": 0,
    "medium_641_1024": 1,
    "large_1025_1600": 2,
    "xlarge_gt1600": 3,
}
ASPECT_ORDER = {"portrait": 0, "square": 1, "landscape": 2, "wide": 3}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--control-ratio", type=float, default=1.3)
    parser.add_argument("--max-images-per-patient", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_key(value, seed, salt):
    return hashlib.sha256(f"{seed}|{salt}|{value}".encode()).hexdigest()


def stable_order(values, seed, salt):
    return sorted(values, key=lambda value: stable_key(value, seed, salt))


def stable_mode(values):
    counts = Counter(values)
    maximum = max(counts.values())
    return sorted(value for value, count in counts.items() if count == maximum)[0]


def patient_table(frame, max_images):
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        identity = group[["split", "source", "label"]].drop_duplicates()
        if len(identity) != 1:
            raise ValueError(f"患者身份不唯一: {patient_id}")
        item = identity.iloc[0]
        dominant_style = stable_mode(group.style_group.astype(str))
        style_images = group.loc[group.style_group.eq(dominant_style)]
        rows.append({
            "patient_id": patient_id,
            "split": item.split,
            "source": item.source,
            "label": int(item.label),
            "dominant_style_group": dominant_style,
            "available_images": int(len(group)),
            "dominant_style_images": int(len(style_images)),
            "image_capacity": int(min(max_images, len(style_images))),
        })
    return pd.DataFrame(rows)


def integer_quotas(counts, total):
    counts = counts.astype(float)
    raw = counts / counts.sum() * total
    quotas = np.floor(raw).astype(int)
    remainder = int(total - quotas.sum())
    ranking = sorted(
        counts.index,
        key=lambda style: (-(raw[style] - quotas[style]), str(style)),
    )
    for style in ranking[:remainder]:
        quotas[style] += 1
    return quotas


def style_cost(left, right):
    left_size, left_aspect, left_frame = left.split("|", 2)
    right_size, right_aspect, right_frame = right.split("|", 2)
    return (
        2 * abs(SIZE_ORDER.get(left_size, 9) - SIZE_ORDER.get(right_size, 9))
        + 2 * abs(ASPECT_ORDER.get(left_aspect, 9) - ASPECT_ORDER.get(right_aspect, 9))
        + 3 * int(left_frame != right_frame)
    )


def select_control_patients(cases, controls, ratio, seed, stratum):
    target_total = min(len(controls), int(round(len(cases) * ratio)))
    if target_total < 1:
        raise ValueError(f"分层无可选非癌患者: {stratum}")
    case_counts = cases.dominant_style_group.value_counts().sort_index()
    quotas = integer_quotas(case_counts, target_total)
    available = {
        style: stable_order(
            group.patient_id.tolist(), seed, f"{stratum}|exact|{style}"
        )
        for style, group in controls.groupby("dominant_style_group", sort=True)
    }
    selected = []
    missing_targets = []
    used = set()
    for style, quota in quotas.items():
        candidates = available.get(style, [])
        exact = candidates[: min(quota, len(candidates))]
        selected.extend(
            (patient_id, style, "exact_style", 0) for patient_id in exact
        )
        used.update(exact)
        missing_targets.extend([style] * (quota - len(exact)))

    remaining = controls.loc[~controls.patient_id.isin(used)].copy()
    for index, target_style in enumerate(missing_targets):
        if remaining.empty:
            raise ValueError(f"非癌候选不足: {stratum}")
        remaining["_cost"] = remaining.dominant_style_group.map(
            lambda style: style_cost(target_style, style)
        )
        minimum = remaining._cost.min()
        candidates = remaining.loc[remaining._cost.eq(minimum), "patient_id"].tolist()
        chosen = stable_order(
            candidates, seed, f"{stratum}|fallback|{target_style}|{index}"
        )[0]
        chosen_row = remaining.loc[remaining.patient_id.eq(chosen)].iloc[0]
        selected.append((chosen, target_style, "nearest_style", int(minimum)))
        remaining = remaining.loc[~remaining.patient_id.eq(chosen)].copy()

    result = pd.DataFrame(
        selected,
        columns=["patient_id", "target_style_group", "match_level", "style_cost"],
    )
    if len(result) != target_total or result.patient_id.duplicated().any():
        raise ValueError(f"非癌患者选择数量异常: {stratum}")
    return result


def select_patients(patients, ratio, seed):
    rows = []
    for (split, source), group in patients.groupby(["split", "source"], sort=True):
        cases = group.loc[group.label.eq(1)].copy()
        controls = group.loc[group.label.eq(0)].copy()
        if cases.empty or controls.empty:
            raise ValueError(f"分层缺少标签: {split}/{source}")
        cases["target_style_group"] = cases.dominant_style_group
        cases["match_level"] = "case"
        cases["style_cost"] = 0
        cases["match_role"] = "case"
        rows.append(cases)
        chosen = select_control_patients(
            cases,
            controls,
            ratio,
            seed,
            f"{split}|{source}|ratio={ratio}",
        )
        chosen = chosen.merge(controls, on="patient_id", validate="one_to_one")
        chosen["match_role"] = "control"
        rows.append(chosen)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(
        ["split", "source", "label", "patient_id"],
        key=lambda column: column.map(SPLIT_ORDER) if column.name == "split" else column,
    ).reset_index(drop=True)


def allocate_image_counts(group, target_total, seed, salt):
    capacities = group.set_index("patient_id").image_capacity.astype(int).to_dict()
    counts = {patient_id: 1 for patient_id in capacities}
    if target_total < len(counts) or target_total > sum(capacities.values()):
        raise ValueError(f"图片目标超出容量: {salt}/{target_total}")
    remaining = target_total - len(counts)
    for rank in range(2, max(capacities.values()) + 1):
        candidates = [
            patient_id for patient_id, capacity in capacities.items()
            if capacity >= rank
        ]
        for patient_id in stable_order(candidates, seed, f"{salt}|rank={rank}"):
            if remaining == 0:
                return counts
            counts[patient_id] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"图片分配未完成: {salt}/{remaining}")
    return counts


def evenly_spaced_rows(group, count):
    group = group.sort_values("relative_path", kind="stable")
    if count == 1:
        return group.iloc[[len(group) // 2]].copy()
    indices = np.linspace(0, len(group) - 1, count).round().astype(int)
    return group.iloc[indices].copy()


def select_images(frame, selected_patients, ratio, max_images, seed):
    patient_lookup = selected_patients.set_index("patient_id")
    if not patient_lookup.index.is_unique:
        raise ValueError("入选患者表存在重复patient_id")
    selected_rows = []
    for (split, source), patients in selected_patients.groupby(
        ["split", "source"], sort=True
    ):
        cases = patients.loc[patients.label.eq(1)]
        controls = patients.loc[patients.label.eq(0)]
        case_capacity = int(cases.image_capacity.sum())
        control_capacity = int(controls.image_capacity.sum())
        case_total = min(case_capacity, int(np.floor(control_capacity / ratio)))
        case_total = max(case_total, len(cases))
        control_total = min(control_capacity, int(round(case_total * ratio)))
        if control_total < len(controls):
            control_total = len(controls)
            case_total = min(case_capacity, int(round(control_total / ratio)))
        targets = {1: case_total, 0: control_total}
        count_by_patient = {}
        for label, label_patients in patients.groupby("label", sort=True):
            count_by_patient.update(
                allocate_image_counts(
                    label_patients,
                    targets[int(label)],
                    seed,
                    f"{split}|{source}|label={label}|ratio={ratio}",
                )
            )
        for patient_id, count in count_by_patient.items():
            signature = patient_lookup.loc[patient_id]
            candidates = frame.loc[
                frame.patient_id.eq(patient_id)
                & frame.style_group.eq(signature.dominant_style_group)
            ]
            chosen = evenly_spaced_rows(candidates, count)
            chosen["dominant_style_group"] = signature.dominant_style_group
            chosen["target_style_group"] = signature.target_style_group
            chosen["match_role"] = signature.match_role
            chosen["match_level"] = signature.match_level
            chosen["style_cost"] = int(signature.style_cost)
            chosen["patient_image_target"] = count
            chosen["selection_rank"] = range(1, len(chosen) + 1)
            selected_rows.append(chosen)
    result = pd.concat(selected_rows, ignore_index=True)
    if result.groupby("patient_id").size().gt(max_images).any():
        raise ValueError("患者图片数超过上限")
    return result.sort_values(
        ["split", "source", "label", "patient_id", "selection_rank"],
        key=lambda column: column.map(SPLIT_ORDER) if column.name == "split" else column,
    ).reset_index(drop=True)


def total_variation(frame, column):
    table = pd.crosstab(frame[column], frame.label, normalize="columns")
    if set(table.columns) != {0, 1}:
        return np.nan
    return float(0.5 * np.abs(table[0] - table[1]).sum())


def audit_dataset(frame, ratio_name):
    patient_frame = frame.drop_duplicates("patient_id")
    rows = []
    for (split, source), images in frame.groupby(["split", "source"], sort=True):
        patients = images.drop_duplicates("patient_id")
        counts = patients.label.value_counts().to_dict()
        image_counts = images.label.value_counts().to_dict()
        rows.append({
            "dataset": ratio_name,
            "split": split,
            "source": source,
            "cancer_patients": int(counts.get(1, 0)),
            "control_patients": int(counts.get(0, 0)),
            "patient_control_case_ratio": counts.get(0, 0) / counts.get(1, 1),
            "cancer_images": int(image_counts.get(1, 0)),
            "control_images": int(image_counts.get(0, 0)),
            "image_control_case_ratio": image_counts.get(0, 0) / image_counts.get(1, 1),
            "patient_style_tv": total_variation(patients, "dominant_style_group"),
            "image_style_tv": total_variation(images, "style_group"),
            "nearest_style_controls": int(
                patients.match_level.eq("nearest_style").sum()
            ),
        })
    rows.append({
        "dataset": ratio_name,
        "split": "all",
        "source": "all",
        "cancer_patients": int(patient_frame.label.eq(1).sum()),
        "control_patients": int(patient_frame.label.eq(0).sum()),
        "patient_control_case_ratio": (
            patient_frame.label.eq(0).sum() / patient_frame.label.eq(1).sum()
        ),
        "cancer_images": int(frame.label.eq(1).sum()),
        "control_images": int(frame.label.eq(0).sum()),
        "image_control_case_ratio": frame.label.eq(0).sum() / frame.label.eq(1).sum(),
        "patient_style_tv": total_variation(patient_frame, "dominant_style_group"),
        "image_style_tv": total_variation(frame, "style_group"),
        "nearest_style_controls": int(
            patient_frame.match_level.eq("nearest_style").sum()
        ),
    })
    return pd.DataFrame(rows)


def validate_manifest(frame):
    if frame.relative_path.duplicated().any():
        raise ValueError("平衡清单图片重复")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("患者跨split")
    if frame.groupby("sha256").split.nunique().gt(1).any():
        raise ValueError("相同SHA跨split")
    if not frame.image_relpath.map(lambda path: (PROJECT_ROOT / path).is_file()).all():
        raise FileNotFoundError("平衡清单存在缺失处理图")
    for _, group in frame.groupby(["split", "source"]):
        if set(group.label) != {0, 1}:
            raise ValueError("某个split×source缺少标签")


def build_dataset(frame, ratio, max_images, seed):
    patients = patient_table(frame, max_images)
    selected_patients = select_patients(patients, ratio, seed)
    selected_images = select_images(
        frame, selected_patients, ratio, max_images, seed
    )
    validate_manifest(selected_images)
    return selected_images, selected_patients


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")
    if args.control_ratio < 1:
        raise ValueError("--control-ratio必须>=1")
    if args.max_images_per_patient < 1:
        raise ValueError("--max-images-per-patient必须>=1")

    frame = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"patient_id": str})
    outputs = []
    for name, ratio in (("primary_1to1p3", args.control_ratio), ("sensitivity_1to1", 1.0)):
        images, patients = build_dataset(
            frame, ratio, args.max_images_per_patient, args.seed
        )
        outputs.append((name, ratio, images, patients, audit_dataset(images, name)))

    args.output.mkdir(parents=True, exist_ok=False)
    audits = []
    for name, ratio, images, patients, audit in outputs:
        images.to_csv(
            args.output / f"m0_balanced_keep_{name}_split_seed42.csv",
            index=False,
            encoding="utf-8-sig",
        )
        patients.to_csv(
            args.output / f"selected_patients_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        audits.append(audit)
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(args.output / "balance_audit.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input": str(args.input.resolve()),
        "split_policy": "reuse_frozen_m0_full_patient_split_seed42",
        "primary_patient_ratio": f"1:{args.control_ratio}",
        "sensitivity_patient_ratio": "1:1",
        "matching_scope": "within_split_and_source",
        "style_policy": "dominant_style_quota_then_nearest_style_fallback",
        "image_policy": (
            "dominant-style images only; 1-3 deterministic evenly spaced frames; "
            "equalize mean images per patient across labels"
        ),
        "max_images_per_patient": args.max_images_per_patient,
        "selection_seed": args.seed,
        "external_216": "remains_excluded",
        "test_policy": "manifest_frozen_now_but_not_evaluated_during_model_selection",
    }
    (args.output / "pre_registered_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit.to_string(index=False))
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
