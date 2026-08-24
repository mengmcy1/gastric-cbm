#!/usr/bin/env python3
"""RP-A development bootstrap worker：按静态index分工完整重算matching。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpa_bootstrap import (
    BOOTSTRAP_COUNT,
    generate_bootstrap_plans,
    structural_failure_record,
    validate_formal_patient_table,
    worker_replicate_indices,
)
from clong_rpa_match_development import (
    FORMAL_SPATIAL_SUPPORT,
    FORMAL_TOP_COUNT,
    compute_matching,
    load_seed,
    validate_cross_seed_alignment,
)
from clong_rpa_null_fdr import assign_target_strata, validate_target_strata
from clong_rpa_train_development import OUTPUT_ROOT as DEVELOPMENT_ROOT
OUTPUT_ROOT = DEVELOPMENT_ROOT / "bootstrap_workers"
P_MIN = 25
A_MIN = 0.25 / 49.0


def parse_args() -> argparse.Namespace:
    """解析worker静态分工、设备和debug规模。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-rank", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-block", type=int, default=32)
    parser.add_argument("--target-block", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-replicates", type=int, default=2)
    parser.add_argument("--debug-cache-version", default="debug_v6/p2_e1")
    return parser.parse_args()


def expanded_patient_instances(
    patients: pd.DataFrame, multiplicity: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """按patient_id、occurrence_index展开患者实例并生成稳定实例ID。"""
    indices, rows = [], []
    for index, row in patients.iterrows():
        for occurrence in range(int(multiplicity[index])):
            indices.append(index)
            rows.append({
                "patient_id": f"{row.patient_id}#occ{occurrence}",
                "label": int(row.label), "source": str(row.source),
                "image_count": int(row.image_count),
            })
    return np.asarray(indices, dtype=int), pd.DataFrame(rows)


def expanded_image_instances(
    images: pd.DataFrame, patients: pd.DataFrame, multiplicity: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """每次患者抽中都携带其全部图像，返回原图行索引与新元数据。"""
    by_patient = {
        str(patient): np.flatnonzero(images.patient_id.astype(str).eq(str(patient)).to_numpy())
        for patient in patients.patient_id
    }
    indices, rows = [], []
    for patient_index, row in patients.iterrows():
        patient = str(row.patient_id)
        for occurrence in range(int(multiplicity[patient_index])):
            instance = f"{patient}#occ{occurrence}"
            for image_index in by_patient[patient]:
                indices.append(int(image_index))
                current = images.iloc[image_index].to_dict()
                current["patient_id"] = instance
                rows.append(current)
    return np.asarray(indices, dtype=int), pd.DataFrame(rows)


def resampled_seed(
    base: dict, patient_indices: np.ndarray, expanded_patients: pd.DataFrame,
    image_indices: np.ndarray, expanded_images: pd.DataFrame, debug: bool,
) -> dict:
    """按同一multiplicity计划构造一个seed的replicate视图并重筛eligible。"""
    presence = np.asarray(base["presence"])[patient_indices]
    frequency = np.asarray(base["active_frequency"])[patient_indices]
    full_nondead = np.asarray(base["presence"]).any(0)
    p_min = 1 if debug else P_MIN
    a_min = 0.0 if debug else A_MIN
    eligible_mask = full_nondead & (presence.sum(0) >= p_min) & (
        frequency.mean(0) >= a_min
    )
    eligible_ids = np.flatnonzero(eligible_mask)
    coverage = presence[:, eligible_ids].mean(0)
    active_frequency = frequency[:, eligible_ids].mean(0)
    strata = assign_target_strata(coverage, active_frequency, eligible_ids)
    if not debug:
        validate_target_strata(strata)
    return {
        **base,
        "patients": expanded_patients,
        "images": expanded_images,
        "presence": presence,
        "ranking": np.asarray(base["ranking"])[patient_indices],
        "mass": np.asarray(base["mass"])[patient_indices],
        "energy": np.asarray(base["energy"])[patient_indices],
        "active_frequency": frequency,
        "eligible_ids": eligible_ids,
        "strata": strata,
        "image_indices": image_indices,
    }


def bootstrap_plans(patients: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    """正式使用冻结400计划；debug只返回少量确定性合法multiplicity。"""
    if not args.debug:
        validate_formal_patient_table(
            patients.patient_id.to_numpy(str), patients.label.to_numpy(int),
            patients.source.to_numpy(str),
        )
        return generate_bootstrap_plans(
            patients.patient_id.to_numpy(str), patients.label.to_numpy(int),
            patients.source.to_numpy(str),
        )
    plans = np.ones((args.debug_replicates, len(patients)), dtype=np.int32)
    if len(patients) >= 4 and args.debug_replicates > 1:
        plans[1, 0], plans[1, 1] = 2, 0
    return plans


def main() -> None:
    """执行本worker负责的replicate并逐行保存不可覆盖JSONL。"""
    args = parse_args()
    if args.worker_count < 1 or not 0 <= args.worker_rank < args.worker_count:
        raise ValueError("worker rank/count非法")
    if not args.debug and args.device != "cuda":
        raise ValueError("正式bootstrap worker必须使用CUDA")
    target = OUTPUT_ROOT / ("debug" if args.debug else "formal")
    target.mkdir(parents=True, exist_ok=True)
    output = target / f"worker_{args.worker_rank:02d}_of_{args.worker_count:02d}.jsonl"
    if output.exists():
        raise FileExistsError(f"worker输出已存在，禁止覆盖: {output}")
    seeds = {seed: load_seed(seed, args) for seed in (42, 43, 44)}
    validate_cross_seed_alignment(seeds)
    patients = seeds[42]["patients"]
    plans = bootstrap_plans(patients, args)
    if args.debug:
        indices = np.arange(len(plans))[np.arange(len(plans)) % args.worker_count == args.worker_rank]
    else:
        indices = worker_replicate_indices(args.worker_rank, args.worker_count)
    device = torch.device(args.device)
    with output.open("x", encoding="utf-8") as handle:
        for replicate_index in indices:
            multiplicity = plans[int(replicate_index)]
            patient_indices, expanded_patients = expanded_patient_instances(patients, multiplicity)
            image_indices, expanded_images = expanded_image_instances(
                seeds[42]["images"], patients, multiplicity
            )
            try:
                sampled = {
                    seed: resampled_seed(
                        data, patient_indices, expanded_patients,
                        image_indices, expanded_images, args.debug,
                    )
                    for seed, data in seeds.items()
                }
                _hypotheses, _edges, _anchors, metrics = compute_matching(
                    sampled, device, args.source_block, args.target_block,
                    min(FORMAL_TOP_COUNT, len(expanded_patients)) if args.debug else FORMAL_TOP_COUNT,
                    min(2, len(expanded_patients)) if args.debug else FORMAL_SPATIAL_SUPPORT,
                    validate_strata=not args.debug, progress=False,
                )
                record = {
                    "replicate_index": int(replicate_index),
                    "status": "completed",
                    "fold_metrics": metrics,
                }
            except (ValueError, RuntimeError) as error:
                message = str(error)
                if not (
                    message.startswith("null_stratification_infeasible:")
                    or message == "strata分箱数无效或Feature不足"
                ):
                    raise
                record = structural_failure_record(
                    int(replicate_index), "null_stratification_infeasible"
                )
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"worker {args.worker_rank}: replicate {replicate_index} DONE", flush=True)
    print(f"bootstrap worker完成: {output}")


if __name__ == "__main__":
    main()
