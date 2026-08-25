#!/usr/bin/env python3
"""只计算RP-A-lite固定缺失index或replicate 53等价性探针。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from clong_rpa_bootstrap import structural_failure_record
from clong_rpa_bootstrap_worker import (
    bootstrap_plans,
    expanded_image_instances,
    expanded_patient_instances,
    resampled_seed,
)
from clong_rpa_lite_core import (
    EQUIVALENCE_INDEX,
    MISSING_INDICES,
    NEW_TARGET_BLOCK,
    RETRY_MISSING_INDICES,
    RETRY_TARGET_BLOCK,
    SOURCE_BLOCK,
)
from clong_rpa_match_development import (
    FORMAL_SPATIAL_SUPPORT,
    FORMAL_TOP_COUNT,
    compute_matching,
    load_seed,
    validate_cross_seed_alignment,
)
from clong_rpa_train_development import OUTPUT_ROOT as SOURCE_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SOURCE_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Lite_Exploratory_20260825"


def parse_args() -> argparse.Namespace:
    """解析Lite工作模式和静态worker分工。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("missing", "equivalence", "retry1", "equivalence64"),
        required=True,
    )
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--source-block", type=int, default=SOURCE_BLOCK)
    parser.add_argument("--target-block", type=int, default=NEW_TARGET_BLOCK)
    parser.set_defaults(
        debug=False,
        debug_replicates=2,
        debug_cache_version="debug_v6/p2_e1",
    )
    return parser.parse_args()


def assigned_indices(args: argparse.Namespace) -> tuple[int, ...]:
    """返回本worker唯一允许计算的Lite index。"""
    if args.mode in {"equivalence", "equivalence64"}:
        if args.worker_rank != 0 or args.worker_count != 1:
            raise ValueError("等价性探针必须使用rank0/count1")
        return (EQUIVALENCE_INDEX,)
    if args.mode == "retry1":
        if args.worker_count != 3 or not 0 <= args.worker_rank < 3:
            raise ValueError("Lite retry1必须固定使用3个worker")
        return tuple(RETRY_MISSING_INDICES[args.worker_rank::args.worker_count])
    if args.worker_count != 3 or not 0 <= args.worker_rank < 3:
        raise ValueError("Lite缺失index必须固定使用3个worker")
    return tuple(MISSING_INDICES[args.worker_rank::args.worker_count])


def main() -> None:
    """从首轮冻结cache完整重算指定Lite replicate。"""
    args = parse_args()
    if SOURCE_ROOT.resolve() != EXPECTED_SOURCE_ROOT.resolve():
        raise RuntimeError(f"Lite必须读取首轮源根: {EXPECTED_SOURCE_ROOT}")
    expected_target_block = (
        RETRY_TARGET_BLOCK if args.mode in {"retry1", "equivalence64"}
        else NEW_TARGET_BLOCK
    )
    if args.source_block != SOURCE_BLOCK or args.target_block != expected_target_block:
        raise ValueError(f"Lite {args.mode}计算块不是32x{expected_target_block}")
    indices = assigned_indices(args)
    target = OUTPUT_ROOT / (
        "equivalence"
        if args.mode in {"equivalence", "equivalence64"}
        else "missing_workers" if args.mode == "missing" else "missing_retry1"
    )
    target.mkdir(parents=True, exist_ok=True)
    name = (
        f"replicate53_32x{expected_target_block}.jsonl"
        if args.mode in {"equivalence", "equivalence64"}
        else f"worker_{args.worker_rank:02d}_of_{args.worker_count:02d}.jsonl"
    )
    output = target / name
    partial = output.with_suffix(".partial.jsonl")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Lite worker输出或残缺现场已存在: {output}")

    seeds = {seed: load_seed(seed, args) for seed in (42, 43, 44)}
    validate_cross_seed_alignment(seeds)
    patients = seeds[42]["patients"]
    plans = bootstrap_plans(patients, args)
    device = torch.device(args.device)
    with partial.open("x", encoding="utf-8") as handle:
        for replicate_index in indices:
            multiplicity = plans[replicate_index]
            patient_indices, expanded_patients = expanded_patient_instances(
                patients, multiplicity
            )
            image_indices, expanded_images = expanded_image_instances(
                seeds[42]["images"], patients, multiplicity
            )
            try:
                sampled = {
                    seed: resampled_seed(
                        data,
                        patient_indices,
                        expanded_patients,
                        image_indices,
                        expanded_images,
                        False,
                    )
                    for seed, data in seeds.items()
                }
                _hypotheses, _edges, _anchors, metrics = compute_matching(
                    sampled,
                    device,
                    args.source_block,
                    args.target_block,
                    FORMAL_TOP_COUNT,
                    FORMAL_SPATIAL_SUPPORT,
                    validate_strata=True,
                    progress=False,
                )
                record = {
                    "replicate_index": replicate_index,
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
                    replicate_index, "null_stratification_infeasible"
                )
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"Lite worker {args.worker_rank}: replicate {replicate_index} DONE", flush=True)
    os.replace(partial, output)
    print(f"RP-A-lite worker完成: {output}")


if __name__ == "__main__":
    main()
