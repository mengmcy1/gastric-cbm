#!/usr/bin/env python3
"""用固定seed42 train-only patch输入校准RP-A GPU identity容差。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpa_eligible import SPATIAL_CACHE, file_sha256, load_sae, validate_lineage
from clong_rpa_gpu_identity import (
    LEVELS,
    ErrorAccumulator,
    derive_tolerances,
    downstream_levels,
    identity_passes,
    residual_preserving_noop,
)
from clong_sae_discovery import CLONG_CHECKPOINT, load_clong_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path(__file__).with_name("rpa_gpu_identity_protocol_v1.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/SAE/RP_A_GPU_Identity_20260824/identity_audit.json"


def parse_args() -> argparse.Namespace:
    """解析唯一正式输出位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """运行固定no-op路径、生成逐级误差和冻结容差候选。"""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"identity audit已存在: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    lineage = validate_lineage()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    metadata = pd.read_csv(SPATIAL_CACHE / "train_metadata.csv", dtype={"patient_id": str})
    patient_ids = np.sort(metadata["patient_id"].unique())[:32]
    selected = metadata["patient_id"].isin(patient_ids).to_numpy()
    spatial = np.load(SPATIAL_CACHE / "train_spatial_features.npy", mmap_mode="r")
    selected_spatial = np.array(spatial[selected], copy=True)
    sae = load_sae(device)
    model = load_clong_model(device)
    accumulators = {level: ErrorAccumulator() for level in LEVELS}

    for start in range(0, len(selected_spatial), 32):
        features = torch.from_numpy(selected_spatial[start:start + 32]).to(device)
        no_op = residual_preserving_noop(features, sae, k=1024)
        reference = downstream_levels(features, model)
        candidate = downstream_levels(no_op, model)
        for level in LEVELS:
            accumulators[level].update(reference[level], candidate[level])

    summaries = {level: accumulators[level].summary() for level in LEVELS}
    tolerances = derive_tolerances(summaries)
    # 以同一固定输入复核生成后的逐元素gate，而不只比较汇总最大值。
    gate = {level: True for level in LEVELS}
    for start in range(0, len(selected_spatial), 32):
        features = torch.from_numpy(selected_spatial[start:start + 32]).to(device)
        no_op = residual_preserving_noop(features, sae, k=1024)
        reference = downstream_levels(features, model)
        candidate = downstream_levels(no_op, model)
        for level in LEVELS:
            tolerance = tolerances[level]
            gate[level] &= identity_passes(
                reference[level], candidate[level], tolerance["atol"], tolerance["rtol"]
            )
    if not all(gate.values()):
        raise RuntimeError("固定audit输入未通过生成后的identity gate")

    payload = {
        "status": "identity_audit_complete_candidate_tolerances",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "lineage": lineage,
        "clong_checkpoint_sha256": file_sha256(CLONG_CHECKPOINT),
        "fixed_input": {
            "patient_count": len(patient_ids),
            "image_count": int(selected.sum()),
            **protocol["fixed_input"],
        },
        "environment": {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "nvidia_driver_version": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines()[0].strip(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "tf32_disabled": True,
        },
        "level_errors": summaries,
        "candidate_tolerances": tolerances,
        "fixed_input_gate_pass": gate,
        "test_internal_test_external_evaluated": False,
        "matching_statistics_generated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
