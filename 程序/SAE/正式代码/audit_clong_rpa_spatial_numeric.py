#!/usr/bin/env python3
"""只读比较RP-A spatial三种数值路径，不输出任何matching统计。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpa_eligible import SPATIAL_CACHE, file_sha256, load_sae, validate_lineage
from clong_rpa_spatial import spatial_pair_matrix
from clong_s2b_core import ACTIVE_EPS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path(__file__).with_name("rpa_spatial_protocol_v1.json")
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "结果/SAE/RP_A_Spatial数值审计_20260824/numerical_audit.json"
)


def parse_args() -> argparse.Namespace:
    """解析设备和唯一正式输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def cpu_float64_reference(
    source: np.ndarray, target: np.ndarray, image_patient_index: np.ndarray,
    patient_count: int,
) -> np.ndarray:
    """以CPU float64逐pair复算候选spatial，作为numerical reference。"""
    output = np.full((source.shape[2], target.shape[2]), np.nan, dtype=np.float64)
    source64, target64 = source.astype(np.float64), target.astype(np.float64)
    source_active = source64.max(axis=1) > ACTIVE_EPS
    target_active = target64.max(axis=1) > ACTIVE_EPS
    for s in range(source.shape[2]):
        for t in range(target.shape[2]):
            patient_values = []
            for patient in range(patient_count):
                image_ids = np.flatnonzero(image_patient_index == patient)
                union = source_active[image_ids, s] | target_active[image_ids, t]
                if not union.any():
                    continue
                image_values = []
                for image_id in image_ids[union]:
                    if source_active[image_id, s] and target_active[image_id, t]:
                        left = source64[image_id, :, s]
                        right = target64[image_id, :, t]
                        image_values.append(float(np.dot(left, right) /
                                                  (np.linalg.norm(left) * np.linalg.norm(right))))
                    else:
                        image_values.append(0.0)
                patient_values.append(float(np.mean(image_values, dtype=np.float64)))
            if patient_values:
                output[s, t] = float(np.mean(patient_values, dtype=np.float64))
    return output


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """只报告误差和finite/NA一致性，不暴露spatial统计值。"""
    finite_match = np.array_equal(np.isfinite(candidate), np.isfinite(reference))
    valid = np.isfinite(reference)
    difference = np.abs(candidate[valid] - reference[valid])
    return {
        "finite_or_na_semantics_match": bool(finite_match),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
    }


def main() -> None:
    """运行固定小块审计并保存仅含数值工程信息的JSON。"""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"正式数值审计已存在: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    validate_lineage()
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
    selected_metadata = metadata.loc[selected].reset_index(drop=True)
    patient_index = {patient_id: idx for idx, patient_id in enumerate(patient_ids)}
    image_patient_index_np = np.asarray(
        [patient_index[patient_id] for patient_id in selected_metadata["patient_id"]],
        dtype=np.int64,
    )
    image_patient_index = torch.from_numpy(image_patient_index_np).to(device)

    sae = load_sae(device)
    hidden_batches = []
    with torch.no_grad():
        for start in range(0, len(selected_spatial), 8):
            features = torch.from_numpy(selected_spatial[start:start + 8]).to(device)
            hidden_batches.append(sae.encode(features, k=1024))
    hidden = torch.cat(hidden_batches, dim=0)
    rng = np.random.Generator(np.random.PCG64(20260824))
    feature_ids = rng.choice(10240, size=96, replace=False)
    source = hidden[:, :, feature_ids[:48]].contiguous()
    target = hidden[:, :, feature_ids[48:]].contiguous()

    # 两条GPU路径先各预热一次，再交替重复计时，避免把CUDA冷启动算入某一路径。
    for dtype in (torch.float32, torch.float64):
        spatial_pair_matrix(source, target, image_patient_index, len(patient_ids), dtype)
    paths = {
        "float32_all": {"elapsed": [], "peak_vram_bytes": []},
        "mixed": {"elapsed": [], "peak_vram_bytes": []},
    }
    dtype_by_name = {"float32_all": torch.float32, "mixed": torch.float64}
    values_by_name = {}
    for _ in range(7):
        for name in ("float32_all", "mixed"):
            dtype = dtype_by_name[name]
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            values, _ = spatial_pair_matrix(
                source, target, image_patient_index, len(patient_ids), dtype
            )
            torch.cuda.synchronize(device)
            paths[name]["elapsed"].append(time.perf_counter() - started)
            paths[name]["peak_vram_bytes"].append(int(torch.cuda.max_memory_allocated(device)))
            values_by_name[name] = values

    source_cpu = source.cpu().numpy()
    target_cpu = target.cpu().numpy()
    started = time.perf_counter()
    reference = cpu_float64_reference(
        source_cpu, target_cpu, image_patient_index_np, len(patient_ids)
    )
    reference_seconds = time.perf_counter() - started
    output = {
        "status": "numerical_audit_complete_no_matching_statistics",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "lineage": validate_lineage(),
        "fixed_selection": {
            "patient_count": len(patient_ids),
            "image_count": len(selected_metadata),
            "source_feature_count": 48,
            "target_feature_count": 48,
            "selection_rule": protocol["numerical_audit"],
        },
        "tf32_disabled": True,
        "float32_all_vs_cpu_float64": {
            **compare(values_by_name["float32_all"], reference),
            "median_elapsed_seconds": float(np.median(paths["float32_all"]["elapsed"])),
            "peak_vram_bytes": max(paths["float32_all"]["peak_vram_bytes"]),
        },
        "mixed_vs_cpu_float64": {
            **compare(values_by_name["mixed"], reference),
            "median_elapsed_seconds": float(np.median(paths["mixed"]["elapsed"])),
            "peak_vram_bytes": max(paths["mixed"]["peak_vram_bytes"]),
        },
        "cpu_float64_reference": {"elapsed_seconds": reference_seconds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
