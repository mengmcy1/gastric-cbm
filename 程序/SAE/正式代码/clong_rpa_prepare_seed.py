#!/usr/bin/env python3
"""为RP-A matching准备单个seed的train/val冻结分析缓存。

每个seed输出同一患者顺序的presence、ranking、activation mass、active
frequency和representation energy，以及图像级K=1024激活和decoder方向。
缓存只服务matching/bootstrap，不包含候选pair矩阵或p值矩阵。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpa_null_fdr import assign_target_strata, validate_target_strata
from clong_rpa_train_development import OUTPUT_ROOT as DEVELOPMENT_ROOT
from clong_s2c_core import K_LIST, MatryoshkaSparseAutoencoder
from clong_s2c_matryoshka import OUTPUT_ROOT as S2C_ROOT, file_sha256
from clong_sae_discovery import load_manifest_frame
from clong_s2b_core import ACTIVE_EPS
from clong_s2b_discovery import choose_device, load_cache


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = DEVELOPMENT_ROOT / "analysis_cache"
P_MIN = 25
A_MIN = 0.25 / 49.0
REPRESENTATION_K = 1024


def parse_args() -> argparse.Namespace:
    """解析seed、设备和隔离debug位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=2)
    parser.add_argument("--debug-epochs", type=int, default=1)
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace) -> Path:
    """按seed与模式定位已完成训练的checkpoint。"""
    if args.debug:
        return (
            DEVELOPMENT_ROOT / "debug"
            / f"p{args.debug_patients_per_class}_e{args.debug_epochs}"
            / f"rpa_development_clong_seed{args.seed}" / "sae_best.pth"
        )
    if args.seed == 42:
        return S2C_ROOT / "s2c_clong_seed42" / "sae_best.pth"
    return DEVELOPMENT_ROOT / f"rpa_development_clong_seed{args.seed}" / "sae_best.pth"


def load_sae(path: Path, device: torch.device) -> MatryoshkaSparseAutoencoder:
    """从checkpoint结构字段恢复Matryoshka字典。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["sae_state_dict"]
    encoder_shape = state["encoder.weight"].shape
    hidden_dim = int(encoder_shape[0])
    k_list = tuple(int(value) for value in payload.get("k_list", K_LIST))
    sae = MatryoshkaSparseAutoencoder(
        input_dim=int(encoder_shape[1]),
        hidden_dim=hidden_dim,
        feature_center=state["decoder_bias"],
        k_list=k_list,
    )
    sae.load_state_dict(state, strict=True)
    return sae.to(device).eval()


def patient_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """构造固定患者顺序并校验标签/来源唯一。"""
    conflicts = metadata.groupby("patient_id").agg(
        label_n=("label", "nunique"), source_n=("source", "nunique")
    )
    if (conflicts != 1).any().any():
        raise RuntimeError("同一患者label/source不唯一")
    return metadata.groupby("patient_id", sort=True).agg(
        label=("label", "first"), source=("source", "first"),
        image_count=("patient_id", "size"),
    ).reset_index()


@torch.no_grad()
def aggregate_split(
    spatial: np.ndarray, metadata: pd.DataFrame,
    sae: MatryoshkaSparseAutoencoder, device: torch.device, batch_size: int,
    activation_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """编码一个split并产生患者矩阵及图像级原始激活。"""
    patients = patient_table(metadata)
    patient_index = {str(pid): i for i, pid in enumerate(patients.patient_id)}
    n_patients, hidden = len(patients), sae.hidden_dim
    presence = np.zeros((n_patients, hidden), dtype=bool)
    ranking = np.zeros((n_patients, hidden), dtype=np.float32)
    mass = np.zeros((n_patients, hidden), dtype=np.float64)
    frequency = np.zeros((n_patients, hidden), dtype=np.float64)
    energy = np.zeros((n_patients, hidden), dtype=np.float64)
    image_count = np.zeros(n_patients, dtype=np.int32)
    image_activations = np.lib.format.open_memmap(
        activation_path, mode="w+", dtype=np.float32,
        shape=(len(metadata), 49, hidden),
    )
    decoder_norm2 = sae.decoder_weight.detach().square().sum(1).cpu().numpy()
    representation_k = int(max(sae.k_list))
    for start in range(0, len(metadata), batch_size):
        stop = min(start + batch_size, len(metadata))
        features = torch.from_numpy(np.asarray(spatial[start:stop]).copy()).to(device)
        current = sae.encode(features, k=representation_k).cpu().numpy()
        image_activations[start:stop] = current
        active = current > ACTIVE_EPS
        for local, row in enumerate(metadata.iloc[start:stop].itertuples(index=False)):
            index = patient_index[str(row.patient_id)]
            values = current[local]
            presence[index] |= active[local].any(0)
            ranking[index] = np.maximum(ranking[index], values.max(0))
            mass[index] += values.mean(0)
            frequency[index] += active[local].mean(0)
            energy[index] += np.square(values).mean(0) * decoder_norm2
            image_count[index] += 1
    if not np.array_equal(image_count, patients.image_count.to_numpy()):
        raise RuntimeError("患者图数聚合错位")
    divisor = image_count[:, None]
    image_activations.flush()
    matrices = {
        "presence": presence,
        "ranking": ranking,
        "mass": (mass / divisor).astype(np.float32),
        "active_frequency": (frequency / divisor).astype(np.float32),
        "energy": (energy / divisor).astype(np.float32),
    }
    return patients, matrices


def output_root(args: argparse.Namespace) -> Path:
    """返回seed缓存目录；debug按规模隔离。"""
    root = OUTPUT_ROOT
    if args.debug:
        root = root / "debug_v6" / f"p{args.debug_patients_per_class}_e{args.debug_epochs}"
    return root / f"seed{args.seed}"


def main() -> None:
    """加载冻结空间表示、编码RP-A统计并写入SHA绑定缓存。"""
    args = parse_args()
    if not args.debug and args.device != "cuda":
        raise ValueError("正式分析缓存必须使用CUDA")
    target = output_root(args)
    if target.exists():
        raise FileExistsError(f"分析缓存已存在，禁止覆盖: {target}")
    path = checkpoint_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"缺少seed checkpoint: {path}")
    device = choose_device(args.device)
    full_args = argparse.Namespace(debug=False, debug_patients_per_class=0, seed=args.seed)
    frame = load_manifest_frame(full_args)
    spatial, _pooled, _attention, metadata = load_cache(frame)
    sae = load_sae(path, device)
    if not args.debug and sae.hidden_dim != 10240:
        raise RuntimeError("正式RP-A字典宽度必须为10240")
    target.mkdir(parents=True)
    file_records = {}
    for split in ("train", "val"):
        split_metadata = metadata[split]
        split_spatial = spatial[split]
        if args.debug:
            selected = []
            for _, group in split_metadata[["patient_id", "label"]].drop_duplicates().groupby("label"):
                selected.extend(group.sample(
                    min(args.debug_patients_per_class, len(group)), random_state=42
                ).patient_id.astype(str).tolist())
            mask = split_metadata.patient_id.astype(str).isin(selected).to_numpy()
            indices = np.flatnonzero(mask)
            split_metadata = split_metadata.iloc[indices].reset_index(drop=True)
            split_spatial = np.asarray(split_spatial[indices])
        activation_path = target / f"{split}_image_activations.npy"
        patients, matrices = aggregate_split(
            split_spatial, split_metadata, sae, device, args.batch_size,
            activation_path,
        )
        patients_path = target / f"{split}_patients.csv"
        metadata_path = target / f"{split}_images.csv"
        patients.to_csv(patients_path, index=False)
        split_metadata[["relative_path", "patient_id", "label", "source"]].to_csv(
            metadata_path, index=False
        )
        for name, values in matrices.items():
            output = target / f"{split}_{name}.npy"
            np.save(output, values)
        file_records[split] = {
            name: file_sha256(target / f"{split}_{name}.npy") for name in matrices
        } | {
            "image_activations": file_sha256(activation_path),
            "patients": file_sha256(patients_path),
            "images": file_sha256(metadata_path),
        }

    train_presence = np.load(target / "train_presence.npy", mmap_mode="r")
    train_frequency = np.load(target / "train_active_frequency.npy", mmap_mode="r")
    nondead = np.asarray(train_presence).any(0)
    runtime_p_min = 1 if args.debug else P_MIN
    runtime_a_min = 0.0 if args.debug else A_MIN
    eligible = nondead & (np.asarray(train_presence).sum(0) >= runtime_p_min) & (
        np.asarray(train_frequency).mean(0) >= runtime_a_min
    )
    eligible_ids = np.flatnonzero(eligible)
    coverage = np.asarray(train_presence).sum(0)[eligible_ids] / train_presence.shape[0]
    active_frequency = np.asarray(train_frequency).mean(0)[eligible_ids]
    strata = assign_target_strata(coverage, active_frequency, eligible_ids)
    if not args.debug:
        validate_target_strata(strata)
    np.save(target / "train_eligible_ids.npy", eligible_ids)
    np.save(target / "train_eligible_strata.npy", strata)
    np.save(target / "decoder_weight.npy", sae.decoder_weight.detach().cpu().numpy())
    derived_files = {
        name: file_sha256(target / name)
        for name in (
            "train_eligible_ids.npy", "train_eligible_strata.npy", "decoder_weight.npy"
        )
    }
    config = {
        "stage": "RP-A per-seed analysis cache",
        "seed": args.seed,
        "debug": bool(args.debug),
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "formal_representation_k": REPRESENTATION_K,
        "runtime_representation_k": int(max(sae.k_list)),
        "formal_p_min": P_MIN,
        "runtime_p_min": runtime_p_min,
        "formal_a_min": A_MIN,
        "runtime_a_min": runtime_a_min,
        "hidden_dim": sae.hidden_dim,
        "eligible_feature_count": int(len(eligible_ids)),
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        "files": file_records,
        "derived_files": derived_files,
    }
    config_path = target / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    names = sorted(path.name for path in target.iterdir() if path.is_file())
    (target / "SHA256SUMS.txt").write_text(
        "\n".join(f"{file_sha256(target / name)}  {name}" for name in names) + "\n",
        encoding="utf-8",
    )
    print(f"seed{args.seed} RP-A分析缓存完成: {target}")


if __name__ == "__main__":
    main()
