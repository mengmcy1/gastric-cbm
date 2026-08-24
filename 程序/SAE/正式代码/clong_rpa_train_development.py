#!/usr/bin/env python3
"""训练 RP-A development-calibration 的 seed43/44 Matryoshka SAE。

本入口复用 S2c 冻结的表示、模型、五层联合损失和训练预算，但不把
seed43/44解释为新的 S2c 产品。训练得到的字典只用于 RP-A 跨初始化
matching；后续表示固定读取 K=1024。gamma_pool 继承 seed42 正式校准
JSON，禁止新 seed 重校准。test/internal test/external 均不读取。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_s2c_core import K_LIST
from clong_s2c_matryoshka import (
    FORMAL_BUDGET,
    GAMMA_JSON_NAME,
    OUTPUT_ROOT as S2C_ROOT,
    evaluate_all_k,
    file_sha256,
    initialize_matryoshka,
    initialization_sha,
    train_matryoshka,
)
from clong_sae_discovery import (
    CLONG_CHECKPOINT_SHA256,
    MANIFEST_SHA256,
    load_clong_model,
    load_manifest_frame,
    verify_s0_lineage,
)
from clong_s2b_discovery import choose_device, load_cache
from train_utils import git_snapshot, json_ready


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824"
GAMMA_JSON = S2C_ROOT / GAMMA_JSON_NAME
GAMMA_JSON_SHA256 = "3e427764415873cb5a3e6d7f8259350c96a968f48e82bd31c37a030b1dc7f35f"
REPRESENTATION_K = 1024
FORMAL_SEEDS = (43, 44)


def parse_args() -> argparse.Namespace:
    """解析正式 seed 和隔离 debug 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=2)
    parser.add_argument("--debug-hidden-dim", type=int, default=64)
    parser.add_argument("--debug-ks", default="4,8,16")
    parser.add_argument("--debug-epochs", type=int, default=3)
    return parser.parse_args()


def experiment_name(seed: int) -> str:
    """返回 RP-A development 字典的固定名称。"""
    return f"rpa_development_clong_seed{seed}"


def validate_args(args: argparse.Namespace) -> None:
    """正式模式只允许43/44和CUDA；debug强制进入独立目录。"""
    if args.seed not in FORMAL_SEEDS:
        raise ValueError("RP-A development训练只允许seed43/44")
    if not args.debug and args.device != "cuda":
        raise ValueError("正式RP-A development训练必须显式使用CUDA")
    if args.debug and args.debug_epochs < 1:
        raise ValueError("debug_epochs必须为正整数")
    if args.debug and args.debug_patients_per_class < 2:
        raise ValueError("debug每类每split至少需要2位患者，保证病灶分层可计算")


def frozen_gamma() -> tuple[float, dict]:
    """读取并核验 seed42 冻结 gamma JSON，不接受命令行覆盖。"""
    if file_sha256(GAMMA_JSON) != GAMMA_JSON_SHA256:
        raise RuntimeError("seed42 gamma_pool校准JSON SHA不一致")
    payload = json.loads(GAMMA_JSON.read_text(encoding="utf-8"))
    gamma = float(payload["gamma_pool"])
    expected = 0.25 * float(payload["median_joint_patch"]) / float(
        payload["median_joint_pool"]
    )
    checks = {
        "seed": int(payload.get("seed", -1)) == 42,
        "k_list": tuple(payload.get("k_list", ())) == K_LIST,
        "batch_count": int(payload.get("batch_count", -1)) == 74,
        "manifest": payload.get("manifest_sha256") == MANIFEST_SHA256,
        "student": payload.get("student_checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "formula": math.isclose(gamma, expected, rel_tol=0.0, abs_tol=1e-12),
    }
    if not all(checks.values()):
        raise RuntimeError(f"冻结gamma_pool血缘失败: {[k for k, v in checks.items() if not v]}")
    return gamma, payload


def debug_subset(
    arrays: tuple[dict, dict, dict, dict], patients_per_class: int, seed: int,
) -> tuple[dict, dict, dict, dict]:
    """从正式缓存按split/label抽取完整患者，供隔离冒烟使用。"""
    spatial, pooled, attention, metadata = arrays
    result = ({}, {}, {}, {})
    for split in ("train", "val"):
        frame = metadata[split]
        chosen: list[str] = []
        patient_labels = frame[["patient_id", "label"]].drop_duplicates()
        for _, group in patient_labels.groupby("label", sort=True):
            chosen.extend(
                group.sample(min(patients_per_class, len(group)), random_state=seed)
                .patient_id.astype(str).tolist()
            )
        mask = frame.patient_id.astype(str).isin(chosen).to_numpy()
        indices = np.flatnonzero(mask)
        result[0][split] = np.asarray(spatial[split][indices])
        result[1][split] = np.asarray(pooled[split][indices])
        result[2][split] = np.asarray(attention[split][indices])
        result[3][split] = frame.iloc[indices].reset_index(drop=True)
    return result


def training_args(args: argparse.Namespace, output_root: Path) -> argparse.Namespace:
    """构造已验证 S2c 训练函数需要的最小参数对象。"""
    return argparse.Namespace(
        seed=args.seed,
        experiment=experiment_name(args.seed),
        debug=args.debug,
        debug_hidden_dim=args.debug_hidden_dim,
        debug_ks=args.debug_ks,
        learning_rate=FORMAL_BUDGET["learning_rate"],
        epochs=args.debug_epochs if args.debug else FORMAL_BUDGET["epochs"],
        patience=min(2, args.debug_epochs) if args.debug else FORMAL_BUDGET["patience"],
        warmup_fraction=FORMAL_BUDGET["warmup_fraction"],
        image_batch_size=FORMAL_BUDGET["image_batch_size"],
        num_workers=args.num_workers,
        device=args.device,
        output_root=output_root,
    )


def output_directory(args: argparse.Namespace) -> Path:
    """返回正式或隔离 debug 目录，存在即拒绝覆盖。"""
    root = (
        OUTPUT_ROOT / "debug" / f"p{args.debug_patients_per_class}_e{args.debug_epochs}"
        if args.debug else OUTPUT_ROOT
    )
    output = root / experiment_name(args.seed)
    if output.exists():
        raise FileExistsError(f"输出已存在，禁止覆盖: {output}")
    output.mkdir(parents=True)
    return output


def main() -> None:
    """执行血缘核验、独立初始化、训练、健康评价和配置落盘。"""
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    lineage = verify_s0_lineage()
    full_args = argparse.Namespace(debug=False, debug_patients_per_class=0, seed=args.seed)
    frame = load_manifest_frame(full_args)
    arrays = load_cache(frame)
    if args.debug:
        arrays = debug_subset(arrays, args.debug_patients_per_class, args.seed)
    spatial, pooled, attention, metadata = arrays
    model = load_clong_model(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_args = training_args(args, OUTPUT_ROOT / "debug" if args.debug else OUTPUT_ROOT)
    sae = initialize_matryoshka(spatial["train"], metadata["train"], train_args, device)
    init_sha = initialization_sha(sae)
    gamma_pool, gamma_record = frozen_gamma()
    output = output_directory(args)
    sae, checkpoint = train_matryoshka(
        sae, spatial, pooled, attention, metadata, model, train_args, device,
        output, gamma_pool,
    )
    evaluation = evaluate_all_k(
        sae, spatial, pooled, attention, metadata, model, train_args, device,
        replication_target_k=None,
    )
    config = {
        "stage": "RP-A development SAE training",
        "status": "rpa_development_seed_trained",
        "seed": args.seed,
        "role": "development_calibration",
        "debug": bool(args.debug),
        "formal_representation_k": REPRESENTATION_K,
        "runtime_representation_k": int(max(sae.k_list)),
        "initialization_sha256": init_sha,
        "checkpoint_sha256": file_sha256(output / "sae_best.pth"),
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "gamma_pool": gamma_pool,
        "gamma_calibration_json_sha256": GAMMA_JSON_SHA256,
        "gamma_seed42_initialization_sha256": gamma_record["s2c_initialization_sha256"],
        "training": {
            "learning_rate": train_args.learning_rate,
            "epochs": train_args.epochs,
            "patience": train_args.patience,
            "warmup_fraction": train_args.warmup_fraction,
            "image_batch_size": train_args.image_batch_size,
            "best_epoch": int(checkpoint["epoch"]),
        },
        "health_evaluation": evaluation,
        "selection_semantics": "K1024 fixed for RP-A; S2c selected_k is diagnostic only",
        "lineage": lineage,
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        "git": git_snapshot(),
    }
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"RP-A seed{args.seed}训练完成: {output}")


if __name__ == "__main__":
    main()
