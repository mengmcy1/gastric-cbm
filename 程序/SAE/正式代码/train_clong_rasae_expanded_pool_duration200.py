#!/usr/bin/env python3
"""确定性重放扩大池RA-SAE前100轮，保留AdamW状态续训至固定第200轮。

旧第100轮checkpoint未保存优化器状态，因此本脚本从同一确定性初始状态
重放1--100轮，在预先固定容差内核对旧模型参数和逐轮损失，再以同一AdamW
状态续训101--200轮。这只检验当前优化设置下从1300到2600次更新的效果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE  # noqa: E402
from clong_s2c_matryoshka import joint_layer_losses  # noqa: E402
from clong_sae_discovery import patient_class_weights  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402
from train_clong_rasae_expanded_pool import (  # noqa: E402
    BATCH_SIZE,
    BASE,
    FORMAL_DRAWS_PER_EPOCH,
    REFERENCE,
    SEED,
    expanded_train_pool,
    sampling_orders,
    verify_initialization,
)


OLD_ROOT = BASE / "expanded_pool_fidelity_20260916"
ATOL = 1e-6
RTOL = 1e-5
FORMAL_EPOCHS = 200
REPLAY_EPOCH = 100


def order_sha256(order: np.ndarray) -> str:
    """计算单轮抽样索引顺序的稳定SHA256。

    Args:
        order (np.ndarray): 一维整数池内索引，包留重复和顺序。

    Returns:
        str: 小端``int64``字节的SHA256。
    """
    values = np.asarray(order, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("抽样顺序必须为一维")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_orders(
    pool: pd.DataFrame, weights: np.ndarray, epochs: int, draws: int,
) -> tuple[list[np.ndarray], pd.DataFrame, dict, pd.DataFrame]:
    """生成并核对完整抽样轨迹。

    Args:
        pool (pd.DataFrame): 固定且有序的训练池。
        weights (np.ndarray): 与池同序的抽样权重。
        epochs (int): 生成轮数。
        draws (int): 每轮有放回抽样图数。

    Returns:
        tuple: 顺序列表、覆盖表、摘要和含逐轮SHA的轨迹表。
    """
    orders, coverage, summary = sampling_orders(pool, weights, epochs, draws)
    trajectory = coverage.copy()
    trajectory["order_sha256"] = [order_sha256(order) for order in orders]
    return orders, coverage, summary, trajectory


def compare_model_states(
    replayed: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
) -> dict:
    """按预先固定容差比较重放与旧第100轮模型状态。

    Args:
        replayed (dict[str, torch.Tensor]): 重放第100轮state_dict。
        reference (dict[str, torch.Tensor]): 旧正式第100轮state_dict。

    Returns:
        dict: 固定容差、全局最大绝对/相对偏差及通过状态。
    """
    if replayed.keys() != reference.keys():
        raise RuntimeError("重放与旧checkpoint的state_dict键不一致")
    max_abs = max_rel = 0.0
    failures = []
    for key in replayed:
        current = replayed[key].detach().cpu()
        expected = reference[key].detach().cpu()
        if current.shape != expected.shape or current.dtype != expected.dtype:
            raise RuntimeError(f"{key}的shape或dtype不一致")
        if current.is_floating_point():
            difference = (current - expected).abs()
            max_abs = max(max_abs, float(difference.max()))
            denominator = expected.abs().clamp_min(1e-12)
            max_rel = max(max_rel, float((difference / denominator).max()))
            if not torch.allclose(current, expected, atol=ATOL, rtol=RTOL):
                failures.append(key)
        elif not torch.equal(current, expected):
            failures.append(key)
    return {
        "atol": ATOL, "rtol": RTOL, "max_abs_difference": max_abs,
        "max_relative_difference": max_rel, "failed_tensors": failures,
        "passed": not failures,
    }


def compare_histories(replayed: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """按固定容差比较前100轮逐轮train loss。

    Args:
        replayed (pd.DataFrame): 重放历史，至少含前100轮。
        reference (pd.DataFrame): 旧正式100轮历史。

    Returns:
        dict: 最大绝对/相对差和固定容差判定。
    """
    old = reference.sort_values("epoch").reset_index(drop=True)
    new = replayed[replayed.epoch.le(REPLAY_EPOCH)].sort_values("epoch").reset_index(drop=True)
    if len(old) != REPLAY_EPOCH or not np.array_equal(old.epoch, new.epoch):
        raise RuntimeError("重放与旧历史的epoch不一致")
    difference = np.abs(new.train_loss.to_numpy() - old.train_loss.to_numpy())
    denominator = np.maximum(np.abs(old.train_loss.to_numpy()), 1e-12)
    return {
        "atol": ATOL, "rtol": RTOL,
        "max_abs_difference": float(difference.max()),
        "max_relative_difference": float((difference / denominator).max()),
        "passed": bool(np.allclose(new.train_loss, old.train_loss, atol=ATOL, rtol=RTOL)),
    }


def optimizer_step_range(optimizer: torch.optim.Optimizer) -> dict:
    """返回AdamW参数状态中的最小和最大step。

    Args:
        optimizer (torch.optim.Optimizer): 已至少执行一次更新的AdamW。

    Returns:
        dict: 带状态参数数量与step范围。
    """
    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
    if not steps:
        raise RuntimeError("AdamW没有可保存的step状态")
    return {"state_parameters": len(steps), "min_step": min(steps), "max_step": max(steps)}


def save_full_checkpoint(
    path: Path, model: ArchetypalMatryoshkaSAE, optimizer: torch.optim.Optimizer,
    config: dict, epoch: int, update_step: int, replay_verification: dict,
) -> None:
    """保存模型、AdamW、epoch和全局更新数。

    Args:
        path (Path): 未存在的checkpoint文件。
        model (ArchetypalMatryoshkaSAE): 当前RA-SAE。
        optimizer (torch.optim.Optimizer): 与模型连续的AdamW。
        config (dict): 冻结实验配置。
        epoch (int): 当前固定轮数。
        update_step (int): 累计optimizer.step次数。
        replay_verification (dict): 第100轮重放核对结果。

    Returns:
        None: 原子性要求由目录独占保证，直接写入文件。
    """
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "config": config, "epoch": epoch, "update_step": update_step,
        "optimizer_step_range": optimizer_step_range(optimizer),
        "replay_verification": replay_verification,
    }, path)


def main() -> None:
    """运行debug链路验收或正式1--200轮确定性重放续训。

    Args:
        None: 输出目录、设备和debug由CLI指定。

    Returns:
        None: 写入抽样轨迹、历史、第100/200轮完整checkpoint和验证结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    epochs = 2 if args.debug else FORMAL_EPOCHS
    draws = 20 if args.debug else FORMAL_DRAWS_PER_EPOCH
    old_config = json.loads((REFERENCE / "config.json").read_text())
    config = {
        "scope": "expanded_pool_fixed_duration_200_with_deterministic_optimizer_replay",
        "seed": SEED, "epochs": epochs, "replay_epoch": REPLAY_EPOCH,
        "draws_per_epoch": draws, "batch": BATCH_SIZE,
        "updates_per_epoch": int(np.ceil(draws / BATCH_SIZE)),
        "hidden_dim": 2560, "k_list": [64, 128, 256], "delta": 0.2,
        "learning_rate": 1e-4, "weight_decay": 0.0,
        "warmup_epochs": 2, "warmup_restarted_after_epoch100": False,
        "gamma_pool": old_config["gamma_pool"], "margin_std": old_config["margin_std"],
        "initialization": "unique", "encoder_step_policy": "base_lr_times_original_pilot_scale",
        "representative_points": "fixed from current duration100 RA checkpoint",
        "feature_center": "fixed from current duration100 RA checkpoint",
        "checkpoint_rule": "fixed epoch100 replay and fixed epoch200; no val selection",
        "replay_atol": ATOL, "replay_rtol": RTOL,
        "old_optimizer_state_available": False,
        "optimizer_recovery": "deterministic replay epochs1-100 then continuous same-optimizer epochs101-200",
        "debug": args.debug, "new_training": True,
        "test_read": False, "external_read": False,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    device = torch.device(args.device)
    saved_pool = pd.read_csv(OLD_ROOT / "expanded_train_pool.csv")
    reconstructed_pool = expanded_train_pool()
    columns = ["source_row", "patient_id", "relative_path", "label"]
    if not saved_pool[columns].equals(reconstructed_pool[columns]):
        raise RuntimeError("重建扩大池的行顺序与旧正式输出不一致")
    pool = saved_pool.copy()
    if args.debug:
        patients = pool[["patient_id", "label"]].drop_duplicates().groupby(
            "label", group_keys=False
        ).head(4)
        pool = pool[pool.patient_id.isin(patients.patient_id)].reset_index(drop=True)
    weights = patient_class_weights(pool).astype(np.float64)
    weights /= weights.sum()
    orders, coverage, sampling_summary, trajectory = build_orders(
        pool, weights, epochs, draws
    )
    if not args.debug:
        orders100, coverage100, _summary100, trajectory100 = build_orders(
            pool, weights, REPLAY_EPOCH, draws
        )
        prefix_exact = all(
            np.array_equal(left, right) for left, right in zip(orders100, orders[:REPLAY_EPOCH])
        )
        if not prefix_exact or not trajectory100.order_sha256.equals(
            trajectory.iloc[:REPLAY_EPOCH].order_sha256.reset_index(drop=True)
        ):
            raise RuntimeError("200轮抽样序列的前100轮与独立生成100轮不逐项一致")
        old_coverage = pd.read_csv(OLD_ROOT / "sampling_coverage_by_epoch.csv")
        if not coverage100[old_coverage.columns].equals(old_coverage):
            raise RuntimeError("重放前100轮的旧覆盖统计不一致")
    else:
        prefix_exact = True
    coverage.to_csv(args.output / "sampling_coverage_by_epoch.csv", index=False)
    trajectory.to_csv(args.output / "sampling_trajectory.csv", index=False)

    pilot_frame = pd.read_csv(REFERENCE / "train_subset.csv")
    pilot_data = read_subset(pilot_frame, "train", device)
    train_data = read_subset(pool, "train", device)
    current_sae, head, classifier_weight, _ = load_models(device)
    current_checkpoint = torch.load(
        REFERENCE / "ra_final.pth", map_location=device, weights_only=False
    )
    state = current_checkpoint["state_dict"]
    model = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], 2560, (64, 128, 256), 0.2,
        True, SEED, "unique",
    ).to(device)
    initialization = verify_initialization(model, pilot_data, pilot_frame, BATCH_SIZE)
    del current_sae, pilot_data
    encoder_lr_scale = initialization["recomputed"]["scale"]
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr_scale": encoder_lr_scale},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")], "lr_scale": 1.0},
    ], lr=config["learning_rate"], weight_decay=0)
    margin = classifier_weight[1] - classifier_weight[0]
    history = []
    update_step = 0
    replay_verification = {"debug": True} if args.debug else None
    old_checkpoint = None if args.debug else torch.load(
        OLD_ROOT / "coverage_ra_final.pth", map_location=device, weights_only=False
    )
    old_history = None if args.debug else pd.read_csv(OLD_ROOT / "history.csv")

    for epoch, order in enumerate(orders, 1):
        model.train()
        warmup = min(1.0, epoch / config["warmup_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * warmup * group["lr_scale"]
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[start:start + BATCH_SIZE], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_k, terms = joint_layer_losses(
                model, train_data["spatial"][indices], train_data["pooled"][indices],
                train_data["attention"][indices], SimpleNamespace(attention_head=head),
                margin, config["margin_std"], config["gamma_pool"],
            )
            loss = torch.stack(list(loss_k.values())).mean()
            patch = torch.stack([term["patch"] for term in terms.values()]).mean()
            loss = patch + warmup * (loss - patch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch{epoch}损失非有限")
            loss.backward()
            optimizer.step()
            update_step += 1
            model.normalize_decoder()
            total_loss += float(loss.detach()) * len(indices)
        row = {
            "epoch": epoch, "train_loss": total_loss / len(order),
            "draws": len(order), "optimizer_steps_this_epoch": int(np.ceil(len(order) / BATCH_SIZE)),
            "cumulative_update_step": update_step,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "dictionary_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(args.output / "history.csv", index=False)
        print(json.dumps(row), flush=True)

        if not args.debug and epoch == REPLAY_EPOCH:
            state_comparison = compare_model_states(model.state_dict(), old_checkpoint["state_dict"])
            history_comparison = compare_histories(history_frame, old_history)
            replay_verification = {
                "sampling_prefix_exact": prefix_exact,
                "model_state": state_comparison,
                "history": history_comparison,
                "update_step": update_step,
                "optimizer_step_range": optimizer_step_range(optimizer),
                "old_optimizer_state_was_not_available": True,
                "optimizer_moments_directly_compared_to_old_run": False,
            }
            (args.output / "epoch100_replay_verification.json").write_text(
                json.dumps(replay_verification, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not state_comparison["passed"] or not history_comparison["passed"]:
                failure = {
                    "status": "failed_to_reproduce_original_training_trajectory_within_frozen_tolerance",
                    "replay_verification": replay_verification,
                    "continuation_started": False,
                }
                (args.output / "replay_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise RuntimeError("未能按约定复现原训练轨迹；停止于第100轮")
            save_full_checkpoint(
                args.output / "replay_epoch100_full.pth", model, optimizer,
                config, epoch, update_step, replay_verification,
            )

    if args.debug:
        save_full_checkpoint(
            args.output / "debug_final_full.pth", model, optimizer,
            config, epochs, update_step, replay_verification,
        )
        final_name = "debug_final_full.pth"
    else:
        if update_step != 2600:
            raise RuntimeError(f"第200轮累计更新数不是2600: {update_step}")
        save_full_checkpoint(
            args.output / "duration200_full.pth", model, optimizer,
            config, FORMAL_EPOCHS, update_step, replay_verification,
        )
        final_name = "duration200_full.pth"
    verification = {
        "status": "debug_completed" if args.debug else "duration200_completed",
        "sampling_prefix_exact": prefix_exact,
        "epochs": epochs, "final_update_step": update_step,
        "epoch101_warmup_restarted": False,
        "final_checkpoint": final_name,
        "replay_verification": replay_verification,
        "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "sampling_summary.json").write_text(
        json.dumps(sampling_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
