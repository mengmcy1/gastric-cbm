#!/usr/bin/env python3
"""以扩大train池重新拟合RA-SAE代表点，按冻结协议训练到固定第200轮。

主动改变的输入因素只有MiniBatchKMeans的代表点拟合数据：从原pilot 196图
改为扩大池1673图。旧feature center、pilot196校准队列与算法、2600次更新、
抽样轨迹、K列表和损失保持不变。新代表点造成的初始字典、校准系数和
encoder实际学习率变化属于整套固定流程的连带变化，不解释为纯代表点几何效应。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.nn import functional as torch_functional


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE, calibrate_initial_encoder  # noqa: E402
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
)
from train_clong_rasae_expanded_pool_duration200 import (  # noqa: E402
    FORMAL_EPOCHS,
    optimizer_step_range,
    order_sha256,
)


CONTROL_ROOT = BASE / "expanded_pool_duration200_20260916"
FORMAL_HIDDEN = 2560
FORMAL_K_LIST = (64, 128, 256)


def fit_representative_points(
    spatial: np.ndarray, image_weights: np.ndarray, center: np.ndarray,
    clusters: int, debug: bool,
) -> tuple[np.ndarray, dict]:
    """在旧中心坐标系中从扩大池patch拟合代表点。

    Args:
        spatial (np.ndarray): ``[N,49,1280]``扩大池特征。
        image_weights (np.ndarray): ``[N]``患者/类别平衡权重，和为1。
        center (np.ndarray): ``[1280]``原pilot固定feature center。
        clusters (int): 代表点数量。
        debug (bool): 是否为缩小链路验收。

    Returns:
        tuple[np.ndarray, dict]: ``[clusters,1280]``中心化代表点及拟合记录。
    """
    if spatial.ndim != 3 or spatial.shape[1:] != (49, 1280):
        raise ValueError("spatial必须为[N,49,1280]")
    if center.shape != (1280,) or not np.isfinite(center).all():
        raise ValueError("固定center无效")
    if len(image_weights) != len(spatial) or not np.isclose(image_weights.sum(), 1.0):
        raise ValueError("拟合权重与扩大池不一致")
    values = (spatial - center).reshape(-1, 1280)
    patch_weights = np.repeat(image_weights / 49.0, 49)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters, random_state=SEED, n_init=1,
        batch_size=1024, max_iter=2 if debug else 20, reassignment_ratio=0,
    )
    kmeans.fit(values, sample_weight=patch_weights)
    points = kmeans.cluster_centers_.astype(np.float32, copy=False)
    if points.shape != (clusters, 1280) or not np.isfinite(points).all():
        raise RuntimeError("新代表点的shape或数值无效")
    record = {
        "fit_images": len(spatial), "fit_patch_vectors": int(len(spatial) * 49),
        "clusters": clusters, "random_state": SEED, "n_init": 1,
        "batch_size": 1024, "max_iter": 2 if debug else 20,
        "reassignment_ratio": 0, "actual_n_steps": int(kmeans.n_steps_),
        "old_center_held_fixed": True,
        "sample_weight": "patient/class-balanced image weight divided by 49",
    }
    return points, record


def initialization_quality(
    model: ArchetypalMatryoshkaSAE, initialization: dict,
    old_center: torch.Tensor, old_scale: float,
) -> dict:
    """验证新代表点下的初始化质量，不要求复现旧校准记录。

    Args:
        model (ArchetypalMatryoshkaSAE): 已用pilot196执行闭式校准的新模型。
        initialization (dict): ``calibrate_initial_encoder``返回记录。
        old_center (torch.Tensor): 原pilot中心。
        old_scale (float): 原pilot代表点的历史校准系数，仅作比较。

    Returns:
        dict: 中心、有限值、松弛项、近重复和初始重构验收。
    """
    center_equal = bool(torch.equal(model.decoder_bias.detach().cpu(), old_center.detach().cpu()))
    relaxation_zero = bool(model.relaxation.eq(0).all())
    finite = bool(all(torch.isfinite(value).all() for value in model.state_dict().values()))
    with torch.no_grad():
        normalized = torch_functional.normalize(model.decoder_weight, dim=1)
        similarities = normalized @ normalized.T
        similarities.fill_diagonal_(-1)
        duplicate_fraction = float((similarities.max(1).values > 0.999).float().mean())
    after = sum(value["after"] for value in initialization["per_k"].values())
    center_only = sum(value["center_only"] for value in initialization["per_k"].values())
    quality = {
        "all_state_finite": finite, "old_center_exactly_equal": center_equal,
        "relaxation_zero_at_start": relaxation_zero,
        "fraction_atoms_with_neighbor_cosine_gt_0p999": duplicate_fraction,
        "joint_after_calibration": float(after), "joint_center_only": float(center_only),
        "joint_reconstruction_better_than_center_only": bool(after < center_only),
        "new_encoder_scale": float(initialization["scale"]),
        "old_encoder_scale_for_comparison_only": float(old_scale),
        "encoder_scale_delta_new_minus_old": float(initialization["scale"] - old_scale),
        "old_initialization_equality_required": False,
    }
    if not finite or not center_equal or not relaxation_zero:
        raise RuntimeError("新代表点初始化的基本验收失败")
    if duplicate_fraction >= 0.05:
        raise RuntimeError("新初始字典至少5%方向近重复")
    if after >= center_only:
        raise RuntimeError("新初始化联合重构不优于固定中心基线")
    return quality


def save_checkpoint(
    path: Path, model: ArchetypalMatryoshkaSAE, optimizer: torch.optim.Optimizer,
    config: dict, update_step: int, fit_record: dict, initialization: dict,
    quality: dict,
) -> None:
    """保存候选的模型、优化器、更新数和初始化血缘。

    Args:
        path (Path): 未存在的checkpoint路径。
        model (ArchetypalMatryoshkaSAE): 最终模型。
        optimizer (torch.optim.Optimizer): 最终AdamW。
        config (dict): 冻结训练配置。
        update_step (int): 累计更新数。
        fit_record (dict): MiniBatchKMeans拟合记录。
        initialization (dict): pilot196闭式校准记录。
        quality (dict): 初始化质量验收。

    Returns:
        None: 写入完整checkpoint。
    """
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "config": config, "epoch": config["epochs"], "update_step": update_step,
        "optimizer_step_range": optimizer_step_range(optimizer),
        "representative_fit": fit_record, "initialization": initialization,
        "initialization_quality": quality,
    }, path)


def main() -> None:
    """拟合扩大池代表点并运行debug或正式200轮训练。

    Args:
        None: 输出、设备和debug由CLI提供。

    Returns:
        None: 写入代表点记录、校准、抽样轨迹、历史及完整checkpoint。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    epochs = 2 if args.debug else FORMAL_EPOCHS
    draws = 20 if args.debug else FORMAL_DRAWS_PER_EPOCH
    hidden = 8 if args.debug else FORMAL_HIDDEN
    k_list = (2, 4, 8) if args.debug else FORMAL_K_LIST
    old_config = json.loads((REFERENCE / "config.json").read_text())
    config = {
        "scope": "expanded_pool_refit_representative_points_fixed_duration200",
        "single_changed_input_factor": "representative_point_fit_data_pilot196_to_expanded1673",
        "effect_scope": "entire_fixed_pipeline_after_changing_fit_data_not_pure_point_geometry",
        "seed": SEED, "epochs": epochs, "draws_per_epoch": draws,
        "batch": BATCH_SIZE, "updates_per_epoch": int(np.ceil(draws / BATCH_SIZE)),
        "hidden_dim": hidden, "k_list": list(k_list), "delta": 0.2,
        "learning_rate": 1e-4, "weight_decay": 0.0, "warmup_epochs": 2,
        "gamma_pool": old_config["gamma_pool"], "margin_std": old_config["margin_std"],
        "initialization": "unique", "feature_center": "old pilot center held exactly fixed",
        "initial_encoder_calibration": "same closed-form algorithm on original pilot196",
        "encoder_step_policy": "base_lr_times_new_pilot196_calibration_scale",
        "checkpoint_rule": "fixed final epoch200; no val selection",
        "debug": args.debug, "new_training": True,
        "test_read": False, "external_read": False,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pool = expanded_train_pool()
    if args.debug:
        patients = pool[["patient_id", "label"]].drop_duplicates().groupby(
            "label", group_keys=False
        ).head(4)
        pool = pool[pool.patient_id.isin(patients.patient_id)].reset_index(drop=True)
    image_weights = patient_class_weights(pool).astype(np.float64)
    image_weights /= image_weights.sum()
    cpu_data = read_subset(pool, "train", torch.device("cpu"))
    old_checkpoint_cpu = torch.load(
        REFERENCE / "ra_final.pth", map_location="cpu", weights_only=False
    )
    old_state = old_checkpoint_cpu["state_dict"]
    old_center_np = old_state["decoder_bias"].cpu().numpy().astype(np.float32, copy=True)
    points_np, fit_record = fit_representative_points(
        cpu_data["spatial"].numpy(), image_weights, old_center_np, hidden, args.debug
    )
    (args.output / "representative_fit.json").write_text(
        json.dumps(fit_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del cpu_data

    device = torch.device(args.device)
    points = torch.from_numpy(points_np).to(device)
    old_center = old_state["decoder_bias"].to(device)
    model = ArchetypalMatryoshkaSAE(
        points, old_center, hidden, k_list, 0.2, True, SEED, "unique"
    ).to(device)
    pilot_frame = pd.read_csv(REFERENCE / "train_subset.csv")
    pilot_data = read_subset(pilot_frame, "train", device)
    pilot_weights = patient_class_weights(pilot_frame).astype(np.float64)
    pilot_weights /= pilot_weights.sum()
    initialization = calibrate_initial_encoder(
        model, pilot_data["spatial"], pilot_data["attention"],
        torch.as_tensor(pilot_weights, device=device), BATCH_SIZE,
    )
    old_initialization = json.loads((REFERENCE / "ra_initialization.json").read_text())
    quality = initialization_quality(
        model, initialization, old_center, float(old_initialization["scale"])
    )
    initialization_record = {"calibration": initialization, "quality": quality}
    (args.output / "initialization.json").write_text(
        json.dumps(initialization_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del pilot_data

    train_data = read_subset(pool, "train", device)
    unused, head, classifier_weight, _ = load_models(device)
    del unused
    orders, coverage, sampling_summary = sampling_orders(pool, image_weights, epochs, draws)
    trajectory = coverage.copy()
    trajectory["order_sha256"] = [order_sha256(order) for order in orders]
    if not args.debug:
        control_trajectory = pd.read_csv(CONTROL_ROOT / "sampling_trajectory.csv")
        if not trajectory.order_sha256.equals(control_trajectory.order_sha256):
            raise RuntimeError("候选的200轮抽样顺序与旧代表点对照不一致")
        if not coverage[control_trajectory.columns.intersection(coverage.columns)].equals(
            control_trajectory[control_trajectory.columns.intersection(coverage.columns)]
        ):
            raise RuntimeError("候选的抽样覆盖轨迹与对照不一致")
    trajectory.to_csv(args.output / "sampling_trajectory.csv", index=False)
    margin = classifier_weight[1] - classifier_weight[0]
    encoder_lr_scale = float(initialization["scale"])
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr_scale": encoder_lr_scale},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")], "lr_scale": 1.0},
    ], lr=config["learning_rate"], weight_decay=0)
    history = []
    update_step = 0
    for epoch, order in enumerate(orders, 1):
        model.train()
        warmup = min(1.0, epoch / config["warmup_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * warmup * group["lr_scale"]
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[start:start + BATCH_SIZE], device=device)
            optimizer.zero_grad(set_to_none=True)
            losses, terms = joint_layer_losses(
                model, train_data["spatial"][indices], train_data["pooled"][indices],
                train_data["attention"][indices], SimpleNamespace(attention_head=head),
                margin, config["margin_std"], config["gamma_pool"],
            )
            loss = torch.stack(list(losses.values())).mean()
            patch = torch.stack([value["patch"] for value in terms.values()]).mean()
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
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        print(json.dumps(row), flush=True)
    expected_steps = epochs * int(np.ceil(draws / BATCH_SIZE))
    if update_step != expected_steps:
        raise RuntimeError(f"最终更新数{update_step}不是预期{expected_steps}")
    checkpoint_name = "debug_final_full.pth" if args.debug else "refit_points_duration200_full.pth"
    save_checkpoint(
        args.output / checkpoint_name, model, optimizer, config, update_step,
        fit_record, initialization, quality,
    )
    verification = {
        "status": "debug_completed" if args.debug else "refit_points_duration200_completed",
        "fit_record": fit_record, "initialization_quality": quality,
        "sampling_trajectory_matches_control": True,
        "final_update_step": update_step,
        "optimizer_step_range": optimizer_step_range(optimizer),
        "final_checkpoint": checkpoint_name,
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
