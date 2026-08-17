#!/usr/bin/env python3
"""在冻结EfficientNet-B0的7x7末层特征图上训练共享空间Token SAE。

每个图像的49个1280维token由同一个Top-K SAE重构，再经GAP和冻结分类头评价保真度。
脚本只读取train/val；未通过未剪枝门槛时不做Feature剪枝或临床概览。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from efficientnet_sae_discovery import (
    ACTIVE_EPS,
    ActivationCapture,
    FeatureDataset,
    INPUT_DIM,
    OUTPUT_ROOT,
    TopKSparseAutoencoder,
    annotate_confusion,
    classification_metrics,
    duplicate_decoder_rate,
    file_sha256,
    git_snapshot,
    json_ready,
    load_dataframe,
    load_model,
    module_sha,
    patient_class_weights,
    product_paths,
    seed_everything,
    validate_products,
)


TOKEN_COUNT = 49


def parse_args() -> argparse.Namespace:
    """解析空间缓存、Top-K SAE训练和debug参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("global", "local"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 202, 503), required=True)
    parser.add_argument("--hidden-dim", type=int, default=10240)
    parser.add_argument("--top-k", type=int, choices=(128, 256), required=True)
    parser.add_argument("--margin-loss-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--sae-image-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-cache-from", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "SAE-B空间Token")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class SpatialFeatureDataset(Dataset):
    """返回一张图的float32空间特征[49,1280]。"""

    def __init__(self, features: np.ndarray):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.features[index], dtype=np.float32))


@torch.no_grad()
def extract_spatial_features(
    model: torch.nn.Module,
    capture: ActivationCapture,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    cache: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """确定性提取[图像,49,1280]空间缓存及冻结分类概率。"""
    features, metadata = {}, {}
    for split in ("train", "val"):
        split_frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        loader = DataLoader(
            FeatureDataset(split_frame, args.branch), batch_size=args.image_batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        feature_batches, probability_batches = [], []
        for batch_index, (images, _) in enumerate(loader, 1):
            logits = model(images.to(device, non_blocking=True))
            if capture.output is None or capture.output.shape[1:] != (INPUT_DIM, 7, 7):
                raise RuntimeError("EfficientNet捕获层形状不是[1280,7,7]")
            tokens = capture.output.permute(0, 2, 3, 1).reshape(-1, TOKEN_COUNT, INPUT_DIM)
            feature_batches.append(tokens.cpu().numpy().astype(np.float32))
            probability_batches.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            print(f"提取 {args.branch}/{split}: [{batch_index}/{len(loader)}]")
        array = np.concatenate(feature_batches)
        split_frame["cancer_probability"] = np.concatenate(probability_batches).astype(np.float32)
        np.save(cache / f"{split}_spatial_features.npy", array)
        split_frame.to_csv(cache / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
        features[split], metadata[split] = array, split_frame
        print(f"{args.branch}/{split}: {len(array)}张，空间特征={array.shape}/{array.dtype}")
    return features, metadata


def reuse_spatial_cache(
    source: Path,
    branch: str,
    seed: int,
    products: dict[str, Path],
    target: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """校验冻结模型与清单血缘后复用空间缓存。"""
    config = json.loads((source.parent / "config.json").read_text(encoding="utf-8"))
    expected = {
        "branch": branch,
        "seed": seed,
        "checkpoint_sha256": file_sha256(products["checkpoint"]),
        "manifest_sha256": file_sha256(products["manifest"]),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"空间缓存{key}与当前任务不一致")
    features, metadata = {}, {}
    for split in ("train", "val"):
        feature_path = source / f"{split}_spatial_features.npy"
        metadata_path = source / f"{split}_metadata.csv"
        features[split] = np.load(feature_path, mmap_mode="r")
        metadata[split] = pd.read_csv(
            metadata_path, encoding="utf-8-sig", dtype={"patient_id": str},
            float_precision="round_trip",
        )
        shutil.copy2(feature_path, target / feature_path.name)
        shutil.copy2(metadata_path, target / metadata_path.name)
    print(f"复用冻结空间特征缓存: {source}")
    return features, metadata


def make_train_loader(
    features: np.ndarray, metadata: pd.DataFrame, batch_size: int, seed: int,
) -> DataLoader:
    """按患者和类别平衡抽取整张图，保留其49个token共同参与margin计算。"""
    weights = patient_class_weights(metadata)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(features), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(SpatialFeatureDataset(features), batch_size=batch_size, sampler=sampler)


def spatial_center(features: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    """计算患者/类别平衡后所有空间token的1280维中心。"""
    weights = patient_class_weights(metadata).astype(np.float64)
    image_means = np.asarray(features, dtype=np.float32).mean(axis=1)
    return np.average(image_means, axis=0, weights=weights).astype(np.float32)


def batch_terms(
    sae: TopKSparseAutoencoder,
    batch: torch.Tensor,
    margin_vector: torch.Tensor,
    margin_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回每图token MSE、margin MSE、每token L0和每图Feature并集大小。"""
    batch_size = len(batch)
    input_dim = batch.shape[-1]
    reconstructed, hidden = sae(batch.reshape(-1, input_dim))
    reconstructed = reconstructed.reshape(batch_size, TOKEN_COUNT, input_dim)
    hidden = hidden.reshape(batch_size, TOKEN_COUNT, -1)
    mse = (reconstructed - batch).square().mean(dim=(1, 2))
    original_gap, reconstructed_gap = batch.mean(1), reconstructed.mean(1)
    margin = (((reconstructed_gap - original_gap) @ margin_vector) / margin_std).square()
    token_l0 = hidden.gt(ACTIVE_EPS).sum(2).float().mean(1)
    image_union_l0 = hidden.gt(ACTIVE_EPS).any(1).sum(1).float()
    return mse, margin, token_l0, image_union_l0


@torch.no_grad()
def evaluate_loss(
    sae: TopKSparseAutoencoder,
    features: np.ndarray,
    metadata: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    margin_vector: torch.Tensor,
    margin_std: float,
    gamma: float,
) -> np.ndarray:
    """按患者/类别平衡返回val总损失及分项。"""
    loader = DataLoader(SpatialFeatureDataset(features), batch_size=batch_size, shuffle=False)
    weights = patient_class_weights(metadata)
    total, start = np.zeros(5), 0
    sae.eval()
    for batch in loader:
        batch = batch.to(device)
        mse, margin, token_l0, union_l0 = batch_terms(sae, batch, margin_vector, margin_std)
        values = torch.stack((mse + gamma * margin, mse, margin, token_l0, union_l0), 1).cpu().numpy()
        current = weights[start:start + len(batch)]
        total += (values * current[:, None]).sum(0)
        start += len(batch)
    return total / weights.sum()


def train_sae(
    features: dict[str, np.ndarray],
    metadata: dict[str, pd.DataFrame],
    classifier_margin: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    model_dir: Path,
) -> tuple[TopKSparseAutoencoder, dict]:
    """整图平衡训练空间SAE，并按val总目标选择checkpoint。"""
    center = spatial_center(features["train"], metadata["train"])
    sae = TopKSparseAutoencoder(
        INPUT_DIM, args.hidden_dim, torch.from_numpy(center).to(device), args.top_k,
    ).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    loader = make_train_loader(
        features["train"], metadata["train"], args.sae_image_batch_size, args.seed,
    )
    train_gap = np.asarray(features["train"], dtype=np.float32).mean(1)
    weights = patient_class_weights(metadata["train"])
    margins = train_gap @ classifier_margin
    margin_mean = float(np.average(margins, weights=weights))
    margin_std = float(np.sqrt(np.average((margins - margin_mean) ** 2, weights=weights)))
    if margin_std <= 1e-8:
        raise RuntimeError("冻结分类margin在train上没有有效方差")
    margin_vector = torch.from_numpy(classifier_margin.astype(np.float32)).to(device)
    warmup_epochs = max(1, math.ceil(args.epochs * args.warmup_fraction))
    best_loss, stale, history = math.inf, 0, []
    best_path = model_dir / "spatial_sae_best.pth"
    for epoch in range(1, args.epochs + 1):
        scale = min(1.0, epoch / warmup_epochs)
        gamma = args.margin_loss_weight * scale
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        sae.train()
        totals, count = np.zeros(5), 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            mse, margin, token_l0, union_l0 = batch_terms(
                sae, batch, margin_vector, margin_std,
            )
            loss = mse.mean() + gamma * margin.mean()
            loss.backward()
            with torch.no_grad():
                weight, gradient = sae.decoder_weight, sae.decoder_weight.grad
                projection = (gradient * weight).sum(1, keepdim=True)
                gradient.sub_(projection / weight.square().sum(1, keepdim=True).clamp_min(1e-12) * weight)
            optimizer.step()
            sae.normalize_decoder()
            values = np.array([
                loss.item(), mse.mean().item(), margin.mean().item(),
                token_l0.mean().item(), union_l0.mean().item(),
            ])
            totals += values * len(batch)
            count += len(batch)
        train_values = totals / count
        val_values = evaluate_loss(
            sae, features["val"], metadata["val"], args.sae_image_batch_size,
            device, margin_vector, margin_std, args.margin_loss_weight,
        )
        row = {
            "epoch": epoch, "warmup_scale": scale, "train_total": train_values[0],
            "train_token_mse": train_values[1], "train_margin_mse": train_values[2],
            "train_token_l0": train_values[3], "train_image_union_l0": train_values[4],
            "val_total": val_values[0], "val_token_mse": val_values[1],
            "val_margin_mse": val_values[2], "val_token_l0": val_values[3],
            "val_image_union_l0": val_values[4],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} train={train_values[0]:.5f} "
            f"val={val_values[0]:.5f} tokenMSE={val_values[1]:.5f} "
            f"margin={val_values[2]:.5f} tokenL0={val_values[3]:.1f} "
            f"unionL0={val_values[4]:.1f}"
        )
        if epoch < warmup_epochs:
            continue
        if val_values[0] < best_loss:
            best_loss, stale = float(val_values[0]), 0
            torch.save({
                "sae_state_dict": sae.state_dict(), "input_dim": INPUT_DIM,
                "hidden_dim": args.hidden_dim, "top_k": args.top_k,
                "margin_loss_weight": args.margin_loss_weight,
                "margin_train_std": margin_std, "epoch": epoch,
                "val_total_loss": best_loss,
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: {args.patience}轮无val改善")
                break
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    sae.load_state_dict(checkpoint["sae_state_dict"], strict=True)
    return sae, checkpoint


@torch.no_grad()
def project_metrics(
    sae: TopKSparseAutoencoder,
    features: np.ndarray,
    metadata: pd.DataFrame,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    patient_threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict:
    """流式重构空间图，返回GAP分类保真与空间稀疏/余弦统计。"""
    loader = DataLoader(SpatialFeatureDataset(features), batch_size=batch_size, shuffle=False)
    original_gaps, reconstructed_gaps = [], []
    cosine_sum, token_count, token_l0_sum, union_l0_sum = 0.0, 0, 0.0, 0.0
    active_features = np.zeros(sae.encoder.out_features, dtype=bool)
    sae.eval()
    for batch in loader:
        batch = batch.to(device)
        flat = batch.reshape(-1, INPUT_DIM)
        reconstructed, hidden = sae(flat)
        reconstructed = reconstructed.reshape(len(batch), TOKEN_COUNT, INPUT_DIM)
        hidden = hidden.reshape(len(batch), TOKEN_COUNT, -1)
        cosine = torch.nn.functional.cosine_similarity(flat, reconstructed.reshape(-1, INPUT_DIM), dim=1)
        cosine_sum += cosine.sum().item()
        token_count += len(cosine)
        token_l0_sum += hidden.gt(ACTIVE_EPS).sum(2).float().sum().item()
        union_l0_sum += hidden.gt(ACTIVE_EPS).any(1).sum(1).float().sum().item()
        active_features |= hidden.gt(ACTIVE_EPS).any(dim=(0, 1)).cpu().numpy()
        original_gaps.append(batch.mean(1).cpu().numpy())
        reconstructed_gaps.append(reconstructed.mean(1).cpu().numpy())
    original_gap = np.concatenate(original_gaps).astype(np.float32)
    reconstructed_gap = np.concatenate(reconstructed_gaps).astype(np.float32)
    metrics = classification_metrics(
        original_gap, reconstructed_gap, metadata, classifier_weight,
        classifier_bias, patient_threshold,
    )
    original_logits = original_gap @ classifier_weight.T + classifier_bias
    original_probability = torch.softmax(torch.from_numpy(original_logits), 1)[:, 1].numpy()
    metrics.update({
        "mean_token_cosine": float(cosine_sum / token_count),
        "mean_token_l0": float(token_l0_sum / token_count),
        "mean_image_union_l0": float(union_l0_sum / len(features)),
        "dead_feature_count": int((~active_features).sum()),
        "frozen_probability_max_abs_diff": float(np.max(np.abs(
            original_probability - metadata.cancer_probability.to_numpy(float)
        ))),
    })
    return metrics


def run_self_test() -> None:
    """验证空间reshape、逐tokenTop-K和图像级margin损失shape。"""
    center = torch.zeros(8)
    sae = TopKSparseAutoencoder(8, 16, center, top_k=3)
    batch = torch.randn(2, TOKEN_COUNT, 8)
    terms = batch_terms(sae, batch, torch.randn(8), 1.0)
    assert all(value.shape == (2,) for value in terms)
    assert float(terms[2].max()) <= 3
    print("EfficientNet spatial SAE self-test passed")


def main() -> None:
    """校验血缘、提取空间缓存、训练SAE并执行未剪枝正式门槛。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.hidden_dim != 10240 or args.margin_loss_weight != 0.1:
        raise ValueError("SAE-B正式协议固定hidden_dim=10240且margin_loss_weight=0.1")
    seed_everything(args.seed)
    products = product_paths(args.branch, args.seed)
    source_config = validate_products(args.branch, args.seed, products)
    output = args.output_root / args.experiment
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    cache, model_dir = output / "空间特征缓存", output / "SAE模型"
    cache.mkdir(parents=True)
    model_dir.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_dataframe(args, products)
    model = load_model(args.branch, products, device)
    frozen_sha = module_sha(model)
    capture = ActivationCapture(model.features[8])
    if args.feature_cache_from:
        features, metadata = reuse_spatial_cache(
            args.feature_cache_from.resolve(), args.branch, args.seed, products, cache,
        )
    else:
        features, metadata = extract_spatial_features(
            model, capture, frame, args, device, cache,
        )
    _, patient_threshold = annotate_confusion(metadata)
    for split in ("train", "val"):
        metadata[split].to_csv(cache / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
    classifier = model.classifier[1]
    weight = classifier.weight.detach().cpu().numpy()
    bias = classifier.bias.detach().cpu().numpy()
    classifier_margin = weight[1] - weight[0]
    sae, best = train_sae(features, metadata, classifier_margin, args, device, model_dir)
    metrics = {"best_epoch": int(best["epoch"]), "splits": {}}
    for split in ("train", "val"):
        metrics["splits"][split] = project_metrics(
            sae, features[split], metadata[split], weight, bias, patient_threshold,
            args.sae_image_batch_size, device,
        )
    val = metrics["splits"]["val"]
    duplicate_rate = duplicate_decoder_rate(sae.decoder_weight.detach().cpu().numpy())
    dead_rate = metrics["splits"]["train"]["dead_feature_count"] / args.hidden_dim
    passed = bool(
        val["patient_original_auc"] - val["patient_reconstructed_auc"] <= 0.01
        and val["patient_prediction_agreement_at_locked_threshold"] >= 0.95
        and val["mean_cosine"] >= 0.90
        and val["recovered_cross_entropy"] >= 0.95
        and val["mean_token_cosine"] >= 0.80
        and val["frozen_probability_max_abs_diff"] <= 1e-5
        and dead_rate <= 0.10 and duplicate_rate <= 0.10
    )
    metrics["quality_gate"] = {
        "maximum_patient_auc_drop": 0.01,
        "minimum_patient_prediction_agreement": 0.95,
        "minimum_gap_cosine": 0.90,
        "minimum_recovered_cross_entropy": 0.95,
        "minimum_token_cosine": 0.80,
        "maximum_frozen_probability_abs_diff": 1e-5,
        "maximum_dead_feature_rate": 0.10,
        "maximum_duplicate_decoder_rate": 0.10,
        "dead_feature_rate_train": dead_rate,
        "duplicate_decoder_rate": duplicate_rate,
        "passed": passed,
    }
    (output / "metrics.json").write_text(
        json.dumps(json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output": str(output.resolve()), "input_shape": [TOKEN_COUNT, INPUT_DIM],
        "cache_dtype": "float32", "checkpoint_sha256": file_sha256(products["checkpoint"]),
        "manifest_sha256": file_sha256(products["manifest"]),
        "source_stage": source_config.get("stage", source_config.get("model")),
        "counts": {
            split: {"images": len(part), "patients": part.patient_id.nunique()}
            for split, part in metadata.items()
        },
        "feature_layer": "features[8] output, 49 spatial tokens without GAP",
        "test_evaluated": False, "external_evaluated": False, **git_snapshot(),
    }
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    capture.close()
    if module_sha(model) != frozen_sha:
        raise RuntimeError("冻结EfficientNet参数或BN统计发生变化")
    print(f"最佳epoch={best['epoch']}；quality_gate={passed}；输出={output}")


if __name__ == "__main__":
    main()
