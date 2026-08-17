#!/usr/bin/env python3
"""Train the MG1 grayscale local teacher or its patch-shuffle control.

Both modes use the frozen independent-axis dynamic ROI v3 manifest. Rectangular
crops are resized directly to 224x224, converted to luma and copied to three
channels. ``real`` preserves spatial content; ``patch_shuffle`` permutes a 7x7
grid while preserving the crop's pixel values and local gray statistics.

Only train and validation rows are read. Internal test and external data are
outside this stage. Model selection uses validation patient AUC, then image AUC,
then validation loss. The geometry-only result is linked from the v3 audit but
does not participate in checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import functional as TF

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256


DEFAULT_MANIFEST = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817/"
    "independent_axis_dynamic/"
    "mage_teacher_roi_manifest_independent_axis_dynamic_v3.csv"
)
DEFAULT_V3_AUDIT = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817/"
    "mg0b_independent_axis_dynamic_roi_v3_audit.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选"
IMAGE_SIZE = 224
PATCH_GRID = 7
LABEL_SMOOTHING = 0.10
GEOMETRY_PATIENT_AUC = 0.6091562910834429
EXPECTED_COUNTS = {
    ("train", 0): 1169,
    ("train", 1): 1181,
    ("val", 0): 257,
    ("val", 1): 240,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    """Parse one paired MG1 run and its frozen optimization parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("real", "patch_shuffle"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v3-audit", type=Path, default=DEFAULT_V3_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage-a-epochs", type=int, default=5)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--stage-a-lr", type=float, default=1e-3)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed Python-independent numerical libraries and deterministic CUDA paths."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from PyTorch's reproducible worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def stable_uniform(*parts: object) -> float:
    """Map stable sample/epoch identifiers to a deterministic unit value."""
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def stable_patch_permutation(sha256: str, epoch: int, seed: int) -> torch.Tensor:
    """Return a deterministic permutation for one image at one epoch."""
    digest = hashlib.sha256(f"{sha256}|{epoch}|{seed}|patch".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
    return torch.randperm(PATCH_GRID * PATCH_GRID, generator=generator)


def patch_shuffle(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Permute a 7x7 grid of 32px patches without changing any pixel values."""
    channels, height, width = tensor.shape
    if height != IMAGE_SIZE or width != IMAGE_SIZE:
        raise ValueError(f"patch-shuffle要求{IMAGE_SIZE}x{IMAGE_SIZE}，实际{height}x{width}")
    patch = IMAGE_SIZE // PATCH_GRID
    tiles = (
        tensor.reshape(channels, PATCH_GRID, patch, PATCH_GRID, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(PATCH_GRID * PATCH_GRID, channels, patch, patch)
    )
    return (
        tiles[permutation]
        .reshape(PATCH_GRID, PATCH_GRID, channels, patch, patch)
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, IMAGE_SIZE, IMAGE_SIZE)
    )


def load_manifest(
    manifest_path: Path,
    audit_path: Path,
    debug: bool,
    debug_patients_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Load and verify the frozen v3 train/val queue and geometry audit lineage."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    product = audit["independent_axis_dynamic"]
    if file_sha256(manifest_path) != product["manifest_sha256"]:
        raise ValueError("MG1 manifest SHA与v3审计不一致")
    recorded_geometry = float(
        product["geometry_audit"]["center_plus_size_shape"]["val_patient_auc"]
    )
    if not math.isclose(recorded_geometry, GEOMETRY_PATIENT_AUC, abs_tol=1e-12):
        raise ValueError("MG1 geometry-only患者AUC与预注册值不一致")

    frame = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        dtype={"patient_id": str},
        float_precision="round_trip",
    )
    required = {
        "image_relpath", "relative_path", "patient_id", "label", "split", "sha256",
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2",
        "source_box_x1", "source_box_y1", "source_box_x2", "source_box_y2",
        "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
        "roi_geometry_variant", "internal_test_read", "external_read",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MG1 v3清单缺少字段: {sorted(missing)}")
    if frame.groupby(["split", "label"]).size().to_dict() != EXPECTED_COUNTS:
        raise ValueError("MG1 v3清单规模与预注册不一致")
    if not frame.roi_geometry_variant.eq("independent_axis_dynamic_margin_v3").all():
        raise ValueError("MG1禁止混入非v3主分支")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("MG1患者跨split")
    if frame[["internal_test_read", "external_read"]].astype(bool).any().any():
        raise ValueError("MG1清单混入锁定数据")
    missing_images = [
        path for path in frame.image_relpath.astype(str)
        if not (PROJECT_ROOT / path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"MG1原图缺失，首例: {missing_images[0]}")

    if debug:
        selected = []
        for split, split_frame in frame.groupby("split", sort=False):
            patients = split_frame[["patient_id", "label"]].drop_duplicates()
            sampled = pd.concat([
                group.sample(min(debug_patients_per_class, len(group)), random_state=seed)
                for _, group in patients.groupby("label", sort=True)
            ])
            selected.append(split_frame.loc[split_frame.patient_id.isin(sampled.patient_id)])
        frame = pd.concat(selected, ignore_index=True)
    return frame.reset_index(drop=True), audit


def patient_class_balanced_weights(frame: pd.DataFrame) -> torch.Tensor:
    """Balance labels, then patients within each label, then images per patient."""
    patient_sizes = frame.groupby("patient_id").size()
    patient_labels = frame[["patient_id", "label"]].drop_duplicates()
    class_patient_counts = patient_labels.label.value_counts()
    return torch.tensor([
        1.0 / (
            float(class_patient_counts.loc[row.label])
            * float(patient_sizes.loc[row.patient_id])
        )
        for row in frame.itertuples(index=False)
    ], dtype=torch.double)


class MG1Dataset(Dataset):
    """Return one v3 luma ROI [3,224,224], label and static coordinate metadata."""

    def __init__(self, frame: pd.DataFrame, mode: str, training: bool, seed: int):
        self.frame = frame.reset_index(drop=True)
        self.mode = mode
        self.training = training
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic flip and patch permutations."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        crop = np.array([
            row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
        ], dtype=float)
        pixels = (crop * np.array([width, height, width, height])).round().astype(int)
        pixels[[0, 2]] = np.clip(pixels[[0, 2]], 0, width)
        pixels[[1, 3]] = np.clip(pixels[[1, 3]], 0, height)
        roi = image.crop(tuple(pixels)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        ).convert("L").convert("RGB")
        tensor = TF.pil_to_tensor(roi).float().div(255.0)

        epoch = self.epoch if self.training else -1
        if self.training and stable_uniform(row.sha256, epoch, self.seed, "flip") < 0.5:
            tensor = torch.flip(tensor, dims=(2,))
        if self.mode == "patch_shuffle":
            permutation = stable_patch_permutation(str(row.sha256), epoch, self.seed)
            tensor = patch_shuffle(tensor, permutation)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": tensor,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "row_index": index,
        }


def build_model(pretrained: bool) -> nn.Module:
    """Create EfficientNet-B0 with a two-class classifier."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def set_stage(model: nn.Module, stage: str) -> None:
    """Set trainable parameters and BN/dropout mode for one frozen MG1 stage."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "A":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif stage == "B":
        for block in model.features[5:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"未知MG1 stage: {stage}")


def set_train_mode(model: nn.Module, stage: str) -> None:
    """Keep frozen BN statistics fixed while enabling only the active stage."""
    model.eval()
    if stage == "B":
        for block in model.features[5:]:
            block.train()
    model.classifier.train()


def patient_mean(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate all image probabilities of each patient by arithmetic mean."""
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), probability=(column, "mean")
    )


def sensitivity_threshold_metrics(frame: pd.DataFrame, column: str) -> dict:
    """Report metrics at the highest observed threshold with sensitivity >=0.90."""
    positive = frame.loc[frame.label.eq(1), column].to_numpy(dtype=float)
    candidates = np.sort(np.unique(positive))[::-1]
    threshold = next(
        float(value) for value in candidates
        if float(np.mean(positive >= value)) >= 0.90
    )
    prediction = frame[column].ge(threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(frame.label, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "threshold": threshold,
        "accuracy": float((tp + tn) / len(frame)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(1.0 - sensitivity),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
    }


def evaluate(
    model: nn.Module, dataset: MG1Dataset, loader: DataLoader, device: torch.device
) -> tuple[pd.DataFrame, dict]:
    """Evaluate one MG1 mode and return image predictions plus image/patient metrics."""
    model.eval()
    rows = []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(F.cross_entropy(logits, labels)) * len(labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for row_index, probability in zip(batch["row_index"].tolist(), probabilities):
                row = dataset.frame.iloc[row_index]
                rows.append({
                    "row_index": row_index,
                    "relative_path": row.relative_path,
                    "patient_id": row.patient_id,
                    "label": int(row.label),
                    "split": row.split,
                    "roi_source": row.roi_source,
                    "cancer_probability": float(probability),
                })
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    patients = patient_mean(predictions, "cancer_probability")
    metrics = {
        "val_loss": loss_sum / len(dataset),
        "val_image_auc": float(roc_auc_score(predictions.label, predictions.cancer_probability)),
        "val_patient_auc": float(roc_auc_score(patients.label, patients.probability)),
        "image_threshold_metrics": sensitivity_threshold_metrics(
            predictions, "cancer_probability"
        ),
        "patient_threshold_metrics": sensitivity_threshold_metrics(patients, "probability"),
    }
    return predictions, metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage: str,
) -> float:
    """Train one epoch with label-smoothed cross entropy."""
    set_train_mode(model, stage)
    total, seen = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        seen += len(labels)
    return total / seen


def git_snapshot() -> dict:
    """Record the current commit and dirty flag without modifying the worktree."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False,
    ).stdout.strip())
    return {"git_commit": commit, "git_dirty": dirty}


def run_self_test() -> None:
    """Check patch conservation, determinism and patient-balanced weights."""
    tensor = torch.arange(3 * IMAGE_SIZE * IMAGE_SIZE, dtype=torch.float32).reshape(
        3, IMAGE_SIZE, IMAGE_SIZE
    )
    permutation = stable_patch_permutation("sample", 3, 42)
    shuffled = patch_shuffle(tensor, permutation)
    assert shuffled.shape == tensor.shape
    assert torch.equal(torch.sort(shuffled.flatten()).values, torch.sort(tensor.flatten()).values)
    assert torch.equal(permutation, stable_patch_permutation("sample", 3, 42))
    frame = pd.DataFrame({
        "patient_id": ["a", "a", "b", "c", "d", "d"],
        "label": [0, 0, 0, 1, 1, 1],
    })
    weights = patient_class_balanced_weights(frame).numpy()
    assert np.isclose(weights[:3].sum(), weights[3:].sum())
    print("MG1 self-test通过: patch像素守恒、确定性与患者类别平衡采样有效")


def main() -> None:
    """Run one formal MG1 mode from lineage checks through checkpoint export."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.seed != 42 and not args.debug:
        raise ValueError("MG1预注册只允许正式seed42")
    seed_everything(args.seed)
    frame, audit = load_manifest(
        args.manifest.resolve(), args.v3_audit.resolve(), args.debug,
        args.debug_patients_per_class, args.seed,
    )
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    val = frame.loc[frame.split.eq("val")].reset_index(drop=True)

    run_name = args.run_name or f"mg1_{args.input_mode}_efficientnet_b0_seed{args.seed}"
    if args.debug:
        run_name += "_debug"
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.num_workers = 0
    output_dir = args.output_root.resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"MG1输出已存在，拒绝覆盖: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MG1Dataset(train, args.input_mode, True, args.seed)
    val_dataset = MG1Dataset(val, args.input_mode, False, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        patient_class_balanced_weights(train), len(train), replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    model = build_model(pretrained=True).to(device)
    print(
        f"设备={device}; mode={args.input_mode}; seed={args.seed}; "
        f"train={len(train)}张/{train.patient_id.nunique()}人; "
        f"val={len(val)}张/{val.patient_id.nunique()}人; "
        f"geometry患者AUC={GEOMETRY_PATIENT_AUC:.4f}"
    )

    history = []
    best = {"key": None, "state": None, "stage": None, "epoch": None, "metrics": None}
    best_stage_a_state = None
    for stage, epochs, learning_rate in (
        ("A", args.stage_a_epochs, args.stage_a_lr),
        ("B", args.stage_b_epochs, args.stage_b_lr),
    ):
        if stage == "B" and best_stage_a_state is not None:
            model.load_state_dict(best_stage_a_state, strict=True)
        set_stage(model, stage)
        optimizer = optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate, weight_decay=args.weight_decay,
        )
        no_improve = 0
        stage_best_key = None
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_dataset.set_epoch(epoch)
            train_loss = train_epoch(model, train_loader, optimizer, device, stage)
            predictions, metrics = evaluate(model, val_dataset, val_loader, device)
            key = (
                metrics["val_patient_auc"], metrics["val_image_auc"], -metrics["val_loss"]
            )
            improved_overall = best["key"] is None or key > best["key"]
            if improved_overall:
                best = {
                    "key": key,
                    "state": deepcopy(model.state_dict()),
                    "stage": stage,
                    "epoch": epoch,
                    "metrics": metrics,
                }
            improved_stage = stage_best_key is None or key > stage_best_key
            if improved_stage:
                stage_best_key = key
                no_improve = 0
                if stage == "A":
                    best_stage_a_state = deepcopy(model.state_dict())
            else:
                no_improve += 1
            history.append({
                "stage": stage, "epoch": epoch, "train_loss": train_loss,
                "elapsed_seconds": time.time() - start, **metrics,
            })
            print(
                f"stage{stage} {epoch:02d}/{epochs} | train={train_loss:.4f} "
                f"val={metrics['val_loss']:.4f} | image AUC={metrics['val_image_auc']:.4f} "
                f"patient AUC={metrics['val_patient_auc']:.4f}"
            )
            if stage == "B" and no_improve >= args.patience:
                print(f"stageB early stop: 连续{args.patience}轮无阶段内改善")
                break

    model.load_state_dict(best["state"], strict=True)
    final_predictions, final_metrics = evaluate(model, val_dataset, val_loader, device)
    if not math.isclose(
        final_metrics["val_patient_auc"], best["metrics"]["val_patient_auc"], abs_tol=1e-12
    ):
        raise RuntimeError("MG1最佳checkpoint复算不一致")
    patient_predictions = patient_mean(final_predictions, "cancer_probability")

    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "mg1_best_teacher.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "efficientnet_b0",
        "input_mode": args.input_mode,
        "seed": args.seed,
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "metrics": final_metrics,
        "manifest_sha256": file_sha256(args.manifest.resolve()),
    }, checkpoint_path)
    final_predictions.to_csv(
        output_dir / "val_image_predictions.csv", index=False, encoding="utf-8-sig"
    )
    patient_predictions.to_csv(
        output_dir / "val_patient_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "stage": "MG1_grayscale_local_teacher",
        "input_mode": args.input_mode,
        "debug": args.debug,
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest.resolve()),
        "v3_audit": str(args.v3_audit.resolve()),
        "v3_audit_sha256": file_sha256(args.v3_audit.resolve()),
        "geometry_patient_auc": GEOMETRY_PATIENT_AUC,
        "train_images": int(len(train)),
        "train_patients": int(train.patient_id.nunique()),
        "val_images": int(len(val)),
        "val_patients": int(val.patient_id.nunique()),
        "input_protocol": {
            "roi": "independent_axis_dynamic_margin_v3",
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "color": "luma copied to 3 channels",
            "square_conversion": False,
            "letterbox": False,
            "bbox_jitter": False,
            "train_horizontal_flip": 0.5,
            "patch_grid": PATCH_GRID if args.input_mode == "patch_shuffle" else None,
        },
        "training": {
            "batch_size": args.batch_size,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "stage_a_lr": args.stage_a_lr,
            "stage_b_lr": args.stage_b_lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "optimizer": "AdamW",
            "label_smoothing": LABEL_SMOOTHING,
            "sampler": "patient_and_class_balanced_with_replacement",
            "selection": "val patient AUC, then image AUC, then lower val loss",
        },
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "metrics": final_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"MG1完成: mode={args.input_mode} best=stage{best['stage']} epoch{best['epoch']} "
        f"image AUC={final_metrics['val_image_auc']:.4f} "
        f"patient AUC={final_metrics['val_patient_auc']:.4f}"
    )
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
