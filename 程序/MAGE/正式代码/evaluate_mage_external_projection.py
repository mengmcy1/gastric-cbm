#!/usr/bin/env python3
"""M0-F/A-long/C-long 外部多中心描述性投影（一次性、只读、冻结阈值）。

协议见 `MAGE实验进度与结果讨论.md`「C-long外部多中心描述性投影协议
（2026-08-19用户冻结）」。要点：

- 三个模型均为seed42冻结checkpoint，阈值来自各自val，不在外部重扫；
- 三模型共用同一套外部Keep预处理图，推理为直接resize到224x224+ImageNet归一化；
- 患者聚合为图像概率算术均值；外部无病灶框，不计算PGA/AiB/nAiB；
- 正式模式强制先通过val自测闸门（AUC偏差<1e-4、冻结阈值混淆矩阵完全一致），
  且M0-F外部AUC须复现既有发布值（容差1e-4），任一失败即拒绝读取/采用外部结果；
- 结果只作探索性描述，不得用于任何模型、epoch、阈值或参数选择。

运行模式：
    默认（正式）：val自测 -> 外部投影 -> 完整指标与定性注意力子集；
    --self-test-val：只跑val自测，不读取外部；
    --debug N：只在val前N张上冒烟，输出到独立debug目录，永不读取外部。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAGE_CODE_DIR = Path(__file__).resolve().parent
TRAIN_CODE_DIR = PROJECT_ROOT / "程序" / "模型训练" / "正式代码"
sys.path.insert(0, str(MAGE_CODE_DIR))
sys.path.insert(0, str(TRAIN_CODE_DIR))

from efficientnet_train_debiased import build_model as build_m0f_model  # noqa: E402
from train_mage_mg1b_attention_teacher import AttentionPoolingTeacher  # noqa: E402


# ---------------------------------------------------------------------------
# 路径与冻结常量
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

M0F_RUN = PROJECT_ROOT / (
    "结果/M0全量诊断_0804/正式验证集筛选/m0_full_keep_efficientnet_b0_seed42"
)
MG2L_RUN_ROOT = PROJECT_ROOT / (
    "结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选"
)
ALONG_RUN = MG2L_RUN_ROOT / "mg2l_arma_efficientnet_b0_seed42"
CLONG_RUN = MG2L_RUN_ROOT / "mg2l_armc_efficientnet_b0_seed42"

EXTERNAL_PREPROCESS_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃镜多中心测试集_M0Keep预处理_v1_20260805"
)
EXTERNAL_SOURCE_MANIFEST = PROJECT_ROOT / (
    "结果/去偏重训练_v1/外部多中心完整测试_v1/preprocess_manifest.csv"
)
PREVIOUS_M0F_EXTERNAL = PROJECT_ROOT / (
    "结果/M0全量诊断_0804/外部多中心完整测试_Keep_v1"
)
PREVIOUS_M0F_PREDICTIONS = (
    PREVIOUS_M0F_EXTERNAL
    / "m0_full_keep_efficientnet_b0_seed42/image_predictions.csv"
)

DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/外部多中心描述性投影_20260819"
DEBUG_OUTPUT = PROJECT_ROOT / "结果/MAGE/调试_外部多中心描述性投影_20260819"

EXTERNAL_EXPECTED_IMAGES = 1941
EXTERNAL_EXPECTED_PATIENTS = 1329
AUC_REPRODUCTION_TOLERANCE = 1e-4
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260805
SINGLE_IMAGE_SEED = 20260806
ATTENTION_SUBSET_SEED = 20260819
STRATUM_MIN_IMAGES = 15
STRATUM_MIN_PATIENTS = 8
PREVIOUS_M0F_PREDICTIONS_SHA256 = (
    "2caded16cb6f676f23180c8dd8ade30f83f756465022168d25dba09e850e73bd"
)
PREVIOUS_M0F_METRICS_SHA256 = (
    "f98b49077c0da9b4e6961fc19bb9a916e5089f6000430c8620bef921fa134549"
)
PREVIOUS_M0F_MANIFEST_SHA256 = (
    "adb1e471286f0533344ec69373c574b03816599473be419ec5a165649dbd9925"
)

# 三模型规格；阈值与期望val指标运行时从各自config读取，不硬编码。
MODEL_SPECS = {
    "M0-F": {
        "kind": "plain",
        "run_dir": M0F_RUN,
        "weight_name": "efficientnet_b0_debiased_best.pth",
        "role": "未使用病灶注意力引导的原始基线",
        "checkpoint_sha256": (
            "d581ad73aae937655d2aba2499afd998234223518f63a0567f364ffb50ebb963"
        ),
    },
    "A-long": {
        "kind": "attention",
        "run_dir": ALONG_RUN,
        "weight_name": "mg2_arma_best_student.pth",
        "role": "相同训练预算、只学习癌/非癌的公平对照",
        "checkpoint_sha256": (
            "cae16c8f04879d702f0f43f9c85f77cd8e88a96d781c3769150a0483a812bd32"
        ),
    },
    "C-long": {
        "kind": "attention",
        "run_dir": CLONG_RUN,
        "weight_name": "mg2_armc_best_student.pth",
        "role": "使用MAGE病灶注意力引导的最终学生",
        "checkpoint_sha256": (
            "29e76977251fbd50c9fd9ecd6ebd26927eaf9b54eeb8e70a0b70d3d5a70ce014"
        ),
    },
}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """返回文件SHA256 hex；参数path为待哈希文件路径。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str) -> str:
    """将CSV路径统一解析为规范化绝对路径。

    参数:
        value (str): 绝对路径或相对项目根目录的路径。相对路径一律锚定
            ``PROJECT_ROOT``，不依赖运行时工作目录。
    返回:
        str: 规范化绝对路径字符串。
    """
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def eval_tensor(image: Image.Image) -> torch.Tensor:
    """统一评估预处理：PIL RGB直接resize到224x224双线性+ImageNet归一化。

    参数:
        image (PIL.Image): 任意尺寸的RGB图。
    返回:
        torch.Tensor: shape [3,224,224] 的归一化张量。与M0-F的
            ``build_transforms`` eval分支及MG2学生验证态逐操作一致。
    """
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return TF.normalize(
        TF.pil_to_tensor(resized).float().div(255.0), IMAGENET_MEAN, IMAGENET_STD
    )


class ProjectionDataset(Dataset):
    """只读图像清单的数据集，逐张返回(eval张量, 行号)。

    构造参数:
        frame (pd.DataFrame): 含 ``image_path``（绝对路径）列的清单。
    """

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.frame.iloc[index]["image_path"]) as handle:
            image = handle.convert("RGB")
        return eval_tensor(image), index


def collect_predictions(
    model: torch.nn.Module,
    kind: str,
    frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    """对整份清单推理，返回癌概率与可选的7x7注意力。

    参数:
        model (torch.nn.Module): 已加载权重并eval的模型。
        kind (str): ``"plain"``（M0-F，仅logits）或 ``"attention"``
            （A-long/C-long，返回logits与attention）。
        frame (pd.DataFrame): ProjectionDataset使用的清单。
        batch_size (int): 批大小；num_workers (int): DataLoader线程数。
        device (torch.device): 推理设备。
    返回:
        tuple: ``probabilities`` shape [N] 的癌概率；``attentions`` 对
            attention模型为shape [N,49]（按7x7行优先展开），plain模型为None。
    """
    loader = DataLoader(
        ProjectionDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    probabilities = np.zeros(len(frame), dtype=np.float64)
    attentions = (
        np.zeros((len(frame), 49), dtype=np.float64) if kind == "attention" else None
    )
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            if kind == "attention":
                logits, attention = model(images)
                attentions[indices.numpy()] = (
                    attention.flatten(1).double().cpu().numpy()
                )
            else:
                logits = model(images)
            probabilities[indices.numpy()] = (
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"    batch {batch_index}/{len(loader)}", flush=True)
    return probabilities, attentions


# ---------------------------------------------------------------------------
# 模型加载与冻结口径
# ---------------------------------------------------------------------------

def verify_checkpoint_sha(name: str, spec: dict, actual_sha: str) -> None:
    """核验checkpoint实际SHA等于协议硬绑定值。

    参数:
        name (str): 模型名；spec (dict): MODEL_SPECS条目（含
            ``checkpoint_sha256``）；actual_sha (str): 权重文件实际SHA256。
    返回: 无；不符即抛ValueError（快速失败）。
    """
    expected_sha = spec["checkpoint_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(
            f"{name} checkpoint SHA与协议硬绑定值不符: "
            f"{actual_sha} != {expected_sha}"
        )


def load_model_spec(name: str, spec: dict, device: torch.device) -> dict:
    """加载一个冻结模型及其val口径（阈值、期望指标、SHA绑定）。

    参数:
        name (str): ``MODEL_SPECS`` 键名。
        spec (dict): 对应规格（kind/run_dir/weight_name/role）。
        device (torch.device): 目标设备。
    返回:
        dict: 含 ``model``、``image_threshold``、``patient_threshold``、
            ``expected``（val自测目标值）与checkpoint/config实际SHA256。
            任何config字段缺失或（MG2组的）checkpoint SHA与config记录不符
            即抛错，快速失败。
    """
    run_dir = spec["run_dir"]
    config_path = run_dir / "config.json"
    weight_path = run_dir / spec["weight_name"]
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"{name}产物缺失: {config_path} 或 {weight_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_sha = sha256_of(weight_path)

    verify_checkpoint_sha(name, spec, weight_sha)
    if spec["kind"] == "attention":
        recorded_sha = config.get("checkpoint_sha256")
        if recorded_sha != weight_sha:
            raise ValueError(
                f"{name} checkpoint SHA与config记录不符: "
                f"{weight_sha} != {recorded_sha}"
            )
        metrics = config["metrics"]
        image_threshold = float(metrics["image_threshold_metrics"]["threshold"])
        patient_threshold = float(metrics["patient_threshold_metrics"]["threshold"])
        expected = {
            "val_image_auc": float(metrics["val_image_auc"]),
            "val_patient_auc": float(metrics["val_patient_auc"]),
            "val_image_cm": list(metrics["image_threshold_metrics"]["confusion_matrix"]),
            "val_patient_cm": list(
                metrics["patient_threshold_metrics"]["confusion_matrix"]
            ),
        }
        model = AttentionPoolingTeacher(pretrained=False)
    else:
        threshold = float(config["threshold_selection"]["threshold"])
        image_threshold = patient_threshold = threshold
        expected = {
            "val_image_auc": float(config["metrics"]["val_image"]["AUC"]),
            "val_patient_auc": float(config["metrics"]["val_patient"]["AUC"]),
            "val_image_cm": list(config["metrics"]["val_image"]["CM"]),
            "val_patient_cm": list(config["metrics"]["val_patient"]["CM"]),
        }
        model = build_m0f_model(pretrained=False)

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "name": name,
        "model": model.to(device).eval(),
        "image_threshold": image_threshold,
        "patient_threshold": patient_threshold,
        "expected": expected,
        "config": config,
        "config_sha256": sha256_of(config_path),
        "weight_path": weight_path,
        "weight_sha256": weight_sha,
    }


# ---------------------------------------------------------------------------
# 清单构建（val自测清单与外部清单）
# ---------------------------------------------------------------------------

def build_val_frame(name: str, spec: dict, config: dict) -> pd.DataFrame:
    """构建某模型val自测清单，列统一为image_path/patient_id/label。

    参数:
        name (str): 模型名；spec (dict): 规格；config (dict): 已加载config。
    返回:
        pd.DataFrame: 该模型val清单（M0-F取run内冻结split快照的val行；
            A-long/C-long取MG2 v3清单的val行）。行数异常即抛错。
    """
    if spec["kind"] == "attention":
        manifest = pd.read_csv(config["manifest"], dtype={"patient_id": str})
        frame = manifest.loc[manifest.split.eq("val")].copy()
    else:
        snapshot = pd.read_csv(
            spec["run_dir"] / "frozen_split_snapshot.csv",
            encoding="utf-8-sig",
            dtype={"patient_id": str},
        )
        frame = snapshot.loc[snapshot.split.eq("val")].copy()
    frame["image_path"] = frame["image_relpath"].map(
        lambda value: str(PROJECT_ROOT / value)
    )
    frame = frame[["image_path", "patient_id", "label"]].reset_index(drop=True)
    if len(frame) != 497:
        raise ValueError(f"{name} val清单行数异常: {len(frame)} != 497")
    missing = [p for p in frame.image_path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} val清单有{len(missing)}张图片缺失")
    return frame


def build_external_frame() -> pd.DataFrame:
    """构建外部Keep预处理清单（与既有M0外部评估完全同口径）。

    返回:
        pd.DataFrame: 1941行，含image_path（FOV遮罩后、含暗部人工回退）、
            patient_id、label及original_width/original_height/
            progress_bar_detected/crop_status等分层元数据。任一连接校验、
            计数（1941张/1329人）或与既有 ``external_keep_manifest.csv``
            图像集合比对失败即抛错。
    """
    source = pd.read_csv(
        EXTERNAL_SOURCE_MANIFEST, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    stage1 = pd.read_csv(
        EXTERNAL_PREPROCESS_ROOT / "01_亮度四边裁剪" / "mapping.csv",
        encoding="utf-8-sig",
    )
    fov = pd.read_csv(
        EXTERNAL_PREPROCESS_ROOT / "02_FOV遮罩" / "mapping.csv", encoding="utf-8-sig"
    )
    if stage1.review_status.eq("error").any() or fov.review_status.eq("error").any():
        raise ValueError("外部Keep预处理存在error行，拒绝评估")
    if source.original_sha256.duplicated().any():
        raise ValueError("外部测试集存在重复SHA256")

    stage1 = stage1[["source", "crop", "relative_path"]].rename(
        columns={"source": "original_path", "crop": "stage1_path"}
    )
    fov = fov[["source", "masked", "review_status", "review_reasons"]].rename(
        columns={
            "source": "stage1_path",
            "masked": "image_path",
            "review_status": "fov_review_status",
            "review_reasons": "fov_review_reasons",
        }
    )
    for frame, column in ((source, "original_path"), (stage1, "original_path"),
                          (stage1, "stage1_path"), (fov, "stage1_path")):
        frame[column] = frame[column].map(resolve_project_path)
    merged = source.merge(stage1, on="original_path", how="inner", validate="one_to_one")
    merged = merged.merge(fov, on="stage1_path", how="inner", validate="one_to_one")
    if len(merged) != len(source):
        raise ValueError(f"外部预处理连接后数量不一致: {len(merged)}/{len(source)}")

    decisions = pd.read_csv(
        EXTERNAL_PREPROCESS_ROOT / "01b_暗部裁剪安全复核" / "人工复核决定_20260805.csv",
        encoding="utf-8-sig",
    )
    decisions = decisions.loc[
        decisions.decision.eq("use_conservative"),
        ["source", "conservative_fov_masked_candidate", "decision"],
    ].rename(columns={"source": "original_path"})
    decisions["original_path"] = decisions.original_path.map(resolve_project_path)
    decisions["conservative_fov_masked_candidate"] = (
        decisions.conservative_fov_masked_candidate.map(resolve_project_path)
    )
    merged = merged.merge(decisions, on="original_path", how="left", validate="one_to_one")
    use_conservative = merged.decision.eq("use_conservative")
    merged["input_variant"] = np.where(
        use_conservative, "conservative_fov_masked", "standard_fov_masked"
    )
    merged.loc[use_conservative, "image_path"] = merged.loc[
        use_conservative, "conservative_fov_masked_candidate"
    ]
    if int(use_conservative.sum()) != len(decisions):
        raise ValueError("暗部人工回退决定未全部连接到外部清单")

    merged["image_path"] = merged.image_path.map(resolve_project_path)
    missing = [p for p in merged.image_path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少{len(missing)}张FOV处理图")
    if merged.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("外部患者跨标签")
    if (len(merged) != EXTERNAL_EXPECTED_IMAGES
            or merged.patient_id.nunique() != EXTERNAL_EXPECTED_PATIENTS):
        raise ValueError(
            f"外部清单计数异常: {len(merged)}张/"
            f"{merged.patient_id.nunique()}人，期望"
            f"{EXTERNAL_EXPECTED_IMAGES}张/{EXTERNAL_EXPECTED_PATIENTS}人"
        )

    previous = pd.read_csv(
        PREVIOUS_M0F_EXTERNAL / "external_keep_manifest.csv", encoding="utf-8-sig",
        dtype={"patient_id": str},
    )
    current_keys = set(
        zip(merged.patient_id, merged.label, merged.image_path)
    )
    previous_keys = set(
        zip(
            previous.patient_id,
            previous.label,
            previous.image_path.map(resolve_project_path),
        )
    )
    if current_keys != previous_keys:
        raise ValueError(
            "与既有M0外部评估的(患者,标签,图像路径)三元组不一致，拒绝投影"
        )
    return merged.reset_index(drop=True)


def load_archived_m0f_probabilities(image_frame: pd.DataFrame) -> np.ndarray:
    """加载并逐行绑定既有M0-F正式外部概率。

    参数:
        image_frame (pd.DataFrame): 当前冻结外部清单，含patient_id/label/image_path。
    返回:
        np.ndarray: 按image_frame原顺序排列的M0-F癌概率。历史预测、汇总和清单
            任一SHA不符，或患者/标签/图像路径三元组不能一对一连接时快速失败。
    """
    pinned_files = {
        PREVIOUS_M0F_PREDICTIONS: PREVIOUS_M0F_PREDICTIONS_SHA256,
        PREVIOUS_M0F_EXTERNAL / "external_metrics_summary.csv": (
            PREVIOUS_M0F_METRICS_SHA256
        ),
        PREVIOUS_M0F_EXTERNAL / "external_keep_manifest.csv": (
            PREVIOUS_M0F_MANIFEST_SHA256
        ),
    }
    for path, expected_sha in pinned_files.items():
        actual_sha = sha256_of(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"M0-F历史外部产物SHA不符: {path}: {actual_sha} != {expected_sha}"
            )

    archived = pd.read_csv(
        PREVIOUS_M0F_PREDICTIONS,
        encoding="utf-8-sig",
        dtype={"patient_id": str},
    )[["patient_id", "label", "image_path", "cancer_probability"]]
    archived["image_path"] = archived.image_path.map(resolve_project_path)
    current = image_frame[["patient_id", "label", "image_path"]].copy()
    current["_row_order"] = np.arange(len(current))
    merged = current.merge(
        archived,
        on=["patient_id", "label", "image_path"],
        how="left",
        validate="one_to_one",
    ).sort_values("_row_order")
    if len(merged) != len(current) or merged.cancer_probability.isna().any():
        raise ValueError("M0-F历史外部概率无法与当前冻结清单逐行一对一连接")
    return merged.cancer_probability.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# 指标、bootstrap与分层
# ---------------------------------------------------------------------------

def patient_mean(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """患者级聚合：每位患者取其图像概率算术均值。

    参数:
        frame (pd.DataFrame): 含patient_id/label/``column``的图像级表。
        column (str): 概率列名。
    返回:
        pd.DataFrame: patient_id/label/probability三列。患者跨标签即抛错。
    """
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("存在跨标签患者，拒绝聚合")
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), probability=(column, "mean")
    )


def threshold_metrics(labels, probabilities, threshold: float) -> dict:
    """在冻结阈值下计算完整操作点指标（不重扫阈值）。

    参数:
        labels: 0/1标签序列；probabilities: 癌概率序列；
        threshold (float): val冻结阈值。
    返回:
        dict: AUC/Accuracy/Sensitivity/Specificity/Precision/F1/FPR/FNR与
            confusion_matrix=[TN,FP,FN,TP]。

    阈值比较语义（协议2026-08-19补充）：冻结阈值按"Sensitivity>=0.90"规则
    选取，其值恰好等于某个被观测到的val概率，严格``>=``在不同浮点环境下
    会让贴阈值样本以1ulp之差翻转分类；因此判定为
    ``prob >= threshold 或 |prob - threshold| <= 1e-7``，即与阈值浮点噪声
    范围内等距的值一律视为阳性侧，保证操作点可复现。AUC为秩统计量，不受
    此容差影响。
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (
        (probabilities >= threshold)
        | np.isclose(probabilities, threshold, rtol=0.0, atol=1e-7)
    ).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float((prediction == labels).mean()),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity else 0.0
        ),
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(1.0 - sensitivity),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
    }


def shared_patient_bootstrap(
    patient_frames: dict[str, pd.DataFrame], iterations: int, seed: int
) -> dict:
    """共享重采样的患者级AUC bootstrap与配对差值。

    参数:
        patient_frames (dict): 模型名 -> 患者级表（patient_id/label/
            probability），各表患者集合必须一致。
        iterations (int): bootstrap次数；seed (int): 冻结种子。
    返回:
        dict: ``auc_ci``（各模型[2.5%,97.5%]分位）与 ``paired_diff_ci``
            （C-long−A-long、C-long−M0-F、A-long−M0-F的差值分位与中位差）。
            同一轮重采样同时用于所有模型，保证差值是配对的。
    """
    names = list(patient_frames)
    base = patient_frames[names[0]]
    for name in names[1:]:
        if not base.patient_id.equals(patient_frames[name].patient_id):
            raise ValueError("各模型患者集合或顺序不一致，拒绝配对bootstrap")
    patient_ids = base.patient_id.to_numpy()
    labels = base.label.to_numpy(dtype=int)
    by_class = {
        label: np.flatnonzero(labels == label) for label in (0, 1)
    }
    probs = {
        name: patient_frames[name].probability.to_numpy(dtype=float)
        for name in names
    }
    rng = np.random.default_rng(seed)
    auc_records = {name: [] for name in names}
    for _ in range(iterations):
        sampled = np.concatenate([
            rng.choice(by_class[0], size=len(by_class[0]), replace=True),
            rng.choice(by_class[1], size=len(by_class[1]), replace=True),
        ])
        sampled_labels = labels[sampled]
        for name in names:
            auc_records[name].append(
                roc_auc_score(sampled_labels, probs[name][sampled])
            )
    result = {"auc_ci": {}, "paired_diff_ci": {}}
    for name in names:
        result["auc_ci"][name] = {
            "mean": float(np.mean(auc_records[name])),
            "ci95": np.quantile(auc_records[name], [0.025, 0.975]).tolist(),
        }
    for left, right in (("C-long", "A-long"), ("C-long", "M0-F"), ("A-long", "M0-F")):
        if left in probs and right in probs:
            diffs = np.asarray(auc_records[left]) - np.asarray(auc_records[right])
            result["paired_diff_ci"][f"{left} - {right}"] = {
                "median_diff": float(np.median(diffs)),
                "ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
            }
    return result


def random_single_image_auc(
    image_frame: pd.DataFrame, probability_columns: list[str],
    iterations: int, seed: int
) -> dict:
    """随机单图敏感性分析：每患者随机抽1图计算患者级AUC的分布。

    参数:
        image_frame (pd.DataFrame): 图像级表（patient_id/label/概率列）。
        probability_columns (list[str]): 各模型概率列名。
        iterations (int): 重复次数；seed (int): 冻结种子。各模型共用同一批
            抽图结果，保持可比。
    返回:
        dict: 模型名 -> {mean, ci95}。
    """
    patients = image_frame[["patient_id", "label"]].drop_duplicates().reset_index(
        drop=True
    )
    indices_by_patient = image_frame.groupby("patient_id", sort=False).indices
    grouped = [indices_by_patient[p] for p in patients.patient_id]
    labels = patients.label.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    picks = np.empty((iterations, len(patients)), dtype=np.int64)
    for patient_index, row_indices in enumerate(grouped):
        picks[:, patient_index] = rng.choice(
            row_indices, size=iterations, replace=True
        )
    result = {}
    for column in probability_columns:
        probabilities = image_frame[column].to_numpy(dtype=float)
        aucs = [
            roc_auc_score(labels, probabilities[picks[i]])
            for i in range(iterations)
        ]
        result[column] = {
            "mean": float(np.mean(aucs)),
            "ci95": np.quantile(aucs, [0.025, 0.975]).tolist(),
        }
    return result


def resolution_tier(width: float) -> str:
    """按原始宽度分四层；参数width为像素宽度，返回层名字符串。"""
    if width <= 800:
        return "<=800"
    if width <= 1100:
        return "801-1100"
    if width <= 1300:
        return "1101-1300"
    return ">1300"


def stratified_auc(
    image_frame: pd.DataFrame, probability_columns: list[str]
) -> list[dict]:
    """按冻结分层字段逐层计算各模型图像/患者AUC。

    参数:
        image_frame (pd.DataFrame): 含分层列与各模型概率列的图像级表。
        probability_columns (list[str]): 模型概率列名。
    返回:
        list[dict]: 每行含field/value/模型/层级/AUC或NA原因。层内图像<15
            或患者<8或只剩单类别时记NA。分层字段：分辨率宽度四层、
            progress_bar_detected、crop_status（协议冻结）。
    """
    frame = image_frame.copy()
    frame["resolution_tier"] = frame.original_width.map(resolution_tier)
    frame["progress_bar"] = frame.progress_bar_detected.map(
        lambda value: "含界面进度条" if value else "无界面进度条"
    )
    records = []
    for field in ("resolution_tier", "progress_bar", "crop_status"):
        for value, group in frame.groupby(field, sort=True):
            for column in probability_columns:
                for level in ("image", "patient"):
                    if level == "patient":
                        table = patient_mean(
                            group[["patient_id", "label", column]], column
                        ).rename(columns={"probability": column})
                    else:
                        table = group
                    enough = (
                        len(group) >= STRATUM_MIN_IMAGES
                        and group.patient_id.nunique() >= STRATUM_MIN_PATIENTS
                        and table.label.nunique() == 2
                    )
                    records.append({
                        "field": field,
                        "value": str(value),
                        "model": column,
                        "level": level,
                        "images": int(len(group)),
                        "patients": int(group.patient_id.nunique()),
                        "auc": (
                            float(roc_auc_score(table.label, table[column]))
                            if enough else None
                        ),
                        "na_reason": (
                            None if enough else "层内样本不足或只剩单类别"
                        ),
                    })
    return records


# ---------------------------------------------------------------------------
# 定性注意力展示子集
# ---------------------------------------------------------------------------

def select_attention_subset(image_frame: pd.DataFrame, per_group: int = 12) -> dict:
    """按C-long外部概率冻结选取定性展示子集。

    参数:
        image_frame (pd.DataFrame): 含patient_id/label/C-long概率列的图像级表。
        per_group (int): 每组张数（协议冻结为12）。
    返回:
        dict: 组名 -> 行号列表。组：癌概率最低/最高的癌图各12、癌概率最低/
            最高的非癌图各12、seed 20260819均匀随机12（从前48张之外抽取，
            保证全子集恰好60张不重复）。
    """
    cancer = image_frame.index[image_frame.label.eq(1)]
    noncancer = image_frame.index[image_frame.label.eq(0)]
    probs = image_frame["C-long"]
    subset = {
        "cancer_lowest_probability": probs.loc[cancer].nsmallest(per_group).index.tolist(),
        "cancer_highest_probability": probs.loc[cancer].nlargest(per_group).index.tolist(),
        "noncancer_highest_probability": (
            probs.loc[noncancer].nlargest(per_group).index.tolist()
        ),
        "noncancer_lowest_probability": (
            probs.loc[noncancer].nsmallest(per_group).index.tolist()
        ),
    }
    fixed = {i for indices in subset.values() for i in indices}
    if len(fixed) != 4 * per_group:
        raise ValueError(f"极值四组存在重叠: {len(fixed)} != {4 * per_group}")
    pool = np.array([i for i in image_frame.index if i not in fixed])
    if len(pool) < per_group:
        raise ValueError(f"随机组可抽池不足: {len(pool)} < {per_group}")
    rng = np.random.default_rng(ATTENTION_SUBSET_SEED)
    subset["random_seed20260819"] = sorted(
        rng.choice(pool, size=per_group, replace=False).tolist()
    )
    return subset


def render_attention_panels(
    image_frame: pd.DataFrame,
    attentions: dict[str, np.ndarray],
    indices: list[int],
    output_dir: Path,
) -> None:
    """为选定图像渲染A-long/C-long注意力叠加对照图。

    参数:
        image_frame (pd.DataFrame): 图像级表（含image_path/patient_id/label
            与各模型概率列）。
        attentions (dict): 模型名 -> [N,49]注意力数组（仅A-long/C-long）。
        indices (list[int]): 去重后的行号列表。
        output_dir (Path): 输出目录（自动创建）。每张输出PNG为三联：
            模型所见输入图、C-long叠加、A-long叠加。同一案例的A/C两张
            热图共享同一色标（vmin=0，vmax=两者最大值），保证颜色深浅
            可横向比较；色标范围标注在标题中。
    返回: 无。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 服务器默认的 DejaVu Sans 不含中文字形，显式使用已安装的 CJK 字体。
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    for row_number, index in enumerate(indices, start=1):
        row = image_frame.iloc[index]
        with Image.open(row.image_path) as handle:
            image = handle.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
        vmax = float(
            max(attentions["C-long"][index].max(), attentions["A-long"][index].max())
        )
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("输入图（Keep预处理+FOV遮罩）")
        for axis, name in ((axes[1], "C-long"), (axes[2], "A-long")):
            attention = attentions[name][index].reshape(7, 7)
            axis.imshow(image)
            axis.imshow(
                attention, cmap="jet", alpha=0.45, vmin=0.0, vmax=vmax,
                extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), interpolation="bilinear",
            )
            axis.set_title(
                f"{name}注意力 p={row[name]:.3f}"
            )
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"label={int(row.label)} patient={row.patient_id} "
            f"| A/C共享色标[0,{vmax:.3f}]",
            fontsize=8,
        )
        figure.savefig(
            output_dir / f"{row_number:02d}_attention_compare.png", dpi=110
        )
        plt.close(figure)


# ---------------------------------------------------------------------------
# val自测闸门
# ---------------------------------------------------------------------------

def run_val_self_test(
    name: str, loaded: dict, batch_size: int, num_workers: int,
    device: torch.device, limit: int | None = None,
) -> dict:
    """在模型各自val清单上重算指标并与冻结config比对。

    参数:
        name (str): 模型名；loaded (dict): load_model_spec返回值；
        batch_size/num_workers: 推理参数；device: 设备；
        limit (int|None): debug时只用前N张（此时不作精确断言）。
    返回:
        dict: 重算值与期望值及是否通过。正式模式（limit=None）下AUC偏差
            >=1e-4或混淆矩阵不完全一致即抛错，拒绝进入外部投影。
    """
    frame = build_val_frame(name, MODEL_SPECS[name], loaded["config"])
    if limit is not None:
        frame = frame.head(limit).copy()
    print(f"[self-test] {name}: val推理{len(frame)}张", flush=True)
    probabilities, _ = collect_predictions(
        loaded["model"], MODEL_SPECS[name]["kind"], frame,
        batch_size, num_workers, device,
    )
    frame["cancer_probability"] = probabilities
    patients = patient_mean(frame, "cancer_probability")
    image_metrics = threshold_metrics(
        frame.label, frame.cancer_probability, loaded["image_threshold"]
    )
    patient_metrics = threshold_metrics(
        patients.label, patients.probability, loaded["patient_threshold"]
    )
    expected = loaded["expected"]
    result = {
        "model": name,
        "val_image_auc": image_metrics["auc"],
        "expected_image_auc": expected["val_image_auc"],
        "val_patient_auc": patient_metrics["auc"],
        "expected_patient_auc": expected["val_patient_auc"],
        "val_image_cm": image_metrics["confusion_matrix"],
        "expected_image_cm": expected["val_image_cm"],
        "val_patient_cm": patient_metrics["confusion_matrix"],
        "expected_patient_cm": expected["val_patient_cm"],
    }
    if limit is not None:
        result["debug_only"] = True
        return result
    checks = [
        abs(result["val_image_auc"] - expected["val_image_auc"])
        < AUC_REPRODUCTION_TOLERANCE,
        abs(result["val_patient_auc"] - expected["val_patient_auc"])
        < AUC_REPRODUCTION_TOLERANCE,
        result["val_image_cm"] == expected["val_image_cm"],
        result["val_patient_cm"] == expected["val_patient_cm"],
    ]
    result["passed"] = bool(all(checks))
    if not result["passed"]:
        raise RuntimeError(
            f"{name} val自测未通过，拒绝读取外部: {json.dumps(result, ensure_ascii=False)}"
        )
    print(
        f"[self-test] {name} 通过: 患者AUC={result['val_patient_auc']:.6f} "
        f"图像AUC={result['val_image_auc']:.6f}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行；返回argparse.Namespace。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument(
        "--self-test-val", action="store_true",
        help="只跑val自测闸门，不读取外部",
    )
    parser.add_argument(
        "--debug", type=int, default=0,
        help=">0时只在val前N张冒烟，输出到debug目录，永不读取外部",
    )
    return parser.parse_args()


def main() -> None:
    """正式流程：val自测 -> 外部投影 -> 指标/bootstrap/分层/定性子集输出。"""
    args = parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定cuda但不可用，快速失败")

    debug_mode = args.debug > 0
    output = DEBUG_OUTPUT if (debug_mode or args.self_test_val) else args.output
    if args.self_test_val or debug_mode:
        output.mkdir(parents=True, exist_ok=True)
    elif output.exists():
        raise FileExistsError(f"正式输出目录已存在，先停止核对: {output}")
    else:
        output.mkdir(parents=True)

    loaded = {
        name: load_model_spec(name, spec, device)
        for name, spec in MODEL_SPECS.items()
    }
    self_test_results = {}
    for name in MODEL_SPECS:
        self_test_results[name] = run_val_self_test(
            name, loaded[name], args.batch_size, args.num_workers, device,
            limit=args.debug if debug_mode else None,
        )
    if args.self_test_val or debug_mode:
        report_name = (
            "val_self_test_debug.json" if debug_mode else "val_self_test_full.json"
        )
        (output / report_name).write_text(
            json.dumps(self_test_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"自测结果已写入 {output / report_name}", flush=True)
        return

    print("[formal] val自测全部通过，开始外部投影", flush=True)
    external = build_external_frame()
    print(f"[formal] 外部清单: {len(external)}张/{external.patient_id.nunique()}人",
          flush=True)

    image_frame = external[[
        "patient_id", "label", "image_path", "original_width", "original_height",
        "progress_bar_detected", "crop_status", "original_relpath",
    ]].copy()
    attentions: dict[str, np.ndarray] = {}
    image_frame["M0-F"] = load_archived_m0f_probabilities(image_frame)

    # M0-F沿用SHA锁定的既有正式外部概率，并复算既有发布指标。
    previous = pd.read_csv(
        PREVIOUS_M0F_EXTERNAL / "external_metrics_summary.csv", encoding="utf-8-sig"
    )
    reference = previous.loc[
        previous.run_name.eq("m0_full_keep_efficientnet_b0_seed42")
    ].iloc[0]
    patients_m0f = patient_mean(
        image_frame[["patient_id", "label", "M0-F"]], "M0-F"
    )
    reproduced_patient = roc_auc_score(patients_m0f.label, patients_m0f.probability)
    reproduced_image = roc_auc_score(image_frame.label, image_frame["M0-F"])
    reproduction = {
        "patient_auc": float(reproduced_patient),
        "expected_patient_auc": float(reference.patient_AUC),
        "image_auc": float(reproduced_image),
        "expected_image_auc": float(reference.image_AUC),
    }
    if (abs(reproduced_patient - reference.patient_AUC) >= AUC_REPRODUCTION_TOLERANCE
            or abs(reproduced_image - reference.image_AUC) >= AUC_REPRODUCTION_TOLERANCE):
        raise RuntimeError(
            f"M0-F历史外部概率复算失败，整轮作废: {json.dumps(reproduction)}"
        )
    print(f"[formal] M0-F历史外部概率复算通过: {reproduction}", flush=True)

    for name in ("A-long", "C-long"):
        spec = MODEL_SPECS[name]
        print(f"[formal] {name} 外部推理", flush=True)
        probabilities, attention = collect_predictions(
            loaded[name]["model"], spec["kind"], image_frame,
            args.batch_size, args.num_workers, device,
        )
        image_frame[name] = probabilities
        attentions[name] = attention

    # 三模型完整指标
    metrics = {}
    patient_frames = {}
    for name in MODEL_SPECS:
        patients = patient_mean(image_frame[["patient_id", "label", name]], name)
        patient_frames[name] = patients
        metrics[name] = {
            "image": threshold_metrics(
                image_frame.label, image_frame[name], loaded[name]["image_threshold"]
            ),
            "patient": threshold_metrics(
                patients.label, patients.probability, loaded[name]["patient_threshold"]
            ),
        }
    bootstrap = shared_patient_bootstrap(
        patient_frames, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
    )
    single_image = random_single_image_auc(
        image_frame, list(MODEL_SPECS), BOOTSTRAP_ITERATIONS, SINGLE_IMAGE_SEED
    )
    stratification = stratified_auc(image_frame, list(MODEL_SPECS))

    # 定性注意力子集
    subset = select_attention_subset(image_frame)
    subset_indices = sorted({i for indices in subset.values() for i in indices})
    render_attention_panels(
        image_frame, attentions, subset_indices, output / "qualitative_attention"
    )

    # 落盘
    attention_frame = pd.DataFrame({
        f"{name}_attention_{cell}": attentions[name][:, cell]
        for name in attentions for cell in range(49)
    })
    image_output = pd.concat(
        [image_frame.reset_index(drop=True), attention_frame], axis=1
    )
    image_output.to_csv(
        output / "external_image_predictions.csv", index=False, encoding="utf-8-sig"
    )
    patient_output = patient_frames["M0-F"][["patient_id", "label"]].copy()
    for name in MODEL_SPECS:
        patient_output[f"{name}_probability"] = patient_frames[name].probability
        patient_output[f"{name}_image_count"] = patient_output.patient_id.map(
            image_frame.groupby("patient_id").size()
        )
    patient_output.to_csv(
        output / "external_patient_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stratification).to_csv(
        output / "external_stratified_auc.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "stage": "MAGE_external_descriptive_projection",
        "protocol_frozen": "2026-08-19",
        "question": (
            "C-long在内部val注意力改善后，能否在外部多中心保持分类能力"
            "甚至改善泛化（探索性描述，非确证）"
        ),
        "models": {
            name: {
                "role": MODEL_SPECS[name]["role"],
                "weight_path": str(loaded[name]["weight_path"]),
                "weight_sha256": loaded[name]["weight_sha256"],
                "config_sha256": loaded[name]["config_sha256"],
                "image_threshold": loaded[name]["image_threshold"],
                "patient_threshold": loaded[name]["patient_threshold"],
                "threshold_source": "各模型val冻结，未在外部重扫",
                "external_probability_source": (
                    "SHA锁定的2026-08-05正式逐图预测"
                    if name == "M0-F" else "本轮冻结checkpoint只读推理"
                ),
            }
            for name in MODEL_SPECS
        },
        "val_self_test": self_test_results,
        "m0f_external_reproduction": reproduction,
        "external_cohort": {
            "images": int(len(image_frame)),
            "patients": int(image_frame.patient_id.nunique()),
            "cancer_images": int(image_frame.label.sum()),
            "cancer_patients": int(
                image_frame.groupby("patient_id").label.first().sum()
            ),
            "note": "该队列已因Y6工程决策降级为外部开发/验证队列，结果为探索性描述",
        },
        "metrics": metrics,
        "patient_bootstrap": bootstrap,
        "random_single_image_auc": single_image,
        "attention_subset": {key: [int(i) for i in value] for key, value in subset.items()},
        "boundaries": [
            "阈值来自val冻结，未在外部重扫",
            "不根据外部结果更换epoch/checkpoint/参数",
            "外部无框，未计算PGA/AiB/nAiB",
            "internal test未读取",
            "结果不回灌MG2判定，不改变SAE计划",
        ],
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": True,
    }
    (output / "external_projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[formal] 完成，输出目录: {output}", flush=True)
    for name in MODEL_SPECS:
        image_m = metrics[name]["image"]
        patient_m = metrics[name]["patient"]
        print(
            f"  {name}: 患者AUC={patient_m['auc']:.4f} "
            f"图像AUC={image_m['auc']:.4f} "
            f"Sens={patient_m['sensitivity']:.4f} Spec={patient_m['specificity']:.4f}",
            flush=True,
        )
    for pair, payload in bootstrap["paired_diff_ci"].items():
        print(f"  配对差值 {pair}: {payload['median_diff']:+.4f} "
              f"CI95={payload['ci95']}", flush=True)


if __name__ == "__main__":
    main()
