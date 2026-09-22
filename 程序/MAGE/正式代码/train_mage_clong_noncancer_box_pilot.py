#!/usr/bin/env python3
"""从同一冻结C-long起点微调两臂，检验非癌矩形框定位监督的有限效果。

两臂保留原CE、logit KD和癌侧attention KD；候选另加独立的非癌框KL。
固定5轮stage-B式微调，不根据val挑epoch，不读取internal test或external。
主要观察患者灵敏度至少90%时的特异度，并同时报告原阈值误报/漏诊。
验证集选择的评价阈值仅用于开发性比较，不自动替换部署阈值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from train_mage_mg1_teacher import (  # noqa: E402
    DEFAULT_MANIFEST, DEFAULT_V3_AUDIT, load_manifest,
    patient_class_balanced_weights, patient_mean, seed_everything,
)
from train_mage_mg1b_attention_teacher import (  # noqa: E402
    AttentionPoolingTeacher, cell_overlap_map, set_stage, set_train_mode,
    target_distribution,
)
from train_mage_mg2_student import (  # noqa: E402
    ALPHA, TAU, LABEL_SMOOTHING, DEFAULT_TEACHER_CACHE, FROZEN_TEACHER_SHA256,
    MG2StudentDataset, compute_mg2_losses, evaluate, flip_box_horizontal,
    lesion_fraction_tercile_bounds, load_teacher_cache, seed_worker,
)

SEED = 42
BATCH = 32
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
BETA = 0.19362648121926898
THRESHOLD = 0.3074711561203003
START = ROOT / ("结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选/"
                "mg2l_armc_efficientnet_b0_seed42/mg2_armc_best_student.pth")
START_SHA = "29e76977251fbd50c9fd9ecd6ebd26927eaf9b54eeb8e70a0b70d3d5a70ce014"
BOX_ROOT = ROOT / "结果/SAE/RA_SAE_Pilot_20260908/noncancer_box_alignment_20260916_v3"
BOX_COLUMNS = ["box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm"]


def attach_noncancer_boxes(frame: pd.DataFrame) -> pd.DataFrame:
    """按图像完整路径接入已核验的非癌框，保持原train/val行序。

    Args:
        frame (pd.DataFrame): 冻结v3 train/val manifest，共2847图。
    Returns:
        pd.DataFrame: 原行序及带``noncancer_``前缀的归一化框坐标；癌图为空。
    """
    tables = []
    for split, expected in (("train", 1169), ("val", 257)):
        box = pd.read_csv(BOX_ROOT / f"{split}_matched_noncancer_boxes.csv")
        if len(box) != expected or box.image_relpath.nunique() != expected:
            raise RuntimeError(f"{split}非癌框数量或主键异常")
        if not (box.width_matches & box.height_matches).all():
            raise RuntimeError(f"{split}存在图像/XML尺寸不一致")
        box = box[["image_relpath", "split", *BOX_COLUMNS]].rename(
            columns={name: f"noncancer_{name}" for name in BOX_COLUMNS}
        )
        tables.append(box)
    output = frame.merge(pd.concat(tables), on=["image_relpath", "split"],
                         how="left", sort=False, validate="one_to_one")
    has_box = output[f"noncancer_{BOX_COLUMNS[0]}"].notna()
    if not has_box.equals(output.label.eq(0)):
        raise RuntimeError("非癌框与v3 manifest标签/划分未一对一对应")
    for split, count in (("train", 1169), ("val", 257)):
        if int((output.split.eq(split) & has_box).sum()) != count:
            raise RuntimeError(f"{split}非癌框覆盖不足")
    coords = output.loc[has_box, [f"noncancer_{name}" for name in BOX_COLUMNS]].to_numpy(float)
    if (not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any()
            or (coords[:, 0] >= coords[:, 2]).any()
            or (coords[:, 1] >= coords[:, 3]).any()):
        raise RuntimeError("非癌框归一化坐标无效")
    return output


class BoxStudentDataset(MG2StudentDataset):
    """保留原癌侧``has_target``，另给非癌图提供独立的框目标和掩码。

    构造参数、训练增强及缓存行为继承MG2StudentDataset；``forward``由学生模型执行。
    ``__getitem__``额外返回``noncancer_target[49]``、``has_noncancer_box``。
    """

    def __getitem__(self, index: int) -> dict:
        """读取单图、同步翻转非癌框并形成7×7面积目标。

        Args:
            index (int): 当前split中的行号。
        Returns:
            dict: 原MG2字段及独立非癌目标``[49]``和布尔掩码。
        """
        sample = super().__getitem__(index)
        row = self.frame.iloc[index]
        is_noncancer = int(row.label) == 0
        target = np.zeros(49, dtype=np.float32)
        if is_noncancer:
            box = np.array([row[f"noncancer_{name}"] for name in BOX_COLUMNS], dtype=np.float64)
            from train_mage_mg1_teacher import stable_uniform
            if self.training and stable_uniform(str(row.sha256), self.epoch, self.seed, "flip") < 0.5:
                box = flip_box_horizontal(box)
            target = target_distribution(cell_overlap_map(box)).reshape(-1)
        sample["noncancer_target"] = torch.from_numpy(target.copy())
        sample["has_noncancer_box"] = torch.tensor(is_noncancer, dtype=torch.bool)
        return sample


def noncancer_attention_kl(attention: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    """只对有框非癌图计算``KL(q_box || student_attention)``。

    Args:
        attention (torch.Tensor): softmax后``[B,1,7,7]``注意力。
        targets (torch.Tensor): 非癌面积目标``[B,49]``；癌图全零。
        mask (torch.Tensor): ``[B]``布尔非癌框掩码。
    Returns:
        torch.Tensor: 有框非癌图的平均KL；无此类图时为连接计算图的零。
    """
    if not bool(mask.any()):
        return attention.sum() * 0.0
    target = targets[mask]
    prediction = attention[mask].flatten(1).clamp_min(1e-8)
    positive = target > 0
    terms = torch.where(positive,
                        target * (target.clamp_min(1e-8).log() - prediction.log()),
                        torch.zeros_like(target))
    return terms.sum(dim=1).mean()


def make_loader(dataset: BoxStudentDataset, frame: pd.DataFrame, workers: int) -> DataLoader:
    """按原MG2患者/类别平衡有放回抽样构造确定性训练加载器。

    Args:
        dataset (BoxStudentDataset): 本臂训练数据集。
        frame (pd.DataFrame): 同序train manifest。
        workers (int): DataLoader工作进程数。
    Returns:
        DataLoader: batch32、每轮2350次有放回抽样的加载器。
    """
    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(patient_class_balanced_weights(frame), len(frame),
                                    replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=BATCH, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=generator)


def verify_paired_inputs(train: pd.DataFrame) -> dict:
    """核对两臂重置随机状态后首批实际增强输入、抽样和框目标相同。

    Args:
        train (pd.DataFrame): 同一训练清单；仅取前若干患者的小样本。
    Returns:
        dict: 实际输入张量和抽样行号的配对核对结果。
    """
    selected = train[["patient_id", "label"]].drop_duplicates().groupby(
        "label", group_keys=False
    ).head(2)
    frame = train[train.patient_id.isin(selected.patient_id)].reset_index(drop=True)
    batches = []
    for _ in range(2):
        seed_everything(SEED)
        random.seed(SEED)
        dataset = BoxStudentDataset(frame, True, SEED, None)
        dataset.set_epoch(1)
        batches.append(next(iter(make_loader(dataset, frame, 0))))
    for key in ("row_index", "image", "label", "has_target", "has_noncancer_box",
                "noncancer_target"):
        if not torch.equal(batches[0][key], batches[1][key]):
            raise RuntimeError(f"两臂随机状态重置后实际小样本输入不一致: {key}")
    return {"images_compared": len(batches[0]["label"]),
            "sampled_rows_equal": True, "augmented_tensors_equal": True,
            "box_targets_equal": True}


def fixed_threshold_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按同一冻结患者阈值评估AUC、Sens、Spec和错误人数。

    Args:
        predictions (pd.DataFrame): val逐图癌概率、患者ID和真实标签。
    Returns:
        tuple[pd.DataFrame, dict]: 患者均值概率表及固定阈值指标。
    """
    patients = patient_mean(predictions, "cancer_probability")
    return patients, patient_classification_summary(patients)


def patient_classification_summary(patients: pd.DataFrame) -> dict:
    """汇总固定阈值与相近灵敏度下的患者分类表现。

    Args:
        patients (pd.DataFrame): 每患者一行，label为0/1，probability为图片概率均值。
    Returns:
        dict: 原阈值指标、非癌高分数量及验证灵敏度至少90%的最高可行阈值指标。
    """
    predicted = patients.probability.ge(THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(patients.label, predicted, labels=[0, 1]).ravel()
    positive = patients.loc[patients.label.eq(1), "probability"].sort_values(ascending=False)
    required_tp = int(np.ceil(.9 * len(positive)))
    comparison_threshold = float(positive.iloc[required_tp - 1])
    # 包含阈值处全部同分患者，因此实际灵敏度可能高于90%，必须如实报告。
    tn90, fp90, fn90, tp90 = confusion_matrix(
        patients.label, patients.probability.ge(comparison_threshold).astype(int),
        labels=[0, 1],
    ).ravel()
    return {
        "patient_auc": float(roc_auc_score(patients.label, patients.probability)),
        "threshold": THRESHOLD,
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "accuracy": float((tp + tn) / len(patients)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "noncancer_probability_ge_0p8": int((patients.label.eq(0) & patients.probability.ge(.8)).sum()),
        "at_sensitivity90": {
            "threshold": comparison_threshold,
            "threshold_role": "validation_development_comparison_only",
            "sensitivity": float(tp90 / (tp90 + fn90)),
            "specificity": float(tn90 / (tn90 + fp90)),
            "tn": int(tn90), "fp": int(fp90), "fn": int(fn90), "tp": int(tp90),
        },
    }


def classification_comparison(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """对齐患者，报告候选相对原模型或同预算对照的分类变化。

    Args:
        reference (pd.DataFrame): 参照患者ID、label、probability。
        candidate (pd.DataFrame): 相同患者的候选结果。
    Returns:
        dict: 固定阈值错误转换、相近灵敏度特异度差及参照/候选分类指标。
    """
    joined = reference.merge(candidate, on=["patient_id", "label"],
                             suffixes=("_reference", "_candidate"), validate="one_to_one")
    if len(joined) != len(reference) or len(joined) != len(candidate):
        raise RuntimeError("分类比较患者或标签不一致")
    r, c = patient_classification_summary(reference), patient_classification_summary(candidate)
    old = joined.probability_reference.ge(THRESHOLD)
    new = joined.probability_candidate.ge(THRESHOLD)
    negative = joined.label.eq(0)
    positive = joined.label.eq(1)
    return {
        "specificity_delta_at_sensitivity90": c["at_sensitivity90"]["specificity"] - r["at_sensitivity90"]["specificity"],
        "actual_sensitivity_delta": c["at_sensitivity90"]["sensitivity"] - r["at_sensitivity90"]["sensitivity"],
        "patient_auc_delta": c["patient_auc"] - r["patient_auc"],
        "fixed_threshold_corrected_fp": int((negative & old & ~new).sum()),
        "fixed_threshold_new_fp": int((negative & ~old & new).sum()),
        "fixed_threshold_corrected_fn": int((positive & ~old & new).sum()),
        "fixed_threshold_new_fn": int((positive & old & ~new).sum()),
        "reference": r, "candidate": c,
    }


def spatial_metrics(predictions: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """分癌/非癌计算面积参照AiB、归一化AiB和峰值入框率。

    Args:
        predictions (pd.DataFrame): 含49格attention的val逐图预测。
        frame (pd.DataFrame): 与预测row_index一致的val manifest及非癌框。
    Returns:
        dict: 各标签图像均值、患者等权均值及大框排除数。
    """
    rows = []
    for record in predictions.itertuples(index=False):
        source = frame.iloc[int(record.row_index)]
        if int(source.label) == 0:
            box = np.array([source[f"noncancer_{name}"] for name in BOX_COLUMNS], dtype=np.float64)
        else:
            box = np.array([source.bbox_x1_norm, source.bbox_y1_norm,
                            source.bbox_x2_norm, source.bbox_y2_norm], dtype=np.float64)
        mass = np.array([getattr(record, f"attention_{y}_{x}")
                         for y in range(7) for x in range(7)], dtype=np.float64).reshape(7, 7)
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        aib = float((mass * cell_overlap_map(box)).sum())
        peak_y, peak_x = divmod(int(mass.argmax()), 7)
        px, py = (peak_x + .5) / 7, (peak_y + .5) / 7
        rows.append({"label": int(source.label), "patient_id": str(source.patient_id),
                     "area": area, "aib": aib, "excess_aib": aib - area,
                     "normalized_aib": (aib - area) / (1 - area) if area < .99 else np.nan,
                     "pga": float(box[0] <= px <= box[2] and box[1] <= py <= box[3])})
    images = pd.DataFrame(rows)
    output = {}
    for label, name in ((0, "noncancer"), (1, "cancer")):
        local = images[images.label.eq(label)]
        patients = local.groupby("patient_id", as_index=False)[
            ["area", "aib", "excess_aib", "normalized_aib", "pga"]
        ].mean()
        output[name] = {"images": len(local), "patients": len(patients),
                        "normalized_aib_evaluable_images": int(local.normalized_aib.notna().sum()),
                        "normalized_aib_evaluable_patients": int(patients.normalized_aib.notna().sum()),
                        "excluded_area_ge_0_99": int(local.normalized_aib.isna().sum()),
                        "image_mean": local[["area", "aib", "excess_aib", "normalized_aib", "pga"]].mean().to_dict(),
                        "patient_equal_mean": patients[["area", "aib", "excess_aib", "normalized_aib", "pga"]].mean().to_dict()}
    return output


def run_arm(arm: str, start_state: dict, train: pd.DataFrame, val: pd.DataFrame,
            cache: dict, device: torch.device, workers: int, output: Path,
            debug: bool) -> tuple[pd.DataFrame, dict]:
    """从相同C-long权重运行一臂，固定末轮评价而不按val选择。

    Args:
        arm (str): ``control``或``noncancer_box``。
        start_state (dict): 冻结C-long完整model_state_dict。
        train (pd.DataFrame): train manifest及独立非癌框。
        val (pd.DataFrame): val manifest及独立非癌框，仅评价。
        cache (dict): 已校验的冻结教师logit/attention缓存。
        device (torch.device): CPU或已核对的CUDA设备。
        workers (int): DataLoader进程数。
        output (Path): 独立实验输出目录。
        debug (bool): 小样本链路验证模式。
    Returns:
        tuple[pd.DataFrame, dict]: val患者预测和分类/空间指标。
    """
    seed_everything(SEED)
    random.seed(SEED)
    train_set = BoxStudentDataset(train, True, SEED, cache)
    val_set = BoxStudentDataset(val, False, SEED, cache)
    train_loader = make_loader(train_set, train, workers)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    model = AttentionPoolingTeacher(pretrained=False).to(device)
    model.load_state_dict(start_state, strict=True)
    set_stage(model, "B")
    if any(parameter.requires_grad for parameter in model.features[:5].parameters()):
        raise RuntimeError("冻结backbone前五块意外被解冻")
    frozen_before = {name: value.detach().clone() for name, value in model.features[:5].state_dict().items()}
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    epochs = 1 if debug else EPOCHS
    expected_updates = (len(train) + BATCH - 1) // BATCH
    for epoch in range(1, epochs + 1):
        train_set.set_epoch(epoch)
        set_train_mode(model, "B")
        totals = {name: 0.0 for name in ("total", "ce", "kd_logit", "kd_cancer_raw",
                                              "kd_cancer_weighted", "kl_noncancer_raw",
                                              "kl_noncancer_weighted")}
        seen = updates = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            teacher_logits = batch["teacher_logits"].to(device, non_blocking=True)
            teacher_target = batch["target"].to(device, non_blocking=True)
            cancer_mask = batch["has_target"].to(device, non_blocking=True)
            noncancer_target = batch["noncancer_target"].to(device, non_blocking=True)
            noncancer_mask = batch["has_noncancer_box"].to(device, non_blocking=True)
            if bool((cancer_mask & noncancer_mask).any()):
                raise RuntimeError("癌图与非癌框损失掩码重叠")
            optimizer.zero_grad(set_to_none=True)
            logits, attention = model(images)
            old = compute_mg2_losses("C", logits, attention, labels, teacher_logits,
                                     teacher_target, cancer_mask, BETA, LABEL_SMOOTHING)
            new_raw = noncancer_attention_kl(attention, noncancer_target, noncancer_mask)
            new_weighted = BETA * new_raw if arm == "noncancer_box" else new_raw * 0.0
            loss = old["total"] + new_weighted
            loss.backward()
            optimizer.step()
            updates += 1
            batch_size = len(labels)
            parts = {"total": loss, "ce": old["ce"], "kd_logit": old["weighted_logit"],
                     "kd_cancer_raw": old["kd_attention"],
                     "kd_cancer_weighted": old["weighted_attention"],
                     "kl_noncancer_raw": new_raw,
                     "kl_noncancer_weighted": new_weighted}
            for name, value in parts.items():
                totals[name] += float(value.detach()) * batch_size
            seen += batch_size
        if updates != expected_updates or seen != len(train):
            raise RuntimeError(f"{arm}第{epoch}轮更新次数或抽样数不符")
        history.append({"epoch": epoch, "updates": updates,
                        **{name: value / seen for name, value in totals.items()}})
        print(json.dumps({"arm": arm, **history[-1]}, ensure_ascii=False), flush=True)
    for name, value in model.features[:5].state_dict().items():
        if not torch.equal(value, frozen_before[name]):
            raise RuntimeError(f"冻结块参数或BN状态改变: {name}")
    predictions, existing = evaluate(model, val_set, val_loader, device,
                                     lesion_fraction_tercile_bounds(train))
    patients, classification = fixed_threshold_metrics(predictions)
    metrics = {"classification": classification,
               "spatial": spatial_metrics(predictions, val),
               "existing_cancer_spatial": existing["spatial"],
               "history": history, "updates": sum(row["updates"] for row in history),
               "frozen_backbone_state_unchanged": True}
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "val_images_internal.csv", index=False)
    patients.to_csv(output / "val_patients_internal.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    torch.save({"model_state_dict": model.state_dict(), "arm": arm,
                "start_checkpoint_sha256": START_SHA, "metrics": metrics},
               output / "final_student.pth")
    return patients, metrics


def summarize(control: pd.DataFrame, candidate: pd.DataFrame,
              control_metrics: dict, candidate_metrics: dict) -> dict:
    """按预定互斥规则汇总空间、分类保护和患者错误转换。

    Args:
        control (pd.DataFrame): 同等微调对照的患者均值概率。
        candidate (pd.DataFrame): 加非癌框臂的患者均值概率。
        control_metrics (dict): 对照固定阈值及空间指标。
        candidate_metrics (dict): 候选固定阈值及空间指标。
    Returns:
        dict: 配对差、转换人数和唯一判定类别。
    """
    joined = control.merge(candidate, on=["patient_id", "label"],
                           suffixes=("_control", "_candidate"), validate="one_to_one")
    correct_a = joined.probability_control.ge(THRESHOLD).astype(int).eq(joined.label)
    correct_b = joined.probability_candidate.ge(THRESHOLD).astype(int).eq(joined.label)
    cm_a, cm_b = control_metrics["classification"], candidate_metrics["classification"]
    s_a, s_b = control_metrics["spatial"], candidate_metrics["spatial"]
    for label in ("noncancer", "cancer"):
        for count in ("images", "patients", "normalized_aib_evaluable_images",
                      "normalized_aib_evaluable_patients", "excluded_area_ge_0_99"):
            if s_a[label][count] != s_b[label][count]:
                raise RuntimeError(f"两臂{label}空间评价有效集合不一致: {count}")
    def spatial_delta(label: str, metric: str) -> float:
        return (s_b[label]["patient_equal_mean"][metric]
                - s_a[label]["patient_equal_mean"][metric])
    conditions = {
        "noncancer_normalized_aib_gain_ge_0_05": spatial_delta("noncancer", "normalized_aib") >= .05,
        "noncancer_pga_gain_ge_0_05": spatial_delta("noncancer", "pga") >= .05,
        "cancer_normalized_aib_drop_le_0_05": spatial_delta("cancer", "normalized_aib") >= -.05,
        "cancer_pga_drop_le_0_05": spatial_delta("cancer", "pga") >= -.05,
        "patient_auc_drop_le_0_01": cm_b["patient_auc"] >= cm_a["patient_auc"] - .01,
        "cancer_fn_not_higher": cm_b["fn"] <= cm_a["fn"],
        "noncancer_fp_increase_le_6": cm_b["fp"] <= cm_a["fp"] + 6,
    }
    noncancer_gain = bool(all(list(conditions.values())[:2]))
    protected = bool(all(list(conditions.values())[2:]))
    status = ("further_research_value" if noncancer_gain and protected else
              "localization_gain_with_tradeoff" if noncancer_gain else
              "prespecified_noncancer_spatial_goal_not_met")
    return {"status": status, "development_criteria_not_clinical_standard": True,
            "noncancer_gain_both": noncancer_gain, "all_protections_met": protected,
            "conditions": conditions,
            "spatial_patient_equal_delta": {
                label: {metric: spatial_delta(label, metric)
                        for metric in ("normalized_aib", "pga", "aib", "excess_aib")}
                for label in ("noncancer", "cancer")},
            "control_correct_candidate_wrong": int((correct_a & ~correct_b).sum()),
            "control_wrong_candidate_correct": int((~correct_a & correct_b).sum()),
            "by_label_conversion": {
                str(label): {
                    "control_correct_candidate_wrong": int((correct_a & ~correct_b & joined.label.eq(label)).sum()),
                    "control_wrong_candidate_correct": int((~correct_a & correct_b & joined.label.eq(label)).sum()),
                } for label in (0, 1)
            },
            "both_correct": int((correct_a & correct_b).sum()),
            "both_wrong": int((~correct_a & ~correct_b).sum()),
            "control": cm_a, "candidate": cm_b}


def main() -> None:
    """只读核对输入，运行两臂同起点固定微调并输出配对结果。

    Args:
        None: 输出目录、设备和debug通过CLI提供。
    Returns:
        None: 在独立目录保存两臂权重、预测、指标和判定。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_sha256(START) != START_SHA:
        raise RuntimeError("C-long共同起点SHA不一致")
    seed_everything(SEED)
    manifest_sha = file_sha256(DEFAULT_MANIFEST)
    frame, _ = load_manifest(DEFAULT_MANIFEST, DEFAULT_V3_AUDIT, False, 3, SEED)
    cache = load_teacher_cache(DEFAULT_TEACHER_CACHE, frame, manifest_sha, False)
    frame = attach_noncancer_boxes(frame)
    if args.debug:
        selected = []
        for split, part in frame.groupby("split", sort=False):
            patients = part[["patient_id", "label"]].drop_duplicates()
            selected_ids = pd.concat([group.head(2) for _, group in patients.groupby("label")])
            selected.append(part[part.patient_id.isin(selected_ids.patient_id)])
        frame = pd.concat(selected, ignore_index=True)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    val = frame[frame.split.eq("val")].reset_index(drop=True)
    paired_input_check = verify_paired_inputs(train)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    start = torch.load(START, map_location="cpu", weights_only=False)
    start_state = start["model_state_dict"]
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps({
        "start_checkpoint": str(START), "start_sha256": START_SHA,
        "manifest_sha256": manifest_sha, "teacher_cache": str(DEFAULT_TEACHER_CACHE),
        "teacher_cache_sha256": file_sha256(DEFAULT_TEACHER_CACHE),
        "noncancer_boxes": str(BOX_ROOT), "train_images": len(train),
        "val_images": len(val), "seed": SEED, "epochs": 1 if args.debug else EPOCHS,
        "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "alpha": ALPHA, "tau": TAU, "beta_cancer": BETA,
        "lambda_noncancer": BETA,
        "same_beta_does_not_imply_same_constraint_strength": True,
        "original_cancer_has_target_unchanged": True,
        "paired_input_check": paired_input_check,
        "fixed_patient_threshold": THRESHOLD,
        "primary_question": "specificity_at_patient_sensitivity_at_least_0p90",
        "classification_comparators": ["unchanged_original_clong", "equal_budget_control"],
        "spatial_status_role": "secondary_localization_result_not_classification_success",
        "debug": args.debug, "test_read": False, "external_read": False,
    }, ensure_ascii=False, indent=2))
    workers = 0 if args.debug else 4
    results = {}
    for arm in ("control", "noncancer_box"):
        results[arm] = run_arm(arm, start_state, train, val, cache, device, workers,
                               args.output / arm, args.debug)
    summary = summarize(results["control"][0], results["noncancer_box"][0],
                        results["control"][1], results["noncancer_box"][1])
    summary["spatial_status"] = summary.pop("status")
    summary["status"] = "classification_comparison_completed_no_automatic_model_replacement"
    original = pd.read_csv(START.parent / "val_patient_predictions.csv")
    original = original[original.patient_id.isin(results["control"][0].patient_id)].copy()
    summary["classification_comparisons"] = {
        "candidate_vs_control": classification_comparison(results["control"][0], results["noncancer_box"][0]),
        "candidate_vs_original": classification_comparison(original, results["noncancer_box"][0]),
        "control_vs_original": classification_comparison(original, results["control"][0]),
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
