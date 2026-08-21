#!/usr/bin/env python3
"""RP-A eligible Feature 聚合审计与活跃频率校准。

本入口只读 S2c seed42 正式字典和冻结 train 空间特征。第一阶段将
K=1024 的 patch 激活按 image/patient 汇总，产出可复用的患者矩阵与逐
Feature 审计表；第二阶段仅用这些 train-only 产物执行 400 次
label×source 分层的确定性 split-half 校准。

test/internal test/external 不读取；本脚本不训练 SAE，也不启动新 seed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "程序/MAGE/正式代码"))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from clong_s2b_core import ACTIVE_EPS  # noqa: E402
from clong_s2c_core import K_LIST, MatryoshkaSparseAutoencoder  # noqa: E402

PROTOCOL_PATH = SCRIPT_DIR / "rpa_eligible_protocol_v1.json"
S2C_ROOT = PROJECT_ROOT / "结果/SAE/CLong_S2c_Matryoshka_20260821"
S2C_RUN = S2C_ROOT / "s2c_clong_seed42"
S2C_CHECKPOINT = S2C_RUN / "sae_best.pth"
S2C_CONFIG = S2C_RUN / "config.json"
SPATIAL_CACHE = PROJECT_ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Eligible校准_20260821"


def parse_args() -> argparse.Namespace:
    """解析执行阶段、设备和 debug 选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "calibrate", "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-images", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """读取 JSON 对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    """以标准 JSON 写入可复算产物。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha_text(*parts: object) -> str:
    """以管道分隔后计算 SHA256，用于确定性患者分半。"""
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def protocol_sha256() -> str:
    """返回静态协议 JSON 的文件 SHA。"""
    return file_sha256(PROTOCOL_PATH)


def validate_protocol(protocol: dict) -> None:
    """校验协议中不得漂移的核心定义。"""
    if not math.isclose(float(protocol["active_eps"]), ACTIVE_EPS, abs_tol=0.0):
        raise RuntimeError("protocol ACTIVE_EPS 与 S2b/S2c 实现不一致")
    if int(protocol["representation_k"]) != max(K_LIST):
        raise RuntimeError("RP-A 解释视图必须是 S2c K=1024")
    if protocol["quantile_method"] != "lower" or int(protocol["split_count"]) != 400:
        raise RuntimeError("分半次数或下分位数实现偏离拟冻结协议")


def validate_lineage() -> dict:
    """核验 S2c checkpoint/config 与冻结空间缓存的血缘。"""
    config = load_json(S2C_CONFIG)
    cache_config_path = SPATIAL_CACHE / "cache_config.json"
    cache_config = load_json(cache_config_path)
    checkpoint_sha = file_sha256(S2C_CHECKPOINT)
    cache_config_sha = file_sha256(cache_config_path)
    if config.get("stage") != "S2c" or config.get("seed") != 42 or config.get("debug"):
        raise RuntimeError("S2c config 不是 seed42 正式产物")
    if config.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("S2c checkpoint SHA 与 config 不一致")
    if config.get("cache_config_sha256") != cache_config_sha:
        raise RuntimeError("S2c config 未绑定当前空间缓存")
    if cache_config["counts"]["train"] != 2350:
        raise RuntimeError("冻结 train 缓存应为2350图")
    for name in ("train_spatial_features.npy", "train_metadata.csv"):
        if file_sha256(SPATIAL_CACHE / name) != cache_config["files"][name]:
            raise RuntimeError(f"空间缓存文件 SHA 不一致: {name}")
    return {
        "s2c_config": {"path": str(S2C_CONFIG), "sha256": file_sha256(S2C_CONFIG)},
        "s2c_checkpoint": {"path": str(S2C_CHECKPOINT), "sha256": checkpoint_sha},
        "spatial_cache_config": {"path": str(cache_config_path), "sha256": cache_config_sha},
        "train_spatial_features": {
            "path": str(SPATIAL_CACHE / "train_spatial_features.npy"),
            "sha256": cache_config["files"]["train_spatial_features.npy"],
        },
        "train_metadata": {
            "path": str(SPATIAL_CACHE / "train_metadata.csv"),
            "sha256": cache_config["files"]["train_metadata.csv"],
        },
    }


def load_sae(device: torch.device) -> MatryoshkaSparseAutoencoder:
    """从 seed42 正式 checkpoint 恢复 Matryoshka SAE。"""
    payload = torch.load(S2C_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["sae_state_dict"]
    sae = MatryoshkaSparseAutoencoder(
        input_dim=1280,
        hidden_dim=10240,
        feature_center=state["decoder_bias"],
        k_list=K_LIST,
    )
    sae.load_state_dict(state)
    sae.to(device).eval()
    return sae


def patient_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """生成一行一患者的冻结顺序表，并校验标签/来源唯一。"""
    counts = metadata.groupby("patient_id").agg(
        label_n=("label", "nunique"), source_n=("source", "nunique")
    )
    if (counts[["label_n", "source_n"]] != 1).any().any():
        raise RuntimeError("同一患者的 label 或 source 不唯一")
    return (
        metadata.groupby("patient_id", sort=True)
        .agg(label=("label", "first"), source=("source", "first"), image_count=("patient_id", "size"))
        .reset_index()
    )


def aggregate_train(
    spatial: np.ndarray,
    metadata: pd.DataFrame,
    sae: MatryoshkaSparseAutoencoder,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """K=1024 逐图编码并按患者汇总 presence/rank/mass/frequency。

    返回患者表和四个 [N_patient, 10240] 矩阵。ranking 只保留秩次
    用途，mass/frequency 按患者内图像均值聚合。
    """
    patients = patient_table(metadata)
    index = {pid: i for i, pid in enumerate(patients["patient_id"])}
    n_patients, hidden_dim = len(patients), sae.hidden_dim
    presence = np.zeros((n_patients, hidden_dim), dtype=bool)
    ranking = np.zeros((n_patients, hidden_dim), dtype=np.float32)
    mass_sum = np.zeros((n_patients, hidden_dim), dtype=np.float64)
    frequency_sum = np.zeros((n_patients, hidden_dim), dtype=np.float64)
    seen_images = np.zeros(n_patients, dtype=np.int32)

    for start in range(0, len(metadata), batch_size):
        stop = min(start + batch_size, len(metadata))
        features = torch.from_numpy(np.array(spatial[start:stop], copy=True)).to(device)
        with torch.no_grad():
            hidden = sae.encode(features, k=1024).cpu().numpy()
        active = hidden > ACTIVE_EPS
        image_presence = active.any(axis=1)
        image_ranking = hidden.max(axis=1)
        image_mass = hidden.mean(axis=1)
        image_frequency = active.mean(axis=1)
        for local, row in enumerate(metadata.iloc[start:stop].itertuples(index=False)):
            p = index[row.patient_id]
            presence[p] |= image_presence[local]
            ranking[p] = np.maximum(ranking[p], image_ranking[local])
            mass_sum[p] += image_mass[local]
            frequency_sum[p] += image_frequency[local]
            seen_images[p] += 1
        print(f"\r聚合进度: {stop}/{len(metadata)}图", end="", flush=True)
    print()
    if not np.array_equal(seen_images, patients["image_count"].to_numpy()):
        raise RuntimeError("患者图像计数与元数据不一致")
    divisor = seen_images[:, None]
    return patients, {
        "presence": presence,
        "ranking": ranking,
        "mass": (mass_sum / divisor).astype(np.float32),
        "active_frequency": (frequency_sum / divisor).astype(np.float32),
    }


def pearson_columns(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """向量化计算每个 Feature 与患者图数的 Pearson 相关。"""
    x = values.astype(np.float64)
    y = target.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean()
    denominator = np.sqrt((x * x).sum(axis=0) * (y * y).sum())
    result = np.zeros(x.shape[1], dtype=np.float64)
    valid = denominator > 0
    result[valid] = (x[:, valid] * y[:, None]).sum(axis=0) / denominator[valid]
    return result


def select_top_q(coverage: np.ndarray, candidates: list[float]) -> float:
    """选出不会向任一非死亡 Feature 的 Top 集合混入零激活的最大 q。"""
    passing = [float(q) for q in candidates if np.all(coverage + 1e-15 >= q)]
    if not passing:
        raise RuntimeError("无 Top-q 候选能保证所有非死亡 Feature 的 Top 患者非零")
    return max(passing)


def build_feature_audit(
    patients: pd.DataFrame, matrices: dict[str, np.ndarray], protocol: dict
) -> tuple[pd.DataFrame, dict]:
    """从完整 train 患者矩阵构造逐 Feature 审计表与 q/P_min 摘要。"""
    presence = matrices["presence"]
    nondead = presence.any(axis=0)
    positive_count = presence.sum(axis=0)
    coverage = positive_count / len(patients)
    q = select_top_q(coverage[nondead], protocol["top_q_candidates"])
    p_min = int(math.ceil(q * len(patients)))
    ranking = matrices["ranking"]
    frequency = matrices["active_frequency"]
    correlations = pearson_columns(ranking, patients["image_count"].to_numpy())
    rank_q = np.quantile(ranking, (0.50, 0.75, 0.90, 0.95, 0.99), axis=0)
    mass = matrices["mass"]
    mass_q = np.quantile(mass, (0.50, 0.90, 0.99), axis=0)
    table = pd.DataFrame({
        "feature_id": np.arange(presence.shape[1]),
        "full_train_nondead": nondead,
        "positive_patient_count": positive_count,
        "patient_coverage": coverage,
        "active_position_frequency": frequency.mean(axis=0),
        "ranking_q50": rank_q[0],
        "ranking_q75": rank_q[1],
        "ranking_q90": rank_q[2],
        "ranking_q95": rank_q[3],
        "ranking_q99": rank_q[4],
        "ranking_max": ranking.max(axis=0),
        "ranking_image_count_pearson": correlations,
        "activation_mass_total": mass.sum(axis=0),
        "mass_q50": mass_q[0],
        "mass_q90": mass_q[1],
        "mass_q99": mass_q[2],
    })
    for candidate in protocol["top_q_candidates"]:
        top_n = int(math.ceil(float(candidate) * len(patients)))
        suffix = f"q{int(round(100 * float(candidate))):02d}"
        table[f"top_{suffix}_patient_count"] = top_n
        table[f"top_{suffix}_nonzero_count"] = np.minimum(positive_count, top_n)
    safe_q90 = table["ranking_q90"].to_numpy()
    ratio = np.full(len(table), np.nan)
    valid = safe_q90 > ACTIVE_EPS
    ratio[valid] = table.loc[valid, "ranking_max"] / safe_q90[valid]
    table["ranking_max_over_q90"] = ratio
    active = table.loc[nondead]
    summary = {
        "patient_count": int(len(patients)),
        "image_count": int(patients["image_count"].sum()),
        "full_train_nondead_feature_count": int(nondead.sum()),
        "coverage_quantiles": quantiles(active["patient_coverage"].to_numpy()),
        "active_frequency_quantiles": quantiles(active["active_position_frequency"].to_numpy()),
        "feature_fraction_above_patient_coverage": {
            str(threshold): float((active["patient_coverage"] > threshold).mean())
            for threshold in (0.25, 0.50, 0.75, 0.90)
        },
        "ranking_max_over_q90_quantiles": quantiles(active["ranking_max_over_q90"].dropna().to_numpy()),
        "ranking_image_count_pearson_quantiles": quantiles(active["ranking_image_count_pearson"].to_numpy()),
        "selected_top_q": q,
        "p_min_train": p_min,
        "val_patient_count_frozen": 260,
        "val_top_q_count": int(math.ceil(q * 260)),
        "ranking_semantics": "ordinal_only_not_cross_feature_amplitude",
    }
    return table, summary


def quantiles(values: np.ndarray) -> dict[str, float]:
    """返回审计中统一使用的分位数。"""
    return {
        f"q{int(q * 100):02d}": float(np.quantile(values, q))
        for q in (0.01, 0.05, 0.50, 0.95, 0.99)
    }


def split_patients(
    patients: pd.DataFrame, protocol_sha: str, split_seed: int, protocol: dict
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """按 label×source 分层确定性分半，独立 offset 决定奇数额外成员方向。"""
    half = np.full(len(patients), -1, dtype=np.int8)
    audit: list[dict] = []
    grouped = patients.groupby(protocol["split_strata"], sort=True, dropna=False)
    for stratum_values, group in grouped:
        values = stratum_values if isinstance(stratum_values, tuple) else (stratum_values,)
        stratum_id = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
        ordered = sorted(
            group.index,
            key=lambda idx: sha_text(
                protocol["patient_order_domain"], protocol_sha, split_seed,
                stratum_id, patients.at[idx, "patient_id"],
            ),
        )
        offset = int(sha_text(
            protocol["stratum_offset_domain"], protocol_sha, split_seed, stratum_id
        ), 16) % 2
        for rank, idx in enumerate(ordered):
            half[idx] = (rank + offset) % 2
        n_a = int((half[group.index] == 0).sum())
        n_b = int((half[group.index] == 1).sum())
        if abs(n_a - n_b) > 1:
            raise RuntimeError("分层分半不满足 |n_A-n_B|<=1")
        audit.append({"stratum_id": stratum_id, "n": len(group), "offset": offset, "n_a": n_a, "n_b": n_b})
    if (half < 0).any():
        raise RuntimeError("存在未分配患者")
    return np.flatnonzero(half == 0), np.flatnonzero(half == 1), audit


def spearman_from_vectors(x: np.ndarray, y: np.ndarray) -> float | None:
    """计算并列值平均秩的 Spearman，常量向量返回 None。"""
    if len(x) < 2:
        return None
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_candidate(
    presence: np.ndarray, frequency: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray,
    nondead: np.ndarray, g: float, patient_fraction: float,
) -> dict:
    """在一次 split 中计算某 g 的 eligible Jaccard/Spearman/最小集合数。"""
    def eligible(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        min_positive = int(math.ceil(patient_fraction * len(indices)))
        counts = presence[indices].sum(axis=0)
        freq = frequency[indices].mean(axis=0)
        mask = nondead & (counts >= min_positive) & (freq >= g)
        return mask, freq

    mask_a, freq_a = eligible(idx_a)
    mask_b, freq_b = eligible(idx_b)
    union = mask_a | mask_b
    intersection = mask_a & mask_b
    if not union.any():
        return {"computable": False, "reason": "empty_union"}
    rho = spearman_from_vectors(freq_a[union], freq_b[union])
    if rho is None:
        return {"computable": False, "reason": "undefined_spearman"}
    return {
        "computable": True,
        "jaccard": float(intersection.sum() / union.sum()),
        "spearman": rho,
        "min_eligible_count": int(min(mask_a.sum(), mask_b.sum())),
        "n_a": int(len(idx_a)),
        "n_b": int(len(idx_b)),
    }


def lower_quantile(values: list[float], q: float) -> float:
    """使用 NumPy lower 定义返回非插值经验下分位数。"""
    return float(np.quantile(np.asarray(values), q, method="lower"))


def calibrate(
    patients: pd.DataFrame, presence: np.ndarray, frequency: np.ndarray,
    nondead: np.ndarray, patient_fraction: float, protocol: dict, protocol_sha: str,
) -> tuple[dict, pd.DataFrame]:
    """执行400次确定性分半，返回最小合格 A_min 和全部原始折记录。"""
    rows: list[dict] = []
    candidates = [float(g) for g in protocol["active_frequency_candidates"]]
    for b in range(int(protocol["split_count"])):
        split_seed = int(protocol["base_split_seed"]) + b
        idx_a, idx_b, strata = split_patients(patients, protocol_sha, split_seed, protocol)
        for g in candidates:
            result = evaluate_candidate(
                presence, frequency, idx_a, idx_b, nondead, g,
                patient_fraction,
            )
            rows.append({
                "split_index": b,
                "split_seed": split_seed,
                "g": g,
                "n_half_a": len(idx_a),
                "n_half_b": len(idx_b),
                "strata_sha256": sha_text(json.dumps(strata, sort_keys=True, ensure_ascii=False)),
                **result,
            })
    frame = pd.DataFrame(rows)
    summaries = []
    passing = []
    for g in candidates:
        part = frame[frame["g"] == g]
        computable = bool(part["computable"].all())
        item = {"g": g, "all_400_computable": computable}
        if computable:
            item.update({
                "jaccard_q05_lower": lower_quantile(part["jaccard"].tolist(), protocol["quantile_q"]),
                "spearman_q05_lower": lower_quantile(part["spearman"].tolist(), protocol["quantile_q"]),
                "min_eligible_q05_lower": int(lower_quantile(part["min_eligible_count"].tolist(), protocol["quantile_q"])),
            })
            item["passed"] = bool(
                item["jaccard_q05_lower"] >= protocol["min_jaccard"]
                and item["spearman_q05_lower"] >= protocol["min_spearman"]
                and item["min_eligible_q05_lower"] >= protocol["min_eligible_features"]
            )
        else:
            item["passed"] = False
        summaries.append(item)
        if item["passed"]:
            passing.append(g)
    selected = min(passing) if passing else None
    return {
        "status": "active_frequency_calibrated" if selected is not None else "active_frequency_calibration_infeasible",
        "selected_a_min": selected,
        "candidate_summaries": summaries,
        "split_count": int(protocol["split_count"]),
        "quantile_impl": "numpy.quantile",
        "quantile_method": protocol["quantile_method"],
        "numpy_version": np.__version__,
    }, frame


def save_matrices(output: Path, patients: pd.DataFrame, matrices: dict[str, np.ndarray]) -> dict:
    """保存患者表与四类聚合矩阵，返回文件 SHA。"""
    paths = {"patients": output / "seed42_train_patients.csv"}
    patients.to_csv(paths["patients"], index=False)
    for name, values in matrices.items():
        paths[name] = output / f"seed42_patient_{name}.npy"
        np.save(paths[name], values)
    return {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}


def load_saved_matrices(output: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """从正式审计缓存恢复患者表和聚合矩阵。"""
    patients = pd.read_csv(output / "seed42_train_patients.csv")
    matrices = {
        name: np.load(output / f"seed42_patient_{name}.npy", mmap_mode="r")
        for name in ("presence", "ranking", "mass", "active_frequency")
    }
    return patients, matrices


def write_sha_manifest(output: Path, names: list[str]) -> None:
    """生成非自指的 SHA256SUMS 侧车文件。"""
    lines = [f"{file_sha256(output / name)}  {name}" for name in names]
    (output / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """组织血缘校验、审计产物化和 split-half 校准。"""
    args = parse_args()
    protocol = load_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    lineage = validate_lineage()
    output = args.output_root
    if args.debug:
        output = OUTPUT_ROOT / "debug" / f"n{args.debug_images}"
    output.mkdir(parents=True, exist_ok=True)
    audit_summary_path = output / "seed42_feature_aggregation_audit_summary.json"
    calibration_path = output / "active_frequency_calibration.json"

    if args.stage in ("audit", "all"):
        if audit_summary_path.exists():
            raise FileExistsError(f"审计产物已存在，禁止覆盖: {audit_summary_path}")
        if not args.debug and args.device != "cuda":
            raise ValueError("正式聚合审计必须显式使用 --device cuda")
        metadata = pd.read_csv(SPATIAL_CACHE / "train_metadata.csv")
        spatial = np.load(SPATIAL_CACHE / "train_spatial_features.npy", mmap_mode="r")
        if args.debug:
            metadata = metadata.iloc[: args.debug_images].reset_index(drop=True)
            spatial = spatial[: args.debug_images]
        device = torch.device(args.device)
        sae = load_sae(device)
        patients, matrices = aggregate_train(spatial, metadata, sae, device, args.batch_size)
        matrix_files = save_matrices(output, patients, matrices)
        feature_table, summary = build_feature_audit(patients, matrices, protocol)
        feature_csv = output / "seed42_feature_aggregation_audit.csv"
        feature_table.to_csv(feature_csv, index=False)
        summary.update({
            "stage": "RP-A eligible aggregation audit",
            "debug": bool(args.debug),
            "protocol": {"path": str(PROTOCOL_PATH), "sha256": protocol_sha256()},
            "code_sha256": file_sha256(Path(__file__)),
            "lineage": lineage,
            "matrix_files": matrix_files,
            "feature_csv_sha256": file_sha256(feature_csv),
        })
        write_json(audit_summary_path, summary)
        print(f"聚合审计完成: q={summary['selected_top_q']}, P_min={summary['p_min_train']}")

    if args.stage in ("calibrate", "all"):
        if calibration_path.exists():
            raise FileExistsError(f"校准产物已存在，禁止覆盖: {calibration_path}")
        if not audit_summary_path.exists():
            raise FileNotFoundError("缺少正式聚合审计摘要，不得直接校准")
        audit_summary = load_json(audit_summary_path)
        if audit_summary["protocol"]["sha256"] != protocol_sha256():
            raise RuntimeError("审计产物绑定的 protocol SHA 已变化")
        if audit_summary["code_sha256"] != file_sha256(Path(__file__)):
            raise RuntimeError("审计后代码 SHA 已变化，禁止继续校准")
        if audit_summary["feature_csv_sha256"] != file_sha256(
            output / "seed42_feature_aggregation_audit.csv"
        ):
            raise RuntimeError("逐 Feature 审计 CSV SHA 不一致")
        for item in audit_summary["matrix_files"].values():
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"患者聚合缓存 SHA 不一致: {item['path']}")
        patients, matrices = load_saved_matrices(output)
        feature_table = pd.read_csv(output / "seed42_feature_aggregation_audit.csv")
        nondead = feature_table["full_train_nondead"].astype(bool).to_numpy()
        calibration, split_frame = calibrate(
            patients,
            np.asarray(matrices["presence"]),
            np.asarray(matrices["active_frequency"]),
            nondead,
            float(audit_summary["selected_top_q"]),
            protocol,
            protocol_sha256(),
        )
        split_csv = output / "active_frequency_split_results.csv"
        split_frame.to_csv(split_csv, index=False)
        calibration.update({
            "debug": bool(args.debug),
            "protocol": {"path": str(PROTOCOL_PATH), "sha256": protocol_sha256()},
            "code_sha256": file_sha256(Path(__file__)),
            "audit_summary_sha256": file_sha256(audit_summary_path),
            "feature_audit_csv_sha256": file_sha256(output / "seed42_feature_aggregation_audit.csv"),
            "patient_presence_sha256": file_sha256(output / "seed42_patient_presence.npy"),
            "patient_active_frequency_sha256": file_sha256(output / "seed42_patient_active_frequency.npy"),
            "split_results_csv_sha256": file_sha256(split_csv),
            "patient_presence_fraction_from_audit_q": float(audit_summary["selected_top_q"]),
            "p_min_train_from_audit": int(audit_summary["p_min_train"]),
            "formal_freeze_ready": bool(not args.debug and calibration["selected_a_min"] is not None),
        })
        write_json(calibration_path, calibration)
        print(f"活跃频率校准完成: status={calibration['status']}, A_min={calibration['selected_a_min']}")

    names = sorted(p.name for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    write_sha_manifest(output, names)
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
