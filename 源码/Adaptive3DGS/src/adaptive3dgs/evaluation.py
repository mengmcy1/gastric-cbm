"""Provider-independent dense geometry and confidence evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .validation import ValidationError


@dataclass(frozen=True, slots=True)
class GeometryTarget:
    depth_z_float32: npt.NDArray[np.float32]
    occlusion_hidden_mask: npt.NDArray[np.bool_]
    outside_source_fov_mask: npt.NDArray[np.bool_]
    observed_from_source_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class GeometryPrediction:
    depth_z_float32: npt.NDArray[np.float32]
    support_probability_float32: npt.NDArray[np.float32]
    geometry_confidence_float32: npt.NDArray[np.float32]


def _check_target_prediction(target: GeometryTarget, prediction: GeometryPrediction) -> tuple[int, int]:
    shape = target.depth_z_float32.shape
    if len(shape) != 2 or any(value.shape != shape for value in (
        target.occlusion_hidden_mask,
        target.outside_source_fov_mask,
        target.observed_from_source_mask,
        prediction.depth_z_float32,
        prediction.support_probability_float32,
        prediction.geometry_confidence_float32,
    )):
        raise ValidationError("geometry target and prediction arrays must share one HxW shape")
    if target.depth_z_float32.dtype != np.float32 or prediction.depth_z_float32.dtype != np.float32:
        raise ValidationError("geometry depths must be float32")
    if prediction.support_probability_float32.dtype != np.float32:
        raise ValidationError("support probability must be float32")
    if prediction.geometry_confidence_float32.dtype != np.float32:
        raise ValidationError("geometry confidence must be float32")
    for name, value in (
        ("support probability", prediction.support_probability_float32),
        ("geometry confidence", prediction.geometry_confidence_float32),
    ):
        if not np.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
            raise ValidationError(f"{name} must be finite and inside [0,1]")
    return shape


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> tuple[float, list[dict[str, float | int | None]]]:
    table: list[dict[str, float | int | None]] = []
    count = len(confidence)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= low) & (confidence <= high if index == bins - 1 else confidence < high)
        selected_count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if selected_count else None
        accuracy = float(correct[selected].mean()) if selected_count else None
        if selected_count:
            ece += selected_count / count * abs(float(mean_confidence) - float(accuracy))
        table.append(
            {
                "low": float(low),
                "high": float(high),
                "pixels": selected_count,
                "mean_confidence": mean_confidence,
                "depth_within_10_percent": accuracy,
            }
        )
    return float(ece), table


def _selective_aurc(confidence: np.ndarray, risk: np.ndarray) -> float:
    if len(confidence) == 0:
        return float("nan")
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)
    return float(cumulative_risk.mean())


def evaluate_occlusion_geometry(
    target: GeometryTarget,
    prediction: GeometryPrediction,
    *,
    support_threshold: float = 0.05,
    ece_bins: int = 10,
) -> dict[str, object]:
    _check_target_prediction(target, prediction)
    if not 0 < support_threshold < 1:
        raise ValueError("support_threshold must be in (0,1)")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")
    truth_valid = np.isfinite(target.depth_z_float32) & (target.depth_z_float32 > 0)
    predicted_valid = np.isfinite(prediction.depth_z_float32) & (prediction.depth_z_float32 > 0)
    hidden = target.occlusion_hidden_mask & truth_valid
    supported = hidden & predicted_valid & (prediction.support_probability_float32 >= support_threshold)
    hidden_pixels = int(hidden.sum())
    supported_pixels = int(supported.sum())
    result: dict[str, object] = {
        "support_threshold": support_threshold,
        "hidden_truth_pixels": hidden_pixels,
        "supported_hidden_pixels": supported_pixels,
        "hidden_coverage": float(supported_pixels / hidden_pixels) if hidden_pixels else None,
        "outside_source_fov_truth_pixels": int((target.outside_source_fov_mask & truth_valid).sum()),
        "outside_source_fov_evaluation": "unsupported_excluded",
    }
    if supported_pixels == 0:
        result["geometry"] = None
        result["confidence"] = None
        return result
    truth = target.depth_z_float32[supported].astype(np.float64)
    predicted = prediction.depth_z_float32[supported].astype(np.float64)
    confidence = prediction.geometry_confidence_float32[supported].astype(np.float64)
    abs_relative = np.abs(predicted - truth) / truth
    ratio = np.maximum(predicted / truth, truth / predicted)
    correct_10 = ratio < 1.10
    ece, calibration = _ece(confidence, correct_10, ece_bins)
    result["geometry"] = {
        "abs_rel": float(abs_relative.mean()),
        "mae": float(np.abs(predicted - truth).mean()),
        "rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
        "delta_1_05": float(np.mean(ratio < 1.05)),
        "delta_1_10": float(correct_10.mean()),
        "predicted_too_near_by_more_than_5_percent": float(np.mean(predicted < 0.95 * truth)),
        "predicted_too_far_by_more_than_5_percent": float(np.mean(predicted > 1.05 * truth)),
    }
    result["confidence"] = {
        "correctness_definition": "max(pred/truth,truth/pred)<1.10",
        "ece": ece,
        "selective_aurc_abs_rel": _selective_aurc(confidence, abs_relative),
        "bins": calibration,
    }
    return result
