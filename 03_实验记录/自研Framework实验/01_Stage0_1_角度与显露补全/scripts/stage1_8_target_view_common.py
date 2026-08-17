"""Shared leakage-safe Stage 1.8 target-view data preparation and metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive3dgs.datasets import load_hypersim_training_triplets
from adaptive3dgs.target_view import (
    forward_splat_source_to_target,
    scale_intrinsics,
    target_camera_conditioning_in_source,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _resize(array: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    value = torch.from_numpy(array)
    if array.ndim == 2:
        value = value[None, None].float()
    elif array.ndim == 3:
        value = value.permute(2, 0, 1)[None].float()
    else:
        raise ValueError("resize supports HxW or HxWxC")
    kwargs = {"size": size, "mode": mode}
    if mode == "bilinear":
        kwargs.update({"align_corners": False, "antialias": True})
    result = F.interpolate(value, **kwargs)[0]
    return result[0].numpy() if array.ndim == 2 else result.permute(1, 2, 0).numpy()


@dataclass
class PreparedTargetSample:
    split: str
    scene: str
    source_frame: int
    side: str
    target_id: str
    target_frame: int
    warped_rgb: torch.Tensor
    warped_depth: torch.Tensor
    warp_valid: torch.Tensor
    rays_in_source: torch.Tensor
    origin_in_source: torch.Tensor
    source_depth_scale: torch.Tensor
    target_rgb: torch.Tensor
    target_depth: torch.Tensor
    observed: torch.Tensor
    occluded: torch.Tensor
    outside: torch.Tensor
    input_hashes: dict[str, str]

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.warped_rgb,
            self.warped_depth,
            self.warp_valid,
            self.rays_in_source,
            self.origin_in_source,
            self.source_depth_scale,
        )


def prepare_target_samples(
    triplet_config: Path,
    dataset_root: Path,
    camera_parameters: Path,
    visibility_root: Path,
    base_record_path: Path,
    size: tuple[int, int],
    device: torch.device,
    *,
    selected: set[tuple[str, str]] | None = None,
) -> list[PreparedTargetSample]:
    base_record = json.loads(base_record_path.read_text(encoding="utf-8"))
    base_by_key = {(item["scene"], int(item["source_frame"])): item for item in base_record["results"]}
    triplets = load_hypersim_training_triplets(triplet_config, dataset_root, camera_parameters)
    prepared: list[PreparedTargetSample] = []
    target_height, target_width = size
    for triplet in triplets:
        key = (triplet.scene, int(triplet.source_frame_index))
        if key not in base_by_key:
            raise RuntimeError(f"frozen BaseDepth missing {key}")
        base_item = base_by_key[key]
        base_path = Path(base_item["arrays"])
        if sha256(base_path) != base_item["arrays_sha256"]:
            raise RuntimeError(f"BaseDepth hash mismatch: {base_path}")
        with np.load(base_path, allow_pickle=False) as values:
            base = values["base_depth_z_float32"].astype(np.float32)
        source_height, source_width = base.shape
        source_rgb = _resize(triplet.source_rgb_uint8.astype(np.float32) / 255.0, size, "bilinear").astype(np.float32)
        base = _resize(base, size, "bilinear").astype(np.float32)
        source_k = scale_intrinsics(
            triplet.source_camera.intrinsics_3x3_float64,
            (source_width, source_height),
            (target_width, target_height),
        )
        for side, observation in zip(("left", "right"), triplet.target_observations):
            if selected is not None and (triplet.scene, side) not in selected:
                continue
            target_frame = int(observation.observation_id.rsplit("-", 1)[1])
            target_k = scale_intrinsics(
                observation.intrinsics_3x3_float64,
                (source_width, source_height),
                (target_width, target_height),
            )
            target_rgb = _resize(observation.rgb_uint8.astype(np.float32) / 255.0, size, "bilinear").astype(np.float32)
            target_depth = _resize(observation.depth_z_float32.astype(np.float32), size, "nearest").astype(np.float32)
            visibility_path = visibility_root / triplet.scene / f"{side}_visibility_truth.npz"
            with np.load(visibility_path, allow_pickle=False) as values:
                observed = _resize(values["observed_from_source_uint8"].astype(np.float32), size, "nearest") > 0.5
                occluded = _resize(values["occlusion_hidden_uint8"].astype(np.float32), size, "nearest") > 0.5
                outside = _resize(values["outside_source_fov_uint8"].astype(np.float32), size, "nearest") > 0.5
            if np.any(observed & occluded) or np.any(observed & outside) or np.any(occluded & outside):
                raise RuntimeError(f"visibility masks overlap: {triplet.scene}/{side}")
            warped_rgb, warped_depth, warp_valid = forward_splat_source_to_target(
                source_rgb,
                base,
                source_k,
                triplet.source_camera.world_to_camera_4x4_float64,
                target_k,
                observation.world_to_camera_4x4_float64,
            )
            rays, origin = target_camera_conditioning_in_source(
                target_height,
                target_width,
                target_k,
                triplet.source_camera.world_to_camera_4x4_float64,
                observation.world_to_camera_4x4_float64,
            )
            valid_base = np.isfinite(base) & (base > 0)
            depth_scale = float(np.median(base[valid_base]))
            def image_tensor(value: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(value).permute(2, 0, 1)[None].float().to(device)
            def map_tensor(value: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(value)[None, None].float().to(device)
            prepared.append(
                PreparedTargetSample(
                    split=triplet.split,
                    scene=triplet.scene,
                    source_frame=int(triplet.source_frame_index),
                    side=side,
                    target_id=observation.observation_id,
                    target_frame=target_frame,
                    warped_rgb=image_tensor(warped_rgb),
                    warped_depth=map_tensor(warped_depth),
                    warp_valid=map_tensor(warp_valid.astype(np.float32)),
                    rays_in_source=image_tensor(rays),
                    origin_in_source=image_tensor(origin),
                    source_depth_scale=torch.tensor([[[[depth_scale]]]], dtype=torch.float32, device=device),
                    target_rgb=image_tensor(target_rgb),
                    target_depth=map_tensor(target_depth),
                    observed=map_tensor(observed.astype(np.float32)).bool(),
                    occluded=map_tensor(occluded.astype(np.float32)).bool(),
                    outside=map_tensor(outside.astype(np.float32)).bool(),
                    input_hashes={
                        "base_arrays": portable(base_path),
                        "base_arrays_sha256": sha256(base_path),
                        "visibility_arrays": portable(visibility_path),
                        "visibility_arrays_sha256": sha256(visibility_path),
                    },
                )
            )
    return prepared


def target_metrics(output, sample: PreparedTargetSample) -> dict[str, object]:
    result: dict[str, object] = {"warp_coverage": float(sample.warp_valid.mean())}
    branch_support = {
        "occlusion_hidden": output.occlusion_support_probability,
        "outside_source_fov": output.outside_support_probability,
    }
    masks = {
        "observed_from_source": sample.observed,
        "occlusion_hidden": sample.occluded,
        "outside_source_fov": sample.outside,
    }
    regions = {}
    for name, mask in masks.items():
        valid = mask & torch.isfinite(sample.target_depth) & (sample.target_depth > 0)
        rgb_error = torch.abs(output.rgb - sample.target_rgb).mean(dim=1, keepdim=True)[valid]
        depth_error = torch.abs(output.depth_z[valid] - sample.target_depth[valid]) / sample.target_depth[valid]
        region = {
            "pixels": int(valid.sum()),
            "rgb_mae_0_255": float(rgb_error.mean() * 255),
            "depth_abs_rel": float(depth_error.mean()),
            "geometry_confidence_mean": float(output.geometry_confidence[valid].mean()),
            "appearance_confidence_mean": float(output.appearance_confidence[valid].mean()),
        }
        if name in branch_support:
            positive = branch_support[name][valid]
            negative_mask = (sample.observed | (sample.outside if name == "occlusion_hidden" else sample.occluded))
            region["branch_positive_support_mean"] = float(positive.mean())
            region["branch_negative_support_mean"] = float(branch_support[name][negative_mask].mean())
        regions[name] = region
    result["regions"] = regions
    return result
