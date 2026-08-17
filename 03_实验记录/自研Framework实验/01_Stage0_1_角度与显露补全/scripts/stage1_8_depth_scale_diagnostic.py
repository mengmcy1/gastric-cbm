#!/usr/bin/env python3
"""Measure the per-target oracle scale upper bound of a failed Stage 1.8 candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from adaptive3dgs.models import TargetViewUNet
from stage1_8_target_view_common import portable, prepare_target_samples, sha256


REGIONS = ("occlusion_hidden", "outside_source_fov")


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "diagnostic_config", "candidate_record", "pool_record", "triplet_config", "dataset_root",
        "camera_parameters", "visibility_root", "base_record", "record",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    args = parser.parse_args()
    for name in (
        "diagnostic_config", "candidate_record", "pool_record", "triplet_config", "dataset_root",
        "camera_parameters", "visibility_root", "base_record", "record",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite {args.record}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must begin with physical GPU index")
    diagnostic = json.loads(args.diagnostic_config.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_record.read_text(encoding="utf-8"))
    pool = json.loads(args.pool_record.read_text(encoding="utf-8"))
    checkpoint_path = Path(candidate["artifacts"]["checkpoint"])
    if sha256(checkpoint_path) != candidate["artifacts"]["checkpoint_sha256"]:
        raise RuntimeError("candidate checkpoint hash mismatch")
    selected = {(item["scene"], item["side"]) for item in pool["eligible_targets"] if item["split"] == "val"}
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    size = tuple(checkpoint["input_size_hw"])
    samples = prepare_target_samples(
        args.triplet_config, args.dataset_root, args.camera_parameters, args.visibility_root,
        args.base_record, size, device, selected=selected,
    )
    model = TargetViewUNet(base_channels=checkpoint["architecture"]["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    totals = {name: {"raw_abs_sum": 0.0, "aligned_abs_sum": 0.0, "pixels": 0} for name in REGIONS}
    targets = []
    with torch.no_grad():
        for sample in samples:
            output = model(*sample.inputs())
            novel = (sample.occluded | sample.outside) & torch.isfinite(sample.target_depth) & (sample.target_depth > 0)
            predicted = output.depth_z[novel]; truth = sample.target_depth[novel]
            log_scale = torch.median(torch.log(truth) - torch.log(predicted.clamp_min(1e-6)))
            scale = torch.exp(log_scale)
            target_result = {"scene": sample.scene, "side": sample.side, "target_frame": sample.target_frame, "oracle_scale": float(scale), "regions": {}}
            for name, mask in (("occlusion_hidden", sample.occluded), ("outside_source_fov", sample.outside)):
                valid = mask & torch.isfinite(sample.target_depth) & (sample.target_depth > 0)
                raw = torch.abs(output.depth_z[valid] - sample.target_depth[valid]) / sample.target_depth[valid]
                aligned = torch.abs(scale * output.depth_z[valid] - sample.target_depth[valid]) / sample.target_depth[valid]
                count = int(valid.sum())
                totals[name]["raw_abs_sum"] += float(raw.sum()); totals[name]["aligned_abs_sum"] += float(aligned.sum()); totals[name]["pixels"] += count
                target_result["regions"][name] = {"pixels": count, "raw_abs_rel": float(raw.mean()), "scale_aligned_abs_rel": float(aligned.mean())}
            targets.append(target_result)
    regions = {
        name: {
            "pixels": value["pixels"],
            "raw_abs_rel": value["raw_abs_sum"] / value["pixels"],
            "scale_aligned_abs_rel": value["aligned_abs_sum"] / value["pixels"],
        }
        for name, value in totals.items()
    }
    threshold = diagnostic["decision"]["both_regions_scale_aligned_abs_rel_maximum_for_scale_dominant"]
    scale_dominant = all(regions[name]["scale_aligned_abs_rel"] <= threshold for name in REGIONS)
    scales = [item["oracle_scale"] for item in targets]
    record = {
        "schema_version": "stage1.8-depth-scale-diagnostic-v1",
        "status": "complete_diagnostic",
        "producer_machine_id": "linux5080", "physical_gpu_index": args.physical_gpu_index,
        "inputs": {
            "diagnostic_config": portable(args.diagnostic_config), "diagnostic_config_sha256": sha256(args.diagnostic_config),
            "candidate_record": portable(args.candidate_record), "candidate_record_sha256": sha256(args.candidate_record),
            "checkpoint": portable(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "pool_record": portable(args.pool_record), "pool_record_sha256": sha256(args.pool_record),
            "held_out_test_used": False, "oracle_scale_available_at_deployment": False,
        },
        "summary": {
            "validation_targets": len(targets), "regions": regions,
            "oracle_scale_minimum": min(scales), "oracle_scale_median": float(np.median(scales)), "oracle_scale_maximum": max(scales),
            "scale_dominant_by_frozen_decision": scale_dominant,
        },
        "targets": targets,
        "decision": (
            "global_scale_is_dominant_next_variable" if scale_dominant
            else "global_scale_alone_is_insufficient_change_rgbd_backbone_or_local_geometry"
        ),
        "quality_boundary": diagnostic["quality_boundary"],
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "summary": record["summary"], "decision": record["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
