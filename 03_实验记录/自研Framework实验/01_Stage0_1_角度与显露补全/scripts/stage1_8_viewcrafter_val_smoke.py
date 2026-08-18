#!/usr/bin/env python3
"""Run the frozen leakage-safe ViewCrafter 512 one-val adapter smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera-parameters", type=Path, required=True)
    parser.add_argument("--adaptive3dgs-src", type=Path, required=True)
    parser.add_argument("--viewcrafter-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--render-preflight-only", action="store_true")
    return parser.parse_args()


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


def resize_chw(value: np.ndarray, size: tuple[int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(value)
    if value.ndim == 2:
        tensor = tensor[None, None].float()
    else:
        tensor = tensor.permute(2, 0, 1)[None].float()
    kwargs: dict[str, object] = {"size": size, "mode": mode}
    if mode == "bilinear":
        kwargs.update({"align_corners": False, "antialias": True})
    return F.interpolate(tensor, **kwargs)


def crop_for_aspect(height: int, width: int, output_height: int, output_width: int) -> tuple[int, int, int, int]:
    input_aspect = width / height
    output_aspect = output_width / output_height
    if input_aspect > output_aspect:
        crop_height = height
        crop_width = int(round(height * output_aspect))
    else:
        crop_width = width
        crop_height = int(round(width / output_aspect))
    return (width - crop_width) // 2, (height - crop_height) // 2, crop_width, crop_height


def adjusted_intrinsics(k: np.ndarray, crop: tuple[int, int, int, int], output_size: tuple[int, int]) -> np.ndarray:
    crop_x, crop_y, crop_width, crop_height = crop
    output_height, output_width = output_size
    result = k.astype(np.float64, copy=True)
    result[0, 2] -= crop_x
    result[1, 2] -= crop_y
    result[0] *= output_width / crop_width
    result[1] *= output_height / crop_height
    return result


def interpolate_c2w(source_w2c: np.ndarray, target_w2c: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    relative_w2c = target_w2c @ np.linalg.inv(source_w2c)
    target_c2w = np.linalg.inv(relative_w2c)
    alphas = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((np.eye(3), target_c2w[:3, :3]))))
    rotations = slerp(alphas).as_matrix()
    translations = alphas[:, None] * target_c2w[:3, 3][None]
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    c2w[:, :3, :3] = rotations
    c2w[:, :3, 3] = translations
    c2w[0] = np.eye(4)
    c2w[-1] = target_c2w
    return c2w, relative_w2c


def make_pytorch3d_cameras(c2w_cv: np.ndarray, source_k: np.ndarray, target_k: np.ndarray,
                           image_size: tuple[int, int], device: torch.device):
    from pytorch3d.renderer import PerspectiveCameras

    frames = len(c2w_cv)
    c2w = torch.from_numpy(c2w_cv).to(device=device, dtype=torch.float32)
    converted_rotation = torch.stack((-c2w[:, :, 0], -c2w[:, :, 1], c2w[:, :, 2]), dim=2)
    converted_c2w = torch.cat((converted_rotation[:, :3], c2w[:, :3, 3:4]), dim=2)
    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32).view(1, 1, 4).repeat(frames, 1, 1)
    w2c = torch.linalg.inv(torch.cat((converted_c2w, bottom), dim=1))
    rotation = w2c[:, :3, :3].permute(0, 2, 1)
    translation = w2c[:, :3, 3]
    alpha = torch.linspace(0, 1, frames, device=device)[:, None]
    source_focal = torch.tensor([[source_k[0, 0], source_k[1, 1]]], device=device, dtype=torch.float32)
    target_focal = torch.tensor([[target_k[0, 0], target_k[1, 1]]], device=device, dtype=torch.float32)
    source_pp = torch.tensor([[source_k[0, 2], source_k[1, 2]]], device=device, dtype=torch.float32)
    target_pp = torch.tensor([[target_k[0, 2], target_k[1, 2]]], device=device, dtype=torch.float32)
    focal = source_focal * (1 - alpha) + target_focal * alpha
    principal = source_pp * (1 - alpha) + target_pp * alpha
    sizes = torch.tensor([list(image_size)], device=device, dtype=torch.float32).repeat(frames, 1)
    return PerspectiveCameras(
        focal_length=focal, principal_point=principal, in_ndc=False, image_size=sizes,
        R=rotation, T=translation, device=device,
    )


def unproject(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float64)
    return (pixels @ np.linalg.inv(k).T) * depth[..., None]


def geometry_gate(points_cv: np.ndarray, relative_w2c: np.ndarray, target_k: np.ndarray,
                  cameras, image_size: tuple[int, int]) -> dict[str, float | int]:
    height, width = image_size
    sampled = points_cv[::8, ::8].reshape(-1, 3)
    target_xyz = sampled @ relative_w2c[:3, :3].T + relative_w2c[:3, 3]
    canonical_h = target_xyz @ target_k.T
    canonical_xy = canonical_h[:, :2] / canonical_h[:, 2:3]
    with torch.no_grad():
        transformed = cameras[-1].transform_points_screen(
            torch.from_numpy(sampled)[None].to(device=cameras.device, dtype=cameras.R.dtype),
            image_size=((height, width),),
        )[0, :, :2].cpu().numpy()
    valid = np.isfinite(sampled).all(axis=1) & np.isfinite(canonical_xy).all(axis=1)
    valid &= np.isfinite(transformed).all(axis=1) & (sampled[:, 2] > 0) & (target_xyz[:, 2] > 0)
    all_error = np.linalg.norm(transformed[valid] - canonical_xy[valid], axis=1)
    raster = valid & (canonical_xy[:, 0] >= 0) & (canonical_xy[:, 0] < width)
    raster &= (canonical_xy[:, 1] >= 0) & (canonical_xy[:, 1] < height)
    raster_error = np.linalg.norm(transformed[raster] - canonical_xy[raster], axis=1)
    if not len(all_error) or not len(raster_error):
        raise RuntimeError("geometry gate found no valid comparison points")
    return {
        "compared_points": int(len(all_error)),
        "raster_compared_points": int(len(raster_error)),
        "offscreen_points": int(len(all_error) - len(raster_error)),
        "median_px": float(np.median(all_error)),
        "p99_px": float(np.quantile(all_error, 0.99)),
        "maximum_px": float(np.max(raster_error)),
        "all_positive_depth_maximum_px_diagnostic": float(np.max(all_error)),
        "maximum_px_scope": "canonical_target_raster_only",
    }


def save_rgb(path: Path, value: torch.Tensor, input_range: tuple[float, float] = (0.0, 1.0)) -> None:
    low, high = input_range
    array = ((value.detach().float().cpu() - low) / (high - low)).clamp(0, 1)
    if array.ndim == 4:
        array = array[0]
    array = array.permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(path)


def save_video(path: Path, frames_thwc: torch.Tensor, fps: int = 8) -> None:
    import imageio.v2 as imageio
    frames = np.rint(frames_thwc.detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=None)


def install_sdpa_spatial_attention() -> None:
    """Use modern PyTorch SDPA where official ViewCrafter expects optional xFormers."""
    import lvdm.modules.attention as attention

    def sdpa_forward(self, x, context=None, mask=None):
        if mask is not None:
            raise NotImplementedError("ViewCrafter spatial attention does not use a mask")
        spatial_self_attention = context is None
        q = self.to_q(x)
        context = x if context is None else context
        k_ip = v_ip = None
        if self.image_cross_attention and not spatial_self_attention:
            context, context_image = context[:, :self.text_context_len], context[:, self.text_context_len:]
            k = self.to_k(context); v = self.to_v(context)
            k_ip = self.to_k_ip(context_image); v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attention:
                context = context[:, :self.text_context_len]
            k = self.to_k(context); v = self.to_v(context)

        batch = q.shape[0]
        def split_heads(value):
            return value.reshape(batch, value.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3)
        qh, kh, vh = map(split_heads, (q, k, v))
        out = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
        out = out.permute(0, 2, 1, 3).reshape(batch, out.shape[2], self.heads * self.dim_head)
        if k_ip is not None:
            out_ip = F.scaled_dot_product_attention(
                qh, split_heads(k_ip), split_heads(v_ip), dropout_p=0.0, is_causal=False,
            )
            out_ip = out_ip.permute(0, 2, 1, 3).reshape(batch, out_ip.shape[2], self.heads * self.dim_head)
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
        return self.to_out(out)

    attention.CrossAttention.efficient_forward = sdpa_forward
    attention.XFORMERS_IS_AVAILBLE = True


def main() -> None:
    args = parse_args()
    required = [args.config, args.dataset_root, args.camera_parameters, args.adaptive3dgs_src, args.viewcrafter_repo]
    if not args.preflight_only and not args.render_preflight_only:
        required.append(args.checkpoint)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite output or record")
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    sys.path.insert(0, str(args.adaptive3dgs_src.resolve()))
    from adaptive3dgs.datasets import load_hypersim_training_triplets

    triplets = load_hypersim_training_triplets(args.config, args.dataset_root, args.camera_parameters)
    if len(triplets) != 1:
        raise RuntimeError("smoke config must load exactly one triplet")
    triplet = triplets[0]
    side = config["selected_target"]["side"]
    target = triplet.target_observations[{"left": 0, "right": 1}[side]]
    expected_frame = int(config["selected_target"]["target_frame"])
    if int(target.observation_id.rsplit("-", 1)[1]) != expected_frame:
        raise RuntimeError("target frame mismatch")

    base_path = Path(config["frozen_base_depth"]["arrays"])
    visibility_path = Path(config["selected_target"]["visibility_arrays"])
    for path, expected in ((base_path, config["frozen_base_depth"]["arrays_sha256"]),
                           (visibility_path, config["selected_target"]["visibility_arrays_sha256"])):
        if sha256(path) != expected:
            raise RuntimeError(f"input hash mismatch: {path}")
    with np.load(base_path, allow_pickle=False) as values:
        base_depth = values[config["frozen_base_depth"]["array_key"]].astype(np.float32)
    source_rgb = triplet.source_rgb_uint8.astype(np.float32) / 255.0
    target_rgb = target.rgb_uint8.astype(np.float32) / 255.0
    original_height, original_width = base_depth.shape
    output_size = tuple(map(int, config["adapter"]["input_size_hw"]))
    crop_x, crop_y, crop_width, crop_height = crop_for_aspect(original_height, original_width, *output_size)
    crop_slice = (slice(crop_y, crop_y + crop_height), slice(crop_x, crop_x + crop_width))
    source_tensor = resize_chw(source_rgb[crop_slice], output_size, "bilinear")
    target_tensor = resize_chw(target_rgb[crop_slice], output_size, "bilinear")
    depth_tensor = resize_chw(base_depth[crop_slice], output_size, "bilinear")
    depth = depth_tensor[0, 0].numpy()
    crop = (crop_x, crop_y, crop_width, crop_height)
    source_k = adjusted_intrinsics(triplet.source_camera.intrinsics_3x3_float64, crop, output_size)
    target_k = adjusted_intrinsics(target.intrinsics_3x3_float64, crop, output_size)
    c2w, relative_w2c = interpolate_c2w(
        triplet.source_camera.world_to_camera_4x4_float64,
        target.world_to_camera_4x4_float64,
        int(config["adapter"]["video_frames"]),
    )
    cpu_device = torch.device("cpu")
    cameras_cpu = make_pytorch3d_cameras(c2w, source_k, target_k, output_size, cpu_device)
    points = unproject(depth, source_k)
    geometry = geometry_gate(points, relative_w2c, target_k, cameras_cpu, output_size)
    gates = config["gates"]
    if geometry["compared_points"] < gates["minimum_compared_geometry_points"]:
        raise RuntimeError(f"too few geometry points: {geometry}")
    if geometry["p99_px"] > gates["geometry_projection_p99_px_max"] or geometry["maximum_px"] > gates["geometry_projection_max_px_max"]:
        raise RuntimeError(f"camera projection gate failed: {geometry}")

    common_record = {
        "schema_version": config["schema_version"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "producer_machine_id": args.machine_id,
        "physical_gpu_index": args.physical_gpu_index,
        "inputs": {
            "config": portable(args.config), "config_sha256": sha256(args.config),
            "dataset_root": portable(args.dataset_root),
            "camera_parameters": portable(args.camera_parameters), "camera_parameters_sha256": sha256(args.camera_parameters),
            "base_depth": portable(base_path), "base_depth_sha256": sha256(base_path),
            "visibility": portable(visibility_path), "visibility_sha256": sha256(visibility_path),
            "target_rgb_model_input": False, "target_depth_model_input": False,
            "held_out_test_read": False, "p01_read": False,
        },
        "target": {
            "split": triplet.split, "scene": triplet.scene, "camera": triplet.camera_name,
            "source_frame": triplet.source_frame_index, "side": side, "target_frame": expected_frame,
            "yaw_deg": config["triplets"][0][f"{side}_yaw_deg"],
        },
        "adapter": {
            "original_shape_hw": [original_height, original_width], "crop_xywh": list(crop),
            "output_shape_hw": list(output_size), "frames": len(c2w),
            "source_intrinsics_after_crop_resize": source_k.tolist(),
            "target_intrinsics_after_crop_resize": target_k.tolist(),
            "relative_target_w2c_cv": relative_w2c.tolist(),
            "coordinate_conversion": "OpenCV RDF point cloud plus official ViewCrafter/PyTorch3D RDF-to-LUF camera-column conversion",
            "geometry_gate": geometry,
        },
        "quality_boundary": config["quality_boundary"],
    }
    save_rgb(args.output_dir / "source.png", source_tensor)
    if args.preflight_only:
        common_record.update({
            "status": "pass_geometry_preflight",
            "model_loaded": False,
            "gpu_inference_run": False,
            "gates": {"frozen": gates, "result": "pass_geometry_only"},
            "artifacts": {"source": {"path": portable(args.output_dir / "source.png"), "sha256": sha256(args.output_dir / "source.png")}},
        })
        args.record.write_text(json.dumps(common_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": common_record["status"], "geometry": geometry}, indent=2))
        return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(int(config["model"]["seed"])); torch.cuda.manual_seed_all(int(config["model"]["seed"]))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    sys.path.insert(0, str(args.viewcrafter_repo.resolve()))
    from omegaconf import OmegaConf
    from pytorch3d.renderer import AlphaCompositor, PointsRasterizationSettings, PointsRasterizer, PointsRenderer
    from pytorch3d.structures import Pointclouds

    started = time.monotonic()
    cameras = make_pytorch3d_cameras(c2w, source_k, target_k, output_size, device)
    settings = PointsRasterizationSettings(
        image_size=output_size, radius=float(config["adapter"]["point_radius_ndc"]),
        points_per_pixel=int(config["adapter"]["points_per_pixel"]), bin_size=0,
    )
    renderer = PointsRenderer(PointsRasterizer(cameras=cameras, raster_settings=settings), AlphaCompositor())
    valid = np.isfinite(points).all(axis=2) & np.isfinite(depth) & (depth > 0)
    point_tensor = torch.from_numpy(points[valid]).float().to(device)
    color_tensor = source_tensor[0].permute(1, 2, 0)[torch.from_numpy(valid)].float().to(device)
    cloud = Pointclouds(points=[point_tensor], features=[color_tensor]).extend(len(c2w))
    with torch.inference_mode():
        renderings = renderer(cloud)[..., :3]
        renderings[0] = source_tensor[0].permute(1, 2, 0).to(device)
    if not bool(torch.isfinite(renderings).all()):
        raise RuntimeError("non-finite point-render conditioning video")

    if args.render_preflight_only:
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        save_rgb(args.output_dir / "target_truth_val_only.png", target_tensor)
        save_rgb(args.output_dir / "rendered_target.png", renderings[-1].permute(2, 0, 1))
        save_video(args.output_dir / "conditioning.mp4", renderings)
        artifacts = {}
        for path in sorted(args.output_dir.iterdir()):
            if path.is_file():
                artifacts[path.stem] = {"path": portable(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        common_record.update({
            "status": "pass_render_preflight",
            "model_loaded": False,
            "gpu_inference_run": False,
            "runtime": {
                "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                "peak_allocated_gib": peak_allocated / 1024 ** 3,
                "peak_reserved_gib": peak_reserved / 1024 ** 3,
            },
            "outputs": {
                "renderings_shape": list(renderings.shape), "renderings_all_finite": True,
                "renderings_minimum": float(renderings.min()), "renderings_maximum": float(renderings.max()),
                "renderings_mean": float(renderings.float().mean()), "renderings_std": float(renderings.float().std()),
            },
            "artifacts": artifacts,
            "gates": {"frozen": gates, "result": "pass_geometry_and_render_only"},
        })
        args.record.write_text(json.dumps(common_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": common_record["status"], "geometry": geometry, "runtime": common_record["runtime"]}, indent=2))
        return

    expected_bytes = int(config["model"]["checkpoint_bytes"])
    if args.checkpoint.stat().st_size != expected_bytes or sha256(args.checkpoint) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint byte count or SHA256 mismatch")

    from torch.utils import checkpoint as _torch_checkpoint  # noqa: F401 -- expose lazy submodule
    from utils.diffusion_utils import image_guided_synthesis, instantiate_from_config, load_model_checkpoint
    install_sdpa_spatial_attention()

    model_cfg = OmegaConf.load(args.viewcrafter_repo / config["model"]["config"])["model"]
    model_cfg["params"]["unet_config"]["params"]["use_checkpoint"] = False
    # The official checkpoint is loaded strictly and contains both OpenCLIP encoders.
    # Avoid open_clip downloading the same LAION bootstrap weights just to overwrite them.
    model_cfg["params"]["cond_stage_config"]["params"]["version"] = None
    model_cfg["params"]["img_cond_stage_config"]["params"]["version"] = None
    model = instantiate_from_config(model_cfg).to(device)
    model.cond_stage_model.device = str(device)
    model.perframe_ae = bool(config["model"]["perframe_ae"])
    model = load_model_checkpoint(model, str(args.checkpoint)).eval()
    height, width = output_size
    noise_shape = [1, model.model.diffusion_model.out_channels, len(c2w), height // 8, width // 8]
    videos = (renderings * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        samples = image_guided_synthesis(
            model, [config["model"]["prompt"]], videos, noise_shape,
            n_samples=1, ddim_steps=int(config["model"]["ddim_steps"]), ddim_eta=float(config["model"]["ddim_eta"]),
            unconditional_guidance_scale=float(config["model"]["unconditional_guidance_scale"]), cfg_img=None,
            fs=int(config["model"]["frame_stride"]), text_input=bool(config["model"]["text_input"]),
            multiple_cond_cfg=False, timestep_spacing=config["model"]["timestep_spacing"],
            guidance_rescale=float(config["model"]["guidance_rescale"]), condition_index=[0],
        )
    generated_raw = samples[0, 0]
    if not bool(torch.isfinite(generated_raw).all()):
        raise RuntimeError("non-finite generated video before official clamp")
    generated_raw_minimum = float(generated_raw.min())
    generated_raw_maximum = float(generated_raw.max())
    generated = torch.clamp(generated_raw, -1.0, 1.0)
    torch.cuda.synchronize(device)
    runtime = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device)); peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if not bool(torch.isfinite(generated).all()):
        raise RuntimeError("non-finite generated video")
    low, high = map(float, gates["generated_range"])
    if float(generated.min()) < low - 1e-4 or float(generated.max()) > high + 1e-4:
        raise RuntimeError("generated video outside frozen range")
    if float(generated.float().std()) < float(gates["minimum_generated_std"]):
        raise RuntimeError("generated video is effectively constant")
    peak_reserved_gib = peak_reserved / 1024 ** 3
    if peak_reserved_gib > float(gates["peak_reserved_gib_max"]):
        raise RuntimeError(f"peak memory gate failed: {peak_reserved_gib:.3f} GiB")

    generated_thwc = ((generated + 1) / 2).permute(1, 2, 3, 0)
    save_rgb(args.output_dir / "target_truth_val_only.png", target_tensor)
    save_rgb(args.output_dir / "rendered_target.png", renderings[-1].permute(2, 0, 1))
    save_rgb(args.output_dir / "generated_target.png", generated[:, -1], (-1, 1))
    save_video(args.output_dir / "conditioning.mp4", renderings)
    save_video(args.output_dir / "generated.mp4", generated_thwc)
    comparison = torch.cat((source_tensor, renderings[-1].permute(2, 0, 1)[None].cpu(),
                            generated[:, -1][None].cpu().add(1).div(2), target_tensor), dim=3)
    save_rgb(args.output_dir / "comparison_source_render_generated_truth.png", comparison)
    artifacts = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            artifacts[path.stem] = {"path": portable(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    common_record.update({
        "status": "pass",
        "model": {
            **config["model"], "checkpoint": portable(args.checkpoint),
            "openclip_bootstrap_pretrained": False,
            "openclip_weights_source": "strict official ViewCrafter checkpoint",
            "spatial_attention_backend": "torch.nn.functional.scaled_dot_product_attention",
            "xformers_downloaded": False,
        },
        "runtime": {
            "seconds_including_model_load": runtime,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": peak_allocated / 1024 ** 3, "peak_reserved_gib": peak_reserved_gib,
        },
        "outputs": {
            "renderings_shape": list(renderings.shape), "renderings_all_finite": True,
            "generated_shape": list(generated.shape), "generated_all_finite": True,
            "generated_raw_minimum_before_official_clamp": generated_raw_minimum,
            "generated_raw_maximum_before_official_clamp": generated_raw_maximum,
            "generated_minimum": float(generated.min()), "generated_maximum": float(generated.max()),
            "generated_mean": float(generated.float().mean()), "generated_std": float(generated.float().std()),
        },
        "artifacts": artifacts,
        "gates": {"frozen": gates, "result": "pass"},
    })
    args.record.write_text(json.dumps(common_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "geometry": geometry, "runtime": common_record["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
