from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import ExifTags, Image, ImageOps, TiffTags
from scipy.spatial import cKDTree

from src.demo.config import RootCfg, load_typed_root_config
from src.model.decoder import Decoder, get_decoder
from src.model.encoder import Encoder, get_encoder
from src.utils.gaussians import Gaussians3D
from src.utils.io import save_video

DEFAULT_FOCAL_35MM_MM = 30.0
INFERENCE_HEIGHT = 1152
INFERENCE_WIDTH = 1536
DEFAULT_VIDEO_FPS = 10
DEFAULT_VIDEO_FRAMES = 60
DEFAULT_RENDER_CHUNK_SIZE = 8
PROMPT_DEPTH_EPS = 1e-6
PROMPT_DISPARITY_MIN_DEPTH = 1e-2


@dataclass(frozen=True)
class ResolvedIntrinsics:
    """Resolved intrinsics for one input image.

    Args:
        intrinsics_px: Pixel-space intrinsics with shape [3, 3].
        focal_length_px: Pixel focal length used for export and trajectory sizing.
    """

    intrinsics_px: torch.Tensor
    focal_length_px: float


@dataclass(frozen=True)
class PromptInputs:
    """Prompt-conditioned depth inputs for one single-view demo sample.

    Args:
        prompt_disparity: Sparse prompt disparity with shape [1, H, W].
        prompt_mask: Prompt validity mask with shape [1, H, W].
    """

    prompt_disparity: torch.Tensor
    prompt_mask: torch.Tensor


@dataclass(frozen=True)
class DemoImageBundle:
    """Original/render image data and inference image data for one demo sample.

    Args:
        original_image_shape: Original image shape as (height, width).
        inference_image: Inference-resolution image tensor with shape [3, H_inf, W_inf].
        original_intrinsics: Pixel-space intrinsics resolved in original image space.
        inference_intrinsics: Pixel-space intrinsics resolved in inference image space.
    """

    original_image_shape: tuple[int, int]
    inference_image: torch.Tensor
    original_intrinsics: ResolvedIntrinsics
    inference_intrinsics: ResolvedIntrinsics


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value}.")


def convert_focallength_mm_to_px(width: float, height: float, focal_mm: float) -> float:
    """Convert 35mm-equivalent focal length in millimeters to pixels."""
    _validate_positive_finite(width, "Image width")
    _validate_positive_finite(height, "Image height")
    _validate_positive_finite(focal_mm, "Focal length in millimeters")
    return focal_mm * math.sqrt(width**2.0 + height**2.0) / math.sqrt(36.0**2 + 24.0**2)


def build_intrinsics_from_focal_px(
    focal_length_px: float,
    width: int,
    height: int,
) -> torch.Tensor:
    """Create a pixel-space OpenCV intrinsics matrix."""
    _validate_positive_finite(focal_length_px, "Focal length in pixels")
    _validate_positive_finite(width, "Image width")
    _validate_positive_finite(height, "Image height")
    return torch.tensor(
        [
            [focal_length_px, 0.0, (width - 1) / 2.0],
            [0.0, focal_length_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def normalize_intrinsics(
    intrinsics_px: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert pixel-space intrinsics into the normalized InfiniSplat convention."""
    intrinsics_norm = intrinsics_px.clone()
    intrinsics_norm[0] = intrinsics_norm[0] / float(width)
    intrinsics_norm[1] = intrinsics_norm[1] / float(height)
    return intrinsics_norm


def scale_intrinsics_px(
    intrinsics_px: torch.Tensor,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> torch.Tensor:
    """Scale pixel-space intrinsics from one image resolution to another."""
    scaled = intrinsics_px.clone()
    scale_x = float(dst_width) / float(src_width)
    scale_y = float(dst_height) / float(src_height)
    scaled[0] = scaled[0] * scale_x
    scaled[1] = scaled[1] * scale_y
    return scaled


def _extract_exif_dict(image: Image.Image) -> dict[str, Any]:
    exif_block = image.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in exif_block.items() if k in ExifTags.TAGS}
    tiff_tags = image.getexif()
    tiff_dict = {TiffTags.TAGS_V2[k].name: v for k, v in tiff_tags.items() if k in TiffTags.TAGS_V2}
    return {**exif_dict, **tiff_dict}


def _extract_exif_focal_length_px(
    image: Image.Image,
    width: int,
    height: int,
) -> float | None:
    exif_dict = _extract_exif_dict(image)
    focal_35mm = _as_float(
        exif_dict.get("FocalLengthIn35mmFilm", exif_dict.get("FocalLenIn35mmFilm"))
    )
    if focal_35mm is None or focal_35mm < 1.0:
        focal_mm = _as_float(exif_dict.get("FocalLength"))
        if focal_mm is None:
            return None
        if focal_mm < 10.0:
            focal_35mm = focal_mm * 8.4
        else:
            focal_35mm = focal_mm
    return convert_focallength_mm_to_px(width, height, focal_35mm)


def _load_intrinsics_override(path: Path) -> torch.Tensor:
    cfg = OmegaConf.load(path)
    container = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(container, dict):
        if "intrinsics_px" in container:
            matrix = container["intrinsics_px"]
        elif "camera" in container and isinstance(container["camera"], dict) and "intrinsics_px" in container["camera"]:
            matrix = container["camera"]["intrinsics_px"]
        else:
            raise KeyError(
                f"Could not find a 3x3 intrinsics matrix in {path}. Expected `intrinsics_px` or `camera.intrinsics_px`."
            )
    else:
        matrix = container
    try:
        intrinsics_px = torch.tensor(matrix, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Intrinsics matrix in {path} must contain only numeric values.") from exc
    if intrinsics_px.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsics matrix in {path}, got {tuple(intrinsics_px.shape)}.")
    if not torch.isfinite(intrinsics_px).all():
        raise ValueError(f"Intrinsics matrix in {path} must contain only finite values.")
    if intrinsics_px[0, 0] <= 0.0 or intrinsics_px[1, 1] <= 0.0:
        raise ValueError(f"Intrinsics matrix in {path} must have positive fx and fy.")
    if abs(float(torch.linalg.det(intrinsics_px).item())) <= torch.finfo(intrinsics_px.dtype).eps:
        raise ValueError(f"Intrinsics matrix in {path} must be non-singular.")
    return intrinsics_px


def resolve_intrinsics(
    image: Image.Image,
    focal_length_px: float | None = None,
    focal_length_mm: float | None = None,
    intrinsics_override_path: Path | None = None,
    default_focal_35mm_mm: float = DEFAULT_FOCAL_35MM_MM,
) -> ResolvedIntrinsics:
    """Resolve pixel-space and normalized intrinsics for one image."""
    width, height = image.size
    if intrinsics_override_path is not None:
        intrinsics_px = _load_intrinsics_override(intrinsics_override_path)
        resolved_focal_px = float(intrinsics_px[0, 0].item())
    elif focal_length_px is not None:
        intrinsics_px = build_intrinsics_from_focal_px(focal_length_px, width, height)
        resolved_focal_px = float(focal_length_px)
    elif focal_length_mm is not None:
        resolved_focal_px = convert_focallength_mm_to_px(width, height, focal_length_mm)
        intrinsics_px = build_intrinsics_from_focal_px(resolved_focal_px, width, height)
    else:
        exif_focal_px = _extract_exif_focal_length_px(image, width, height)
        if exif_focal_px is not None:
            resolved_focal_px = exif_focal_px
            intrinsics_px = build_intrinsics_from_focal_px(resolved_focal_px, width, height)
        else:
            resolved_focal_px = convert_focallength_mm_to_px(width, height, default_focal_35mm_mm)
            intrinsics_px = build_intrinsics_from_focal_px(resolved_focal_px, width, height)

    return ResolvedIntrinsics(
        intrinsics_px=intrinsics_px,
        focal_length_px=resolved_focal_px,
    )


def load_demo_image_bundle(
    image_path: Path,
    focal_length_px: float | None = None,
    focal_length_mm: float | None = None,
    intrinsics_override_path: Path | None = None,
    default_focal_35mm_mm: float = DEFAULT_FOCAL_35MM_MM,
) -> DemoImageBundle:
    """Load both original-resolution and inference-resolution image/intrinsics bundles."""
    with Image.open(image_path) as image_pil:
        image_rgb = ImageOps.exif_transpose(image_pil).convert("RGB")
        original_width, original_height = image_rgb.size
        original_intrinsics = resolve_intrinsics(
            image=image_rgb,
            focal_length_px=focal_length_px,
            focal_length_mm=focal_length_mm,
            intrinsics_override_path=intrinsics_override_path,
            default_focal_35mm_mm=default_focal_35mm_mm,
        )

        target_height = INFERENCE_HEIGHT
        target_width = INFERENCE_WIDTH

        inference_image_rgb = image_rgb
        inference_intrinsics = original_intrinsics
        if (original_height, original_width) != (target_height, target_width):
            inference_image_rgb = image_rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)
            inference_intrinsics_px = scale_intrinsics_px(
                original_intrinsics.intrinsics_px,
                src_width=original_width,
                src_height=original_height,
                dst_width=target_width,
                dst_height=target_height,
            )
            inference_intrinsics = ResolvedIntrinsics(
                intrinsics_px=inference_intrinsics_px,
                focal_length_px=float(inference_intrinsics_px[0, 0].item()),
            )

        return DemoImageBundle(
            original_image_shape=(original_height, original_width),
            inference_image=tf.ToTensor()(inference_image_rgb),
            original_intrinsics=original_intrinsics,
            inference_intrinsics=inference_intrinsics,
        )


def _to_single_channel_depth(depth: np.ndarray, depth_path: Path) -> np.ndarray:
    """Convert loaded depth arrays to a single-channel float32 map."""
    depth = np.asarray(depth)
    while depth.ndim > 3:
        singleton_axes = [axis for axis, size in enumerate(depth.shape) if size == 1]
        if not singleton_axes:
            break
        depth = np.squeeze(depth, axis=singleton_axes[0])
    if depth.ndim == 3 and depth.shape[0] in (1, 3) and depth.shape[2] not in (1, 3):
        depth = np.moveaxis(depth, 0, -1)
    if depth.ndim == 2:
        return depth.astype(np.float32)
    if depth.ndim == 3:
        if depth.shape[2] == 1:
            return depth[:, :, 0].astype(np.float32)
        if np.issubdtype(depth.dtype, np.integer) and depth.shape[2] >= 3:
            ch0 = depth[:, :, 0]
            ch1 = depth[:, :, 1]
            ch2 = depth[:, :, 2]
            if np.array_equal(ch0, ch1) and np.array_equal(ch1, ch2):
                return ch0.astype(np.float32)
            return (
                ch0.astype(np.float32) * (256.0**2)
                + ch1.astype(np.float32) * 256.0
                + ch2.astype(np.float32)
            )
        return depth[:, :, 0].astype(np.float32)
    raise ValueError(f"Unsupported depth shape {depth.shape} for file: {depth_path}")


def _decode_sparse_depth(
    mask: np.ndarray,
    value: np.ndarray,
    depth_path: Path,
) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    value = np.asarray(value, dtype=np.float32)
    depth = np.zeros(mask.shape, dtype=np.float32)
    if value.shape == mask.shape:
        depth[mask] = value[mask]
        return depth

    value_flat = value.reshape(-1)
    valid_count = int(mask.sum())
    if value_flat.size < valid_count:
        raise ValueError(
            f"Depth value count ({value_flat.size}) is smaller than the mask valid count "
            f"({valid_count}): {depth_path}"
        )
    depth[mask] = value_flat[:valid_count]
    return depth


def _load_depth_from_npz(depth_path: Path) -> np.ndarray:
    with np.load(depth_path, allow_pickle=False) as npz_data:
        if len(npz_data.files) == 0:
            raise ValueError(f"Empty npz depth file: {depth_path}")
        if "mask" in npz_data.files and "value" in npz_data.files:
            return _decode_sparse_depth(
                npz_data["mask"],
                npz_data["value"],
                depth_path,
            )
        preferred_keys = ("depth", "data", "depth_map", "arr_0")
        key = next((k for k in preferred_keys if k in npz_data.files), npz_data.files[0])
        depth = np.asarray(npz_data[key])
    return _to_single_channel_depth(depth, depth_path)


def _find_h5_dataset(node: Any) -> Any:
    preferred_keys = ("dataset", "depth", "data")
    for key in preferred_keys:
        if key in node:
            child = node[key]
            if child.__class__.__name__ == "Dataset":
                return child
    for key in node.keys():
        child = node[key]
        if child.__class__.__name__ == "Dataset":
            return child
    for key in node.keys():
        child = node[key]
        if child.__class__.__name__ == "Group":
            found = _find_h5_dataset(child)
            if found is not None:
                return found
    return None


def _load_depth_from_h5(depth_path: Path) -> np.ndarray:
    import h5py

    with h5py.File(depth_path, "r") as h5_file:
        dataset = _find_h5_dataset(h5_file)
        if dataset is None:
            raise ValueError(f"No dataset found in h5 file: {depth_path}")
        depth = np.asarray(dataset)
    return _to_single_channel_depth(depth, depth_path)


def _load_depth_from_npy(depth_path: Path) -> np.ndarray:
    return _to_single_channel_depth(np.load(depth_path, allow_pickle=False), depth_path)


def _load_depth_from_exr(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to read EXR depth file: {depth_path}")
    return _to_single_channel_depth(depth, depth_path)


def _read_depth_array(depth_path: Path) -> np.ndarray:
    """Load a raw depth array from a supported prompt depth file."""
    ext = depth_path.suffix.lower()
    if ext == ".npz":
        return _load_depth_from_npz(depth_path)
    if ext in (".hdf5", ".h5"):
        return _load_depth_from_h5(depth_path)
    if ext == ".npy":
        return _load_depth_from_npy(depth_path)
    if ext == ".exr":
        return _load_depth_from_exr(depth_path)
    raise ValueError(f"Unsupported prompt depth extension `{ext}`: {depth_path}")


def load_depth(
    depth_path: Path,
    tar_size: tuple[int, int],
    num_samples: int = 1500,
    min_prompt: int = 1,
    max_prompt: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load dense depth and build a sparse prompt map for InfiniDepth."""
    depth = _read_depth_array(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth = cv2.resize(depth, tar_size[::-1], interpolation=cv2.INTER_NEAREST)

    depth_mask = ((depth > min_prompt) & (depth < max_prompt)).astype(np.float32)
    valid_depth = depth * depth_mask
    if (valid_depth > 0.1).sum() > num_samples:
        height, width = depth.shape
        sample_depth = valid_depth.reshape(-1)
        nonzero_index = np.flatnonzero(sample_depth > 0.1)
        index = np.random.permutation(nonzero_index)[:num_samples]
        sample_mask = np.ones_like(sample_depth)
        sample_mask[index] = 0.0
        sample_depth[sample_mask.astype(bool)] = 0.0
        sample_depth = sample_depth.reshape(height, width)
    else:
        sample_depth = valid_depth

    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    sample_depth_tensor = torch.from_numpy(sample_depth).unsqueeze(0).unsqueeze(0).float()
    depth_mask_tensor = torch.from_numpy(depth_mask).unsqueeze(0).unsqueeze(0)
    return depth_tensor, sample_depth_tensor, depth_mask_tensor


def load_prompt_depth_tensors(
    prompt_depth_path: Path,
    image_shape: tuple[int, int],
) -> PromptInputs:
    """Load one prompt depth file and build prompt tensors for InfiniDepth demo inference."""
    if not prompt_depth_path.exists():
        raise FileNotFoundError(f"Prompt depth file not found: {prompt_depth_path}")
    if not prompt_depth_path.is_file():
        raise ValueError(f"Prompt depth path is not a file: {prompt_depth_path}")

    _, prompt_depth_tensor, _ = load_depth(
        depth_path=prompt_depth_path,
        tar_size=image_shape,
    )

    prompt_depth = prompt_depth_tensor.squeeze(0).squeeze(0).numpy()
    prompt_mask = (prompt_depth > PROMPT_DISPARITY_MIN_DEPTH).astype(np.float32)
    valid_mask = prompt_mask > 0.0
    if not valid_mask.any():
        raise ValueError(
            f"Prompt depth file `{prompt_depth_path}` does not contain any valid prompt pixels after resizing."
        )

    prompt_disparity = np.zeros_like(prompt_depth, dtype=np.float32)
    prompt_disparity[valid_mask] = 1.0 / np.clip(
        prompt_depth[valid_mask],
        a_min=PROMPT_DEPTH_EPS,
        a_max=None,
    )

    return PromptInputs(
        prompt_disparity=torch.from_numpy(prompt_disparity).unsqueeze(0),
        prompt_mask=torch.from_numpy(prompt_mask).unsqueeze(0),
    )


def _requires_prompt_depth(cfg: RootCfg) -> bool:
    """Return whether the current encoder requires an explicit prompt depth input."""
    return cfg.model.encoder.name == "infinisplat_infinidepth"


def validate_prompt_configuration(cfg: RootCfg, prompt_depth_enabled: bool) -> bool:
    """Validate whether prompt inputs match the selected encoder configuration."""
    requires_prompt = _requires_prompt_depth(cfg)
    if requires_prompt and not prompt_depth_enabled:
        raise ValueError(
            "`infinisplat_infinidepth` demo inference requires `--prompt-depth` because the encoder uses prompt-conditioned InfiniDepth."
        )
    if not requires_prompt and prompt_depth_enabled:
        raise ValueError(
            "`--prompt-depth` is only supported when the resolved encoder is `infinisplat_infinidepth`."
        )
    return requires_prompt


def build_single_view_batch(
    image: torch.Tensor,
    intrinsics_px: torch.Tensor,
    device: torch.device | str = "cpu",
    prompt_inputs: PromptInputs | None = None,
) -> dict[str, torch.Tensor]:
    """Build the minimal single-view batch required by the encoder."""
    device = torch.device(device)
    _, height, width = image.shape
    intrinsics_norm = normalize_intrinsics(intrinsics_px, width, height)
    eye = torch.eye(4, dtype=torch.float32, device=device)
    context = {
        "image": rearrange(image.to(device=device, dtype=torch.float32), "c h w -> 1 1 c h w"),
        "intrinsics": rearrange(intrinsics_norm.to(device=device, dtype=torch.float32), "i j -> 1 1 i j"),
        "extrinsics": rearrange(eye, "i j -> 1 1 i j"),
    }
    if prompt_inputs is not None:
        context["prompt_disparity"] = rearrange(
            prompt_inputs.prompt_disparity.to(device=device, dtype=torch.float32),
            "c h w -> 1 1 c h w",
        )
        context["prompt_mask"] = rearrange(
            prompt_inputs.prompt_mask.to(device=device, dtype=torch.float32),
            "c h w -> 1 1 c h w",
        )
    return context


def load_demo_config(
    experiment_name: str,
) -> RootCfg:
    """Compose the Hydra config used for demo inference."""
    config_dir = Path(__file__).resolve().parents[2] / "config"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_dict = compose(
            config_name="inference",
            overrides=[
                f"+experiment={experiment_name}",
            ],
        )
    return load_typed_root_config(cfg_dict)


def _extract_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unexpected checkpoint format at {checkpoint_path}.")


def _load_prefixed_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    module_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not module_state:
        if len(module.state_dict()) == 0:
            return
        raise KeyError(f"Did not find any weights with prefix `{prefix}` in the checkpoint.")
    missing_keys, unexpected_keys = module.load_state_dict(module_state, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Unexpected state-dict mismatch for prefix `{prefix}`: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )


def load_demo_model(
    cfg: RootCfg,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Encoder, Decoder]:
    """Instantiate encoder/decoder and load checkpoint weights for demo inference."""
    encoder = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    state_dict = _extract_state_dict(checkpoint_path)
    _load_prefixed_state_dict(encoder, state_dict, "encoder.")
    _load_prefixed_state_dict(decoder, state_dict, "decoder.")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def load_demo_encoder(
    cfg: RootCfg,
    checkpoint_path: Path,
    device: torch.device,
) -> Encoder:
    """Instantiate the encoder and restore only its checkpoint weights."""
    encoder = get_encoder(cfg.model.encoder)
    state_dict = _extract_state_dict(checkpoint_path)
    _load_prefixed_state_dict(encoder, state_dict, "encoder.")
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.inference_mode()
def run_single_image_inference(
    encoder: Encoder,
    image: torch.Tensor,
    intrinsics_px: torch.Tensor,
    device: torch.device,
    prompt_inputs: PromptInputs | None = None,
) -> dict[str, torch.Tensor]:
    """Run encoder-only inference for one image."""
    context = build_single_view_batch(
        image=image,
        intrinsics_px=intrinsics_px,
        device=device,
        prompt_inputs=prompt_inputs,
    )
    return encoder(context)


def _filter_gaussians_by_mask(
    gaussians: Gaussians3D,
    keep_mask: torch.Tensor,
) -> Gaussians3D:
    """Select a single-image subset of Gaussians.

    Args:
        gaussians: Gaussian container with shape [1, N, ...].
        keep_mask: Boolean keep mask with shape [N].

    Returns:
        Filtered Gaussian container with shape [1, M, ...].
    """
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, keep_mask],
        singular_values=gaussians.singular_values[:, keep_mask],
        quaternions=gaussians.quaternions[:, keep_mask],
        colors=gaussians.colors[:, keep_mask],
        opacities=gaussians.opacities[:, keep_mask],
        covariances=gaussians.covariances[:, keep_mask] if gaussians.covariances is not None else None,
    )


@torch.inference_mode()
def filter_final_gaussian_floaters(gaussians: Gaussians3D) -> Gaussians3D:
    """Remove spatial outliers from final single-image Gaussians."""
    if gaussians.mean_vectors.shape[0] != 1:
        raise ValueError("Single-image floater filtering expects batch size 1.")

    total = int(gaussians.mean_vectors.shape[1])
    points = gaussians.mean_vectors[0]
    candidate_indices = torch.nonzero(
        torch.isfinite(points).all(dim=-1),
        as_tuple=False,
    ).flatten()
    if candidate_indices.numel() == 0:
        return gaussians

    candidate_points = points[candidate_indices].detach().float().cpu().numpy()
    if len(candidate_points) < 2:
        return gaussians
    neighbor_count = min(16, len(candidate_points) - 1)
    distances, _ = cKDTree(candidate_points).query(
        candidate_points,
        k=neighbor_count + 1,
        workers=-1,
    )
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = mean_distances.mean() + 2.5 * mean_distances.std()
    inlier_indices = np.flatnonzero(mean_distances <= threshold)

    keep_mask = torch.zeros(total, device=points.device, dtype=torch.bool)
    if inlier_indices.size > 0:
        inlier_indices_tensor = torch.as_tensor(
            inlier_indices,
            device=points.device,
            dtype=torch.long,
        )
        keep_mask[candidate_indices[inlier_indices_tensor]] = True
    if not keep_mask.any():
        return gaussians
    return _filter_gaussians_by_mask(gaussians, keep_mask)


def _normalize_vector(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def _build_look_at_c2w(
    eye_positions: torch.Tensor,
    look_at: torch.Tensor,
    world_up: torch.Tensor,
) -> torch.Tensor:
    """Construct OpenCV-style camera-to-world matrices from eye positions."""
    forward = _normalize_vector(look_at.unsqueeze(0) - eye_positions)
    right = torch.cross(forward, world_up.unsqueeze(0).expand_as(forward), dim=-1)
    near_parallel = right.norm(dim=-1) < 1e-6
    if near_parallel.any():
        fallback_up = torch.tensor([0.0, 0.0, -1.0], device=eye_positions.device, dtype=eye_positions.dtype)
        right = torch.where(
            near_parallel[:, None],
            torch.cross(forward, fallback_up.unsqueeze(0).expand_as(forward), dim=-1),
            right,
        )
    right = _normalize_vector(right)
    down = _normalize_vector(torch.cross(forward, right, dim=-1))

    c2w = repeat(
        torch.eye(4, dtype=eye_positions.dtype, device=eye_positions.device),
        "i j -> t i j",
        t=eye_positions.shape[0],
    ).clone()
    c2w[:, :3, 0] = right
    c2w[:, :3, 1] = down
    c2w[:, :3, 2] = forward
    c2w[:, :3, 3] = eye_positions
    return c2w


def create_demo_trajectory(
    gaussians: Gaussians3D,
    intrinsics_px: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a single-image novel-view trajectory."""
    device = gaussians.mean_vectors.device
    height, width = image_shape
    focal_length_px = float(intrinsics_px[0, 0].item())

    points = gaussians.mean_vectors[0].detach().to(device=device, dtype=torch.float32)
    valid_mask = torch.isfinite(points).all(dim=-1) & (points[:, 2] > 1e-4)
    if valid_mask.any():
        valid_points = points[valid_mask]
    else:
        valid_points = points[torch.isfinite(points).all(dim=-1)]
    if valid_points.numel() == 0:
        raise ValueError("Could not derive a valid trajectory because all gaussian centers are invalid.")

    look_at = valid_points.median(dim=0).values
    min_depth = torch.quantile(valid_points[:, 2], 0.1).clamp_min(1e-3)

    diagonal = math.sqrt((width / focal_length_px) ** 2 + (height / focal_length_px) ** 2)
    max_lateral_offset = 0.08 * diagonal * float(min_depth.item())
    max_medial_offset = 0.15 * float(min_depth.item())

    phase = torch.linspace(
        0.0,
        1.0,
        steps=DEFAULT_VIDEO_FRAMES,
        device=device,
        dtype=torch.float32,
    )
    eye_positions = torch.stack(
        [
            max_lateral_offset * torch.sin(2.0 * torch.pi * phase),
            torch.zeros(DEFAULT_VIDEO_FRAMES, device=device),
            max_medial_offset * (1.0 - torch.cos(2.0 * torch.pi * phase)) / 2.0,
        ],
        dim=-1,
    )

    c2w = _build_look_at_c2w(
        eye_positions=eye_positions,
        look_at=look_at,
        world_up=torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32),
    )
    extrinsics = torch.linalg.inv(c2w)
    intrinsics_norm = normalize_intrinsics(intrinsics_px, width, height).to(device=device, dtype=torch.float32)
    intrinsics_norm = repeat(intrinsics_norm, "i j -> 1 t i j", t=DEFAULT_VIDEO_FRAMES)
    return extrinsics.unsqueeze(0), intrinsics_norm


@torch.inference_mode()
def render_novel_view_video_from_single_view(
    decoder: Decoder,
    gaussians: Gaussians3D,
    render_intrinsics_px: torch.Tensor,
    render_image_shape: tuple[int, int],
    output_path: Path,
) -> Path:
    """Render an RGB novel-view video for a single-image gaussian scene."""
    device = gaussians.mean_vectors.device
    extrinsics, intrinsics_norm = create_demo_trajectory(
        gaussians=gaussians,
        intrinsics_px=render_intrinsics_px.to(device=device, dtype=torch.float32),
        image_shape=render_image_shape,
    )

    frames: list[torch.Tensor] = []
    num_frames = extrinsics.shape[1]
    for start in range(0, num_frames, DEFAULT_RENDER_CHUNK_SIZE):
        end = min(start + DEFAULT_RENDER_CHUNK_SIZE, num_frames)
        rendered = decoder.forward(
            gaussians=gaussians,
            extrinsics=extrinsics[:, start:end],
            intrinsics=intrinsics_norm[:, start:end],
            image_shape=render_image_shape,
        )
        frames.extend(frame.detach().cpu() for frame in rendered[0])

    save_video(frames, output_path, fps=DEFAULT_VIDEO_FPS)
    return output_path


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
    return checkpoint_path


def patch_supersplat_html_auto_rotate(html_path: Path) -> None:
    """Patch SuperSplat HTML so generated camera animation starts paused.

    Args:
        html_path: Generated SuperSplat HTML path.

    Returns:
        None.
    """
    source = html_path.read_text(encoding="utf-8")
    old = "noanim: url.searchParams.has('noanim'),"
    new = "noanim: true,"
    if new in source:
        return
    if old not in source:
        raise RuntimeError(
            "Could not pause SuperSplat HTML animation; the viewer bundle changed."
        )
    html_path.write_text(source.replace(old, new, 1), encoding="utf-8")


def patch_supersplat_html_viewer_bridge(
    html_path: Path,
    viewer_script_path: Path | None = None,
) -> None:
    """Report SuperSplat's first rendered frame to its embedding container."""
    source = html_path.read_text(encoding="utf-8")
    marker = "data-infinisplat-viewer-bridge"
    first_frame_hook = "window.firstFrame?.();"
    if marker in source:
        return
    hook_source = source
    if viewer_script_path is not None:
        hook_source += viewer_script_path.read_text(encoding="utf-8")
    if first_frame_hook not in hook_source or "</head>" not in source:
        raise RuntimeError(
            "Could not install the SuperSplat viewer bridge; the viewer bundle changed."
        )

    bridge = f"""
        <script {marker}>
            (() => {{
                let viewerState = "loading";
                const setViewerState = (state) => {{
                    if (viewerState === "ready" && state !== "ready") return;
                    viewerState = state;
                    const frame = window.frameElement;
                    const host = frame?.closest("[data-infinisplat-viewer]");
                    if (host) host.dataset.viewerState = state;
                    window.parent.postMessage(
                        {{ type: "infinisplat:viewer-state", state }},
                        "*"
                    );
                }};

                window.firstFrame = () => setViewerState("ready");
                window.addEventListener("error", () => setViewerState("error"));
                window.addEventListener(
                    "unhandledrejection",
                    () => setViewerState("error")
                );
                window.setTimeout(() => {{
                    if (viewerState !== "ready") setViewerState("timeout");
                }}, 60000);
            }})();
        </script>
"""
    html_path.write_text(
        source.replace("</head>", f"{bridge}\n    </head>", 1),
        encoding="utf-8",
    )
