from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .depth_pro import DepthProConfig, create_model_and_transforms


def _inference_precision(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def _as_depth_map(depth: torch.Tensor, batch_size: int) -> torch.Tensor:
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth with 2 or 3 dims, got shape {tuple(depth.shape)}")
    if depth.shape[0] != batch_size:
        raise ValueError(
            f"Depth batch mismatch: expected {batch_size}, got {depth.shape[0]}"
        )
    return depth.unsqueeze(1)


def _as_normalized_intrinsics_batch(
    intrinsics: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if intrinsics is None:
        return None
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected normalized intrinsics with shape [B,3,3] or [3,3], got {tuple(intrinsics.shape)}"
        )
    if intrinsics.shape[0] == 1 and batch_size > 1:
        intrinsics = intrinsics.expand(batch_size, -1, -1)
    if intrinsics.shape[0] != batch_size:
        raise ValueError(
            f"Intrinsics batch mismatch: expected {batch_size}, got {intrinsics.shape[0]}"
        )
    return intrinsics


class DepthPro(nn.Module):
    """InfiniSplat wrapper around the vendored upstream DepthPro package."""

    def __init__(self):
        super().__init__()
        self.device_hint = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = _inference_precision(self.device_hint)

        depthpro_cfg = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            decoder_features=256,
            checkpoint_uri=None,
            fov_encoder_preset="dinov2l16_384",
            use_fov_head=True,
        )
        self.model, self.transform = create_model_and_transforms(
            config=depthpro_cfg,
            device=self.device_hint,
            precision=self.precision,
        )

        self.model.eval()

    def _prepare_input(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B,3,H,W], got shape {tuple(image.shape)}")
        self._ensure_runtime_dtype()
        model_device = next(self.model.parameters()).device
        image = image.to(device=model_device, dtype=torch.float32)
        image = image.clamp(0.0, 1.0)
        image = 2.0 * image - 1.0
        return image.to(dtype=self.precision)

    def _ensure_runtime_dtype(self) -> None:
        param = next(self.model.parameters())
        desired_dtype = _inference_precision(param.device)
        if param.dtype != desired_dtype:
            self.model.to(dtype=desired_dtype)
            self.precision = desired_dtype

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run DepthPro with the standardized InfiniSplat depth-only interface.

        Args:
            batch: Dictionary containing:
                - image: Tensor with shape [B, 3, H, W].
                - intrinsics: Optional normalized intrinsics with shape [B, 3, 3]
                  or [3, 3]. When provided, fx is converted to pixel-space focal
                  length and passed to DepthPro. If missing, DepthPro falls back
                  to its internal FoV / focal estimation path.

        Returns:
            Dense metric depth with shape [B, 1, H, W].
        """
        image = batch["image"]
        prepared = self._prepare_input(image)
        batch_size = prepared.shape[0]

        focal_px: torch.Tensor | None = None
        intrinsics = _as_normalized_intrinsics_batch(
            batch.get("intrinsics"),
            batch_size,
            prepared.device,
        )
        if intrinsics is not None:
            focal_px = intrinsics[:, 0, 0] * float(image.shape[-1])

        with torch.no_grad():
            prediction = self.model.infer(prepared, f_px=focal_px)

        return _as_depth_map(prediction["depth"], batch_size).to(dtype=torch.float32)
