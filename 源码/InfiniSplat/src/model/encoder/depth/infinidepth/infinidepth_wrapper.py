from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from src.model.encoder.depth.infinidepth.implicit_pda import (
    InfiniDepth as InfiniDepthModel,
)
from src.model.encoder.depth.infinidepth.sampling_utils import make_2d_uniform_coord


class InfiniDepth(nn.Module):
    """Prompt-conditioned InfiniDepth dense-depth predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.model = InfiniDepthModel(
            model_path=None,
            geometry_type="disparity",
            use_prompt=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a dense depth map from RGB and sparse disparity prompts."""
        required_keys = ("image", "prompt_disparity", "prompt_mask")
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise AssertionError(
                "InfiniDepth requires prompt-conditioned inputs. "
                f"Missing keys: {missing_keys}"
            )

        image = batch["image"]
        batch_size, _, height, width = image.shape
        query_coords = make_2d_uniform_coord((height, width)).to(image.device)
        query_coords = query_coords.unsqueeze(0).expand(batch_size, -1, -1)
        depth, _, _, _ = self.model.inference(
            image=image,
            query_coord=query_coords,
            prompt_depth=batch["prompt_disparity"],
            prompt_mask=batch["prompt_mask"],
        )
        return rearrange(depth, "b (h w) c -> b c h w", h=height, w=width)
