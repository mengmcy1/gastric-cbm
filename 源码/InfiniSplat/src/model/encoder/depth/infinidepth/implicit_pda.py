import os
import torch
import torch.nn as nn
import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from src.model.encoder.depth.infinidepth.implicit_promptda_simple.config import (
    dinov3_model_configs,
    model_configs,
)
from src.model.encoder.depth.infinidepth.implicit_promptda_simple.low_level_implicit import (
    LowLevelImplicitHead,
)
from src.model.encoder.depth.infinidepth.implicit_promptda_simple.prompt_models import (
    GeneralPromptModel,
    SelfAttnPromptModel,
)
from src.model.encoder.depth.infinidepth.implicit_promptda_simple.convolution import BasicEncoder
from src.model.encoder.depth.infinidepth.warp_utils import WarpMedian
from typing import Optional

EPS = 1e-6
acc_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _clear_torchhub_package(package_name: str) -> None:
    stale_modules = [
        module_name
        for module_name in list(sys.modules)
        if module_name == package_name or module_name.startswith(f"{package_name}.")
    ]
    for module_name in stale_modules:
        sys.modules.pop(module_name, None)


class InfiniDepth(nn.Module):
    use_bn = False
    use_clstoken = False

    def __init__(self,
                 model_path: Optional[str] = None,
                 geometry_type: str = "disparity",
                 encoder: str = "vitl16",  # dinov2 -> vitl  dinov3 -> vitl16
                 backbone: str = "dinov3",
                 use_prompt=True,
                 ):
        super().__init__()
        if backbone != "dinov3":
            model_config = model_configs[encoder]
        else:
            model_config = dinov3_model_configs[encoder]
        self.model_config = model_config
        self.use_prompt = use_prompt

        # Learnable module definitions
        # backbone
        if backbone == "dinov2":
            raise NotImplementedError("Dinov2 backbone is not implemented in this version.")
        elif backbone == "dinov3":
            _clear_torchhub_package("dinov3")
            self.pretrained = torch.hub.load(
                "src/model/encoder/depth/infinidepth/implicit_promptda_simple/torchhub/dinov3",
                f"dinov3_{encoder}",  # vitl16, vith16plus, vit7b16
                source="local",
                pretrained=False,
            )
            _clear_torchhub_package("dinov3")
            self.patch_size = 16

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.basic_encoder = BasicEncoder(
            input_dim=3,
            output_dim=128,
            stride=4
        )
        
        # prompt
        if use_prompt:
            self.prompt_model = GeneralPromptModel(prompt_stage=[3], block=SelfAttnPromptModel(num_blocks=4, pe="qk"))
            self.warp_func = WarpMedian()
        else:
            poly_features = PolynomialFeatures(degree=1, include_bias=False)
            ransac = RANSACRegressor(max_trials=1000)
            self.align_model = make_pipeline(poly_features, ransac)
            self.scale = 1.0
            self.shift = 0.0

        # implicit depth decoder head
        self.depth_implicit_head = LowLevelImplicitHead(                
            hidden_dim=dim,
            basic_dim=128,  # BasicEncoder output dim
            fusion_type="concat",  # concat, gated
            out_dim=1,
            hidden_list=[1024, 256, 32],
        )

        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.geometry_type = geometry_type

        if model_path is not None:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                self.load_state_dict({k[9:]: v for k, v in checkpoint["state_dict"].items()})
            else:
                raise FileNotFoundError(f"Model file {model_path} not found")

        # The wrapper owns device placement and train/eval mode.

    @torch.no_grad() 
    def inference(self,
                image: torch.Tensor,
                query_coord: torch.Tensor,
                prompt_depth: torch.Tensor,
                prompt_mask=None,
                use_batch_infer=True,
                align_mode: str = "fit_and_apply"):
        if prompt_mask is None:
            prompt_mask = prompt_depth > 0
        
        if self.use_prompt:
            prompt_depth, prompt_mask, reference_meta = self.warp_func.warp(
                prompt_depth,
                prompt_depth=prompt_depth,
                prompt_mask=prompt_mask,
            )

        if use_batch_infer:
            pred, dino_feat, basic_feat = self.batch_forward(image, query_coord, prompt_depth, prompt_mask, bsize=100000)
        else:
            pred, dino_feat, basic_feat = self.forward(image, query_coord, prompt_depth, prompt_mask)

        if self.use_prompt:
            pred = self.warp_func.unwarp(
                pred,
                reference_meta=reference_meta[...,0],
            )
        else:
            if align_mode == "fit_and_apply":
                pred = self._ransac_align_depth(pred, prompt_depth, prompt_mask, update_params=True).to(image.device)
            elif align_mode == "apply_cached":
                pred = self._apply_scale_shift(pred, self.scale, self.shift)
            elif align_mode == "none":
                pass
            else:
                raise ValueError(f"Unknown align_mode: {align_mode}")

        if self.geometry_type == "depth":
            pred_depth = pred
            pred_disparity = 1.0 / torch.clamp(pred, min=5e-3)
        elif self.geometry_type == "disparity":
            pred_disparity = pred
            pred_depth = 1.0 / torch.clamp(pred, min=5e-3)

        return pred_depth, pred_disparity, dino_feat, basic_feat

    def batch_forward(self, x, coord, prompt_depth=None, prompt_mask=None, bsize=3000):
        """
        Forward pass with batching to avoid OOM.
        """
        h, w = x.shape[-2:]

        # DINOv3 branch (semantic features) - uses ImageNet normalization
        x_dino = (x - self._mean) / self._std
        with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=self.model_config["layer_idxs"],
                return_class_token=True,
            )
        dino_feat = features[-1][0].clone()
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        
        if self.use_prompt:
            features = self.prompt_model(features, prompt_depth, prompt_mask, patch_h, patch_w)

        # BasicEncoder branch (low-level features) - uses [-1, 1] normalization
        x_basic = 2.0 * x - 1.0
        basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]
  
        # generate feature map for learning implicit function
        feat = self.depth_implicit_head.encode_feat(features, patch_h, patch_w)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            # batch querying
            pred = self.depth_implicit_head.decode_dpt(feat, basic_feat, coord[:, ql: qr, :], cell=None)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        return pred, dino_feat, basic_feat

    def forward(self, x, coords, prompt_depth, prompt_mask):
        h, w = x.shape[-2:]

        # DINOv3 branch (semantic features) - uses ImageNet normalization
        x_dino = (x - self._mean) / self._std
        with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=self.model_config["layer_idxs"],
                return_class_token=True,
            )
        dino_feat = features[-1][0].clone()
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        if self.use_prompt:
            features = self.prompt_model(features, prompt_depth, prompt_mask, patch_h, patch_w)

        # BasicEncoder branch (low-level features) - uses [-1, 1] normalization
        x_basic = 2.0 * x - 1.0
        basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            depth = self.depth_implicit_head(features, basic_feat, patch_h, patch_w, coords, cell=None)

        return depth, dino_feat, basic_feat

    def _apply_scale_shift(self, pred, scale, shift):
        if type(pred).__module__ == torch.__name__:
            return pred * float(scale) + float(shift)

        pred_np = np.asarray(pred, dtype=np.float32)
        return pred_np * float(scale) + float(shift)

    def _ransac_align_depth(self, pred, gt, mask0=None, geometry_type=None, update_params=True):
        if geometry_type is None:
            geometry_type = self.geometry_type

        if type(pred).__module__ == torch.__name__:
            pred = pred.detach().cpu().numpy()
        if type(gt).__module__ == torch.__name__:
            gt = gt.detach().cpu().numpy()

        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)

        # Keep the expected tensor layout as (B, N, C) and avoid squeeze-induced shape changes.
        if pred.ndim == 1:
            pred = pred[None, :, None]
        elif pred.ndim == 2:
            pred = pred[:, :, None]
        elif pred.ndim != 3:
            raise ValueError(f"Expected pred with 1/2/3 dims, but got shape {pred.shape}")

        bsz = pred.shape[0]

        if gt.ndim == 0:
            gt_flat = gt.reshape(1, 1)
        elif gt.ndim == 1:
            gt_flat = gt.reshape(1, -1)
        else:
            gt_flat = gt.reshape(gt.shape[0], -1)

        mask_flat = None
        if mask0 is not None:
            if type(mask0).__module__ == torch.__name__:
                mask0 = mask0.detach().cpu().numpy()
            mask0 = np.asarray(mask0)
            if mask0.ndim == 0:
                mask_flat = (mask0.reshape(1, 1) > 0)
            elif mask0.ndim == 1:
                mask_flat = (mask0.reshape(1, -1) > 0)
            else:
                mask_flat = (mask0.reshape(mask0.shape[0], -1) > 0)

        if gt_flat.shape[0] == 1 and bsz > 1:
            gt_flat = np.repeat(gt_flat, bsz, axis=0)
        if mask_flat is not None and mask_flat.shape[0] == 1 and bsz > 1:
            mask_flat = np.repeat(mask_flat, bsz, axis=0)

        if gt_flat.shape[0] != bsz:
            bsz = min(bsz, gt_flat.shape[0])
            pred = pred[:bsz]
            gt_flat = gt_flat[:bsz]
            if mask_flat is not None:
                mask_flat = mask_flat[:bsz]
        elif mask_flat is not None and mask_flat.shape[0] != bsz:
            bsz = min(bsz, mask_flat.shape[0])
            pred = pred[:bsz]
            gt_flat = gt_flat[:bsz]
            mask_flat = mask_flat[:bsz]

        pred_metric = pred.copy()
        scales = []
        shifts = []

        for bi in range(bsz):
            pred_vec = pred[bi, :, 0].reshape(-1)
            gt_vec = gt_flat[bi].reshape(-1)
            n = min(pred_vec.shape[0], gt_vec.shape[0])

            curr_mask = None
            if mask_flat is not None:
                n = min(n, mask_flat[bi].shape[0])
                curr_mask = mask_flat[bi][:n]

            scale = 1.0
            shift = 0.0
            if n >= 2:
                pred_n = pred_vec[:n]
                gt_n = gt_vec[:n]
                valid = gt_n > 1e-8
                if curr_mask is not None:
                    valid = valid & curr_mask

                if valid.sum() >= 2:
                    gt_mask = gt_n[valid].astype(np.float32)
                    pred_mask = pred_n[valid].astype(np.float32)

                    if geometry_type == "disparity":
                        gt_mask = np.clip(gt_mask, 1e-8, None)
                    if geometry_type == "depth":
                        gt_mask = np.log(gt_mask + 1.0)

                    try:
                        self.align_model.fit(pred_mask[:, None], gt_mask[:, None])
                        a = self.align_model.named_steps['ransacregressor'].estimator_.coef_.item()
                        b = self.align_model.named_steps['ransacregressor'].estimator_.intercept_.item()
                    except Exception:
                        a, b = 1.0, 0.0

                    if a > 0:
                        scale, shift = a, b
                    else:
                        pred_mean = np.mean(pred_mask)
                        gt_mean = np.mean(gt_mask)
                        scale = gt_mean / (pred_mean + EPS)
                        shift = 0.0

            pred_metric[bi, :n, :] = self._apply_scale_shift(pred[bi, :n, :], scale, shift)
            scales.append(float(scale))
            shifts.append(float(shift))

        if update_params and not self.use_prompt and len(scales) > 0:
            self.scale = float(np.mean(scales))
            self.shift = float(np.mean(shifts))

        return torch.from_numpy(pred_metric)
