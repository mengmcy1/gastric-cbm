import torch.nn as nn

from src.model.encoder.depth.infinidepth.implicit_promptda_simple.prompt_models.selfattn import (
    SelfAttnPromptModel,
)

__all__ = [
    "GeneralPromptModel",
    "SelfAttnPromptModel",
]


class GeneralPromptModel(nn.Module):
    def __init__(self, prompt_stage=[3], **kwargs):
        super().__init__()
        self.prompt_stage = prompt_stage
        self.prompt_idmap = {i: idx for idx, i in enumerate(self.prompt_stage)}
        block = kwargs.get("block")
        self.prompt_model = nn.ModuleList([block for _ in range(len(self.prompt_stage))])

    def forward(self, features, prompt_depth, prompt_mask, patch_h, patch_w):
        for i in range(len(features)):
            if i not in self.prompt_stage: # prompt_stage = [3]
                continue
            features[i][0] = self.prompt_model[self.prompt_idmap[i]](
                features[i][0],
                prompt_depth,
                prompt_mask,
                patch_h,
                patch_w,
            )
        return features
