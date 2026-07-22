"""EfficientNet-B0 去偏重训练入口：固定 manifest/fold，共享 P6 增强与评估。"""

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from train_utils import create_training_parser, run_training


MODEL_SLUG = 'efficientnet_b0_debiased'
MODEL_NAME = 'EfficientNet-B0'
LABEL_SMOOTHING = 0.10


def build_model(pretrained=True):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def set_trainable_stage1(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_trainable_stage2(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for index in range(5, len(model.features)):
        for parameter in model.features[index].parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def main():
    parser = create_training_parser(MODEL_NAME)
    args = parser.parse_args()
    run_training(
        args=args,
        model_slug=MODEL_SLUG,
        model_display_name=MODEL_NAME,
        build_model=build_model,
        set_trainable_stage1=set_trainable_stage1,
        set_trainable_stage2=set_trainable_stage2,
        stage2_description='features[5:] + classifier',
        label_smoothing=LABEL_SMOOTHING,
    )


if __name__ == '__main__':
    main()
