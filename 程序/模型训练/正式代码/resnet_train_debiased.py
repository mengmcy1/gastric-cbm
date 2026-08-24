"""ResNet50 去偏重训练入口：固定 manifest/fold，共享 P6 增强与评估。"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from train_utils import create_training_parser, run_training


MODEL_SLUG = 'resnet50_debiased'
MODEL_NAME = 'ResNet50'
LABEL_SMOOTHING = 0.05


def build_model(pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_trainable_stage1(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def set_trainable_stage2(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
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
        stage2_description='layer4 + fc',
        label_smoothing=LABEL_SMOOTHING,
    )


if __name__ == '__main__':
    main()
