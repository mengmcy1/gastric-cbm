"""EfficientNet-B0 迁移学习：先训练分类头，再微调高层特征。"""

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from train_utils import run_training, seed_everything


DEBUG = False
DEBUG_SAMPLES = 200

BATCH_SIZE = 32
NUM_WORKERS = 0 if DEBUG else 4
RANDOM_SEED = 42

STAGE1_EPOCHS = 2 if DEBUG else 10
STAGE2_EPOCHS = 2 if DEBUG else 20
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def main():
    seed_everything(RANDOM_SEED)
    model = build_model()
    run_training(
        model=model,
        stage1_modules=[model.classifier],
        stage2_modules=[model.features[5:], model.classifier],
        stage1_name='冻结 backbone，仅训练 classifier',
        stage2_name='微调 features[5:] + classifier',
        save_filename='efficientnet_b0_best.pth',
        model_name='EfficientNet-B0',
        label_smoothing=0.1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED,
        stage1_epochs=STAGE1_EPOCHS,
        stage2_epochs=STAGE2_EPOCHS,
        stage1_lr=STAGE1_LR,
        stage2_lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY,
        early_stop_patience=EARLY_STOP_PATIENCE,
        debug=DEBUG,
        debug_samples=DEBUG_SAMPLES,
    )


if __name__ == '__main__':
    main()
