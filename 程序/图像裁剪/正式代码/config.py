#!/usr/bin/env python3
"""gastric-cbm 图像裁剪候选流程的集中路径配置。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


MODEL_PATH = _path_from_env(
    "GASTRIC_FOV_MODEL_PATH",
    MODULE_ROOT / "模型" / "shiye_V1.onnx",
)

# 默认路径只作为清晰的占位约定。正式运行应显式传入--input和--output，
# 避免在新数据目录尚未冻结时误扫其他数据版本。
INPUT_ROOT = _path_from_env(
    "GASTRIC_CROP_INPUT_ROOT",
    PROJECT_ROOT / "数据" / "待裁剪原图",
)
OUTPUT_ROOT = _path_from_env(
    "GASTRIC_CROP_OUTPUT_ROOT",
    PROJECT_ROOT / "数据整理记录" / "新数据" / "图像裁剪候选_v1",
)
STAGE1_OUTPUT = _path_from_env(
    "GASTRIC_CROP_STAGE1_OUTPUT",
    OUTPUT_ROOT / "01_亮度四边裁剪",
)
STAGE1_CROPS = _path_from_env(
    "GASTRIC_CROP_STAGE1_CROPS",
    STAGE1_OUTPUT / "crops",
)
STAGE2_OUTPUT = _path_from_env(
    "GASTRIC_CROP_STAGE2_OUTPUT",
    OUTPUT_ROOT / "02_FOV遮罩",
)
STAGE2_MASKED = _path_from_env(
    "GASTRIC_CROP_STAGE2_MASKED",
    STAGE2_OUTPUT / "masked",
)
STAGE2_MASKS = _path_from_env(
    "GASTRIC_CROP_STAGE2_MASKS",
    STAGE2_OUTPUT / "masks",
)
STAGE2_MAPPING = _path_from_env(
    "GASTRIC_CROP_STAGE2_MAPPING",
    STAGE2_OUTPUT / "mapping.csv",
)

# 画中画保留/notch消融只改变画中画像素，不改变图像尺寸和bbox坐标。
PIP_IMAGE_ROOT = _path_from_env(
    "GASTRIC_PIP_IMAGE_ROOT",
    STAGE1_CROPS,
)
STAGE3_OUTPUT = _path_from_env(
    "GASTRIC_PIP_DETECTION_OUTPUT",
    OUTPUT_ROOT / "03_画中画筛查",
)
STAGE3_QUALITY_FLAGS = STAGE3_OUTPUT / "quality_flags.csv"
STAGE3_PIP_LABELS = STAGE3_OUTPUT / "confirmed_pip_labels.csv"
STAGE3_VALIDATION_OUTPUT = OUTPUT_ROOT / "00_debug" / "画中画检测验证"
STAGE3_VALIDATION_LIST = OUTPUT_ROOT / "00_debug" / "画中画验证清单.csv"
STAGE4_OUTPUT = _path_from_env(
    "GASTRIC_PIP_BRANCH_OUTPUT",
    OUTPUT_ROOT / "04_画中画Notch候选",
)
PIP_VALIDATION_LABELS = OUTPUT_ROOT / "人工记录" / "validation_labels.csv"
PIP_OVERRIDES = OUTPUT_ROOT / "人工记录" / "pip_rect_overrides.csv"
