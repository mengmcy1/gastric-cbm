#!/usr/bin/env python3
"""为训练/验证图像的两种翻转状态构建绑定SHA的MG2教师缓存。

MG2教师输出只取决于图像和水平翻转状态，因此本脚本提前计算冻结MG1b教师
对每张图两种状态的分类输出与注意力。处理顺序与训练一致：先翻转完整图并
同步变换裁剪框，再提取灰度教师ROI。缓存记录教师权重和清单SHA，B/C组必须
加载并校验同一缓存，A组不读取缓存。本脚本只读取训练和验证队列。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256
from train_mage_mg1_teacher import DEFAULT_MANIFEST, DEFAULT_V3_AUDIT, load_manifest
from train_mage_mg2_student import (
    CACHE_FORMAT,
    DEFAULT_DEBUG_TEACHER_CACHE,
    DEFAULT_TEACHER_CACHE,
    DEFAULT_TEACHER_CHECKPOINT,
    FROZEN_TEACHER_SHA256,
    GRID_SIZE,
    IMAGE_SIZE,
    flip_box_horizontal,
    load_frozen_teacher,
    teacher_roi_tensor,
    validate_teacher_cache,
)


def parse_args() -> argparse.Namespace:
    """解析缓存构建输入、输出路径和调试模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v3-audit", type=Path, default=DEFAULT_V3_AUDIT)
    parser.add_argument("--teacher-checkpoint", type=Path,
                        default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None,
                        help="缺省为正式缓存路径；--debug时缺省为debug缓存路径")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=3)
    return parser.parse_args()


def build_entries(
    frame,
    model,
    device: torch.device,
    batch_size: int,
) -> dict:
    """对每行图像的两种翻转状态运行冻结教师。

    参数:
        frame (pd.DataFrame): train+val清单行（debug时为debug子集），
            行序即缓存 ``order`` 的逐图对齐基准。
        model (AttentionPoolingTeacher): 已核验SHA的冻结MG1b教师，
            eval()且requires_grad=False。
        device (torch.device): 前向设备。
        batch_size (int): 前向batch大小。
    返回:
        dict: ``{sha256: {"flip0"/"flip1": {"logits": Tensor[2],
            "attention": Tensor[49]}}}``；attention为float32、行优先
            (y,x)展开，逐条目校验总和为1。
    """
    entries = {}
    tensors = {state: [] for state in ("flip0", "flip1")}
    for index, row in enumerate(frame.itertuples(index=False)):
        with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
            image = handle.convert("RGB")
        crop = np.array([
            row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
        ], dtype=np.float64)
        # 与训练时完全同序：先翻转完整图并同步变换crop_box，再裁教师ROI。
        tensors["flip0"].append(teacher_roi_tensor(image, crop))
        tensors["flip1"].append(
            teacher_roi_tensor(TF.hflip(image), flip_box_horizontal(crop))
        )
        if (index + 1) % 200 == 0:
            print(f"已读取 {index + 1}/{len(frame)} 张（每图x2翻转状态）")

    total = len(frame)
    for state in ("flip0", "flip1"):
        stacked = torch.stack(tensors[state])
        for start in range(0, total, batch_size):
            block = stacked[start:start + batch_size].to(device)
            with torch.inference_mode():
                logits, attention = model(block)
            attention = attention.reshape(len(block), GRID_SIZE * GRID_SIZE).cpu()
            logits = logits.cpu()
            for offset in range(len(block)):
                sha = str(frame.iloc[start + offset].sha256)
                entries.setdefault(sha, {})[state] = {
                    "logits": logits[offset].float(),
                    "attention": attention[offset].float(),
                }
        done = sum(1 for entry in entries.values() if state in entry)
        print(f"{state}: 完成 {done}/{total} 张教师前向")
    return entries


def main() -> None:
    """构建、自校验并保存MG2教师缓存。"""
    args = parse_args()
    manifest_sha = file_sha256(args.manifest.resolve())
    frame, _ = load_manifest(
        args.manifest.resolve(), args.v3_audit.resolve(), args.debug,
        args.debug_patients_per_class, 42,
    )
    if not set(frame.split.unique()).issubset({"train", "val"}):
        raise ValueError("MG2教师缓存只允许train/val；manifest含其他split时必须显式排除")
    frame = frame.reset_index(drop=True)

    output = args.output
    if output is None:
        output = DEFAULT_DEBUG_TEACHER_CACHE if args.debug else DEFAULT_TEACHER_CACHE
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"教师缓存已存在，拒绝覆盖；请先人工归档后再重建: {output}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了--device cuda，但当前PyTorch无法使用CUDA")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    model, teacher_sha = load_frozen_teacher(args.teacher_checkpoint.resolve(), device)
    print(
        f"设备={device}; 图片={len(frame)}张 x2翻转状态; "
        f"教师SHA={teacher_sha[:12]}...; manifest SHA={manifest_sha[:12]}..."
    )
    entries = build_entries(frame, model, device, args.batch_size)
    cache = {
        "format": CACHE_FORMAT,
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": teacher_sha,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "grid_size": GRID_SIZE,
        "image_size": IMAGE_SIZE,
        "debug": bool(args.debug),
        "order": [str(value) for value in frame.sha256],
        "entries": entries,
    }
    # 保存前按训练侧的同一校验函数做逐图对齐与总和自检。
    validate_teacher_cache(cache, frame, manifest_sha, args.debug)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"教师缓存完成: {output} (sha256={file_sha256(output)})")


if __name__ == "__main__":
    main()
