#!/usr/bin/env python3
"""RP-D热图缩放、独立图像资产和面板渲染纯函数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont, ImageOps


def positive_q99(values: np.ndarray) -> tuple[float, str]:
    """计算完整train正激活Q99；全零时返回固定状态。"""
    positive = np.asarray(values, dtype=np.float32)
    positive = positive[positive > 0]
    if positive.size == 0:
        return 0.0, "all_zero"
    return float(np.quantile(positive, 0.99)), "ok"


def display_response(raw: np.ndarray, q99: float) -> np.ndarray:
    """按冻结Q99缩放7x7响应，禁止逐图min-max。"""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (7, 7):
        raise ValueError(f"RP-D raw map必须为7x7: {raw.shape}")
    if q99 == 0:
        return np.zeros((7, 7), dtype=np.float32)
    return np.clip(raw / float(q99), 0, 1).astype(np.float32)


def resize_preserving_aspect(image: Image.Image, max_side: int) -> Image.Image:
    """保持纵横比，将显示资产最长边限制到固定像素。"""
    image = image.convert("RGB")
    scale = min(1.0, float(max_side) / max(image.size))
    size = tuple(max(1, int(round(value * scale))) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def render_heatmap(response: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """将已冻结缩放的响应绘制为turbo色图。"""
    upsampled = Image.fromarray(np.uint8(np.clip(response, 0, 1) * 255)).resize(
        size, Image.Resampling.BILINEAR,
    )
    values = np.asarray(upsampled, dtype=np.float32) / 255.0
    colored = colormaps["turbo"](values)[..., :3]
    return Image.fromarray(np.uint8(np.clip(colored, 0, 1) * 255), mode="RGB")


def render_overlay(
    original: Image.Image, heatmap: Image.Image, response: np.ndarray, max_alpha: float,
) -> Image.Image:
    """按响应强度叠加热图；原图本身不写入技术文字。"""
    alpha_map = Image.fromarray(np.uint8(np.clip(response, 0, 1) * 255)).resize(
        original.size, Image.Resampling.BILINEAR,
    )
    alpha = (np.asarray(alpha_map, dtype=np.float32) / 255.0 * float(max_alpha))[..., None]
    base = np.asarray(original, dtype=np.float32)
    color = np.asarray(heatmap, dtype=np.float32)
    mixed = base * (1 - alpha) + color * alpha
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 255)), mode="RGB")


def save_case_assets(
    image_path: Path,
    raw_map: np.ndarray,
    q99: float,
    output_dir: Path,
    max_side: int,
    overlay_alpha: float,
    save_original: bool,
) -> dict[str, Path]:
    """保存独立original/raw/heatmap/overlay资产。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(image_path) as source:
        original = resize_preserving_aspect(source, max_side)
    response = display_response(raw_map, q99)
    heatmap = render_heatmap(response, original.size)
    overlay = render_overlay(original, heatmap, response, overlay_alpha)
    paths = {
        "raw_7x7": output_dir / "raw_7x7.npy",
        "heatmap": output_dir / "heatmap.png",
        "overlay": output_dir / "overlay.png",
    }
    np.save(paths["raw_7x7"], np.asarray(raw_map, dtype=np.float32))
    heatmap.save(paths["heatmap"])
    overlay.save(paths["overlay"])
    if save_original:
        paths["original"] = output_dir.parent / "original.png"
        original.save(paths["original"])
    return paths


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    """载入资产并按白底完整缩放到面板格。"""
    with Image.open(path) as source:
        return ImageOps.pad(source.convert("RGB"), size, color="white", method=Image.Resampling.LANCZOS)


def placeholder(size: tuple[int, int], text: str) -> Image.Image:
    """生成固定缺失病例占位块。"""
    image = Image.new("RGB", size, (242, 242, 242))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    x = (size[0] - (box[2] - box[0])) // 2
    y = (size[1] - (box[3] - box[1])) // 2
    draw.multiline_text((x, y), text, fill=(70, 70, 70), font=font, align="center")
    return image
