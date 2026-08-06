from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stage01_common import file_hash, prepare_output_dir, write_json


EXPECTED_MD5 = "E3AA4AAA15225A33EC84F9F4BC47E500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Big-LaMa TorchScript 完成端点显露区域 RGB。")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--skip-md5-check", action="store_true")
    return parser.parse_args()


def resize_for_model(image: np.ndarray, mask: np.ndarray, max_side: int):
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width)) if max_side > 0 else 1.0
    target_w = max(8, round(width * scale))
    target_h = max(8, round(height * scale))
    target_w = max(8, (target_w // 8) * 8)
    target_h = max(8, (target_h // 8) * 8)
    image_small = np.asarray(
        Image.fromarray(image).resize((target_w, target_h), Image.Resampling.LANCZOS)
    )
    mask_small = np.asarray(
        Image.fromarray(mask).resize((target_w, target_h), Image.Resampling.NEAREST)
    )
    return image_small, mask_small, scale


def main() -> None:
    args = parse_args()
    for path in (args.image, args.mask, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用。")
    observed_md5 = file_hash(args.model, "md5")
    if not args.skip_md5_check and observed_md5 != EXPECTED_MD5:
        raise ValueError(f"Big-LaMa MD5 不匹配：{observed_md5} != {EXPECTED_MD5}")

    output_dir = prepare_output_dir(args.output_dir)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L"))
    if image.shape[:2] != mask.shape:
        raise ValueError("RGB 与掩码分辨率不一致。")
    mask_binary = mask > 0
    image_small, mask_small, scale = resize_for_model(image, mask, args.max_side)

    device = torch.device(args.device)
    image_tensor = torch.from_numpy(image_small.copy()).float().permute(2, 0, 1)[None] / 255.0
    mask_tensor = torch.from_numpy((mask_small > 0).astype(np.float32))[None, None]
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    # Windows 的 LibTorch 文件接口无法可靠处理含中文字符的路径；
    # 通过 Python 已打开的二进制流加载，避免把 Unicode 路径交给 C++ fopen。
    with args.model.open("rb") as model_stream:
        model = torch.jit.load(model_stream, map_location=device).eval()
    load_seconds = time.perf_counter() - load_start
    warm_start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(image_tensor, mask_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - warm_start

    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    prediction = prediction[0].clamp(0, 1)
    prediction = F.interpolate(
        prediction[None], size=image.shape[:2], mode="bilinear", align_corners=False
    )[0]
    raw = (prediction.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    composited = image.copy()
    composited[mask_binary] = raw[mask_binary]
    iio.imwrite(output_dir / "inpaint_raw.png", raw)
    iio.imwrite(output_dir / "inpaint_composited.png", composited)

    boundary = mask_binary ^ np.asarray(
        Image.fromarray(mask_binary.astype(np.uint8) * 255).filter(
            __import__("PIL.ImageFilter", fromlist=["MaxFilter"]).MaxFilter(5)
        )
    ).astype(bool)
    seam_l1 = (
        float(np.abs(raw.astype(np.float32) - image.astype(np.float32))[boundary].mean())
        if boundary.any()
        else 0.0
    )
    write_json(
        output_dir / "lama_run.json",
        {
            "schema_version": "1.0-lama-inpaint",
            "model": {
                "path": str(args.model.resolve()),
                "format": "TorchScript",
                "md5": observed_md5,
                "sha256": file_hash(args.model, "sha256"),
                "provenance": "IOPaint/Sanster Big-LaMa deployment conversion; not the official training checkpoint",
            },
            "inputs": {
                "image": str(args.image.resolve()),
                "mask": str(args.mask.resolve()),
                "mask_fraction": float(mask_binary.mean()),
            },
            "device": str(device),
            "source_resolution_hw": [int(image.shape[0]), int(image.shape[1])],
            "model_resolution_hw": [int(image_small.shape[0]), int(image_small.shape[1])],
            "resize_scale": scale,
            "timing_seconds": {"model_load": load_seconds, "inference": inference_seconds},
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "boundary_raw_vs_source_l1_0_255": seam_l1,
            "note": "仅掩码内部采用模型输出；掩码外像素保持逐值不变。",
        },
    )


if __name__ == "__main__":
    main()
