from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image

from stage01_common import REPO_ROOT, file_hash, prepare_output_dir, read_json, write_json


EXPECTED_MODEL_SHA256 = "D960C4E6B3266B6B9FA74EE4458A9482160D54C06D7738696BC9A9E2B34C66DC"
EXPECTED_MAT_COMMIT = "d273d891ecdad2e1df106516423a75bc45b2d800"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用官方 MAT Places-512 权重执行固定端点补全。")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--mat-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=240)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_file(entry: dict, label: str) -> Path:
    path = resolve_repo_path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    observed = file_hash(path)
    if observed != entry["sha256"].upper():
        raise ValueError(f"{label} SHA256 不匹配：{observed} != {entry['sha256']}")
    return path


def copy_params_and_buffers(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    src_tensors = dict(src.named_parameters()) | dict(src.named_buffers())
    for name, tensor in list(dst.named_parameters()) + list(dst.named_buffers()):
        if name not in src_tensors:
            raise KeyError(f"MAT 权重缺少参数：{name}")
        tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("experiment_id") != "P01_ang30_static_inpaint_ab_v1":
        raise ValueError("拒绝使用未冻结的静态 A/B 清单。")
    endpoint = manifest["endpoints"][args.side]
    image_path = verify_file(endpoint["image"], f"{args.side} image")
    mask_path = verify_file(endpoint["mask"], f"{args.side} mask")

    mat_root = args.mat_root.resolve()
    model_path = args.model.resolve()
    if not (mat_root / "legacy.py").is_file():
        raise FileNotFoundError(f"MAT 源码不完整：{mat_root}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_sha256 = file_hash(model_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError(f"MAT 权重 SHA256 不匹配：{model_sha256}")
    sys.path.insert(0, str(mat_root))
    import legacy  # type: ignore
    from networks.mat import Generator  # type: ignore

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用。")

    image = np.asarray(Image.open(image_path).convert("RGB"))
    hole_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if image.shape[:2] != hole_mask.shape:
        raise ValueError("RGB 与掩码分辨率不一致。")
    height, width = image.shape[:2]
    if height % 512 or width % 512:
        raise ValueError("原生 MAT 输入要求宽高均为 512 的整数倍；本脚本不隐式缩放。")

    output_dir = prepare_output_dir(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    load_start = time.perf_counter()
    with model_path.open("rb") as handle:
        saved = legacy.load_network_pkl(handle)["G_ema"]
    generator = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=512,
        img_channels=3,
    ).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(saved, generator)
    del saved
    load_seconds = time.perf_counter() - load_start

    image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)[None]
    image_tensor = image_tensor.to(device) / 127.5 - 1.0
    # MAT 官方约定：1=已知像素、0=待补区域；项目清单的白色区域表示待补，因此这里取反。
    known_mask = torch.from_numpy((~hole_mask).astype(np.float32))[None, None].to(device)
    z = torch.from_numpy(np.random.randn(1, generator.z_dim)).to(device)
    label = torch.zeros((1, generator.c_dim), device=device)

    inference_start = time.perf_counter()
    with torch.inference_mode():
        prediction = generator(
            image_tensor,
            known_mask,
            z,
            label,
            truncation_psi=1.0,
            noise_mode="random",
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start

    raw = ((prediction[0].permute(1, 2, 0) + 1.0) * 127.5)
    raw = raw.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    composited = image.copy()
    composited[hole_mask] = raw[hole_mask]
    if not np.array_equal(composited[~hole_mask], image[~hole_mask]):
        raise AssertionError("MAT 输出改写了掩码外已知像素。")
    iio.imwrite(output_dir / "inpaint_raw.png", raw)
    iio.imwrite(output_dir / "inpaint_composited.png", composited)

    write_json(
        output_dir / "mat_run.json",
        {
            "schema_version": "1.0-mat-inpaint",
            "status": "success",
            "experiment_id": manifest["experiment_id"],
            "side": args.side,
            "seed": args.seed,
            "source": {
                "repository": "https://github.com/fenglinglwb/MAT",
                "commit": EXPECTED_MAT_COMMIT,
                "compatibility_patch": "scripts/mat_pytorch29_custom_ops_loader.patch",
            },
            "model": {
                "path": str(model_path),
                "sha256": model_sha256,
                "name": "Places_512_FullData.pkl",
                "provenance": "third-party mirror of the released MAT checkpoint; original OneDrive link returned 404",
                "mirror": "https://huggingface.co/Icar/mat_places512_full",
            },
            "inputs": {
                "image": str(image_path),
                "image_sha256": endpoint["image"]["sha256"],
                "mask": str(mask_path),
                "mask_sha256": endpoint["mask"]["sha256"],
                "hole_fraction": float(hole_mask.mean()),
            },
            "preprocessing": {
                "mode": "native_no_resize_no_pad",
                "resolution_hw": [height, width],
                "mat_mask_semantics": "1=known, 0=hole",
                "noise_mode": "random",
                "truncation_psi": 1.0,
            },
            "runtime": {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "model_load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                ),
            },
            "output": {
                "raw_sha256": file_hash(output_dir / "inpaint_raw.png"),
                "composited_sha256": file_hash(output_dir / "inpaint_composited.png"),
                "known_pixels_exact": True,
            },
        },
    )


if __name__ == "__main__":
    main()
