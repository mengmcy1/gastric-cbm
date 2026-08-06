from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("无法从脚本位置定位项目根目录。")


REPO_ROOT = find_repo_root()
SHARP_SRC = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
if str(SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(SHARP_SRC))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def prepare_output_dir(path: Path, allow_nonempty: bool = False) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing and not allow_nonempty:
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{path}（{len(existing)} 项）")
    return path


def configure_sharp_cuda_toolkit() -> None:
    import os

    conda_prefix = Path(sys.prefix)
    cuda_toolkit = conda_prefix / "Library"
    cuda_bin = cuda_toolkit / "bin"
    if (cuda_bin / "nvcc.exe").is_file():
        os.environ.setdefault("CUDA_HOME", str(cuda_toolkit))
        os.environ.setdefault("CUDA_PATH", str(cuda_toolkit))
        os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(cuda_bin))


def create_camera_context(ply_path: Path):
    configure_sharp_cuda_toolkit()
    from sharp.utils import camera
    from sharp.utils.gaussians import load_ply

    gaussians, metadata = load_ply(ply_path)
    width, height = map(int, metadata.resolution_px)
    f_px = float(metadata.focal_length_px)
    intrinsics = torch.tensor(
        [
            [f_px, 0.0, (width - 1) / 2.0, 0.0],
            [0.0, f_px, (height - 1) / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px, lookat_mode="point"
    )
    return gaussians, metadata, model


def endpoint_angle_deg(angle_total_deg: float, side: str) -> float:
    if side == "left":
        return -float(angle_total_deg) / 2.0
    if side == "right":
        return float(angle_total_deg) / 2.0
    if side == "center":
        return 0.0
    raise ValueError(f"未知视角：{side}")


def eye_position(focus_depth: float, angle_deg: float, trajectory_mode: str) -> torch.Tensor:
    angle_rad = math.radians(float(angle_deg))
    if trajectory_mode == "legacy_lateral":
        # 与 sharp_angle_protocol_render.py v1 完全一致：z=0 平面横移。
        return torch.tensor(
            [focus_depth * math.tan(angle_rad), 0.0, 0.0], dtype=torch.float32
        )
    if trajectory_mode == "true_arc":
        # 真圆弧且中心帧保持原相机：圆心/注视点位于 (0,0,focus_depth)，
        # 半径必须等于 focus_depth，才能同时满足中心 eye=(0,0,0)。
        return torch.tensor(
            [
                focus_depth * math.sin(angle_rad),
                0.0,
                focus_depth * (1.0 - math.cos(angle_rad)),
            ],
            dtype=torch.float32,
        )
    raise ValueError(f"未知轨迹模式：{trajectory_mode}")


def camera_info_for_angle(camera_model, angle_deg: float, trajectory_mode: str):
    eye = eye_position(float(camera_model.depth_quantiles.focus), angle_deg, trajectory_mode)
    return eye, camera_model.compute(eye)


def angle_sequence(angle_total_deg: float, num_steps: int) -> np.ndarray:
    if num_steps < 2:
        raise ValueError("num_steps 至少为 2。")
    return np.linspace(-float(angle_total_deg) / 2.0, float(angle_total_deg) / 2.0, num_steps)
