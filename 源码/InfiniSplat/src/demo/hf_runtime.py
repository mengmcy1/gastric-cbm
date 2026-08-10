from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from huggingface_hub import hf_hub_download

from src.demo.infer_single_image import (
    filter_final_gaussian_floaters,
    load_demo_config,
    load_demo_encoder,
    load_demo_image_bundle,
    patch_supersplat_html_auto_rotate,
    patch_supersplat_html_viewer_bridge,
    run_single_image_inference,
)
from src.model.encoder import Encoder
from src.utils.gaussians import Gaussians3D, save_ply

MODEL_REPO_ID = "PLUS-WAVE/InfiniSplat"
RGB_CHECKPOINT_FILE = "checkpoints/infinisplat_rgb.ckpt"
LOCAL_RGB_CHECKPOINT = Path(__file__).resolve().parents[2] / RGB_CHECKPOINT_FILE
RGB_EXPERIMENT = "infinisplat_hypersim_rgb"
SPLAT_TRANSFORM_PACKAGE = "@playcanvas/splat-transform@3.1.6"
VIEWER_SETTINGS = Path(__file__).resolve().parents[2] / "config" / "viewer_settings.json"
ARTIFACT_VERSION = 1
VIEWER_ASSET_FILENAMES = ("index.js", "index.css", "settings.json")


@dataclass(frozen=True)
class GaussianArtifact:
    """CPU-resident Gaussian tensors and camera metadata for one request."""

    gaussians: Gaussians3D
    focal_length_px: float
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class ExportedArtifacts:
    """Public output files generated for one request."""

    scene_ply: Path
    scene_sog: Path
    viewer_html: Path
    standalone_html: Path


@dataclass(frozen=True)
class BrowserViewerArtifacts:
    """Browser viewer files generated before standalone HTML bundling."""

    scene_sog: Path
    viewer_html: Path


@dataclass(frozen=True)
class ViewerTemplate:
    """Prebuilt viewer shell and content-addressed static assets."""

    viewer_html: Path
    viewer_assets_dir: Path


def _gaussians_to_dict(gaussians: Gaussians3D) -> dict[str, torch.Tensor | None]:
    return {
        "mean_vectors": gaussians.mean_vectors.detach().cpu(),
        "singular_values": gaussians.singular_values.detach().cpu(),
        "quaternions": gaussians.quaternions.detach().cpu(),
        "colors": gaussians.colors.detach().cpu(),
        "opacities": gaussians.opacities.detach().cpu(),
        "covariances": None if gaussians.covariances is None else gaussians.covariances.detach().cpu(),
    }


def save_gaussian_artifact(artifact: GaussianArtifact, path: Path) -> Path:
    """Serialize one request artifact using the safe torch data subset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": ARTIFACT_VERSION,
            "gaussians": _gaussians_to_dict(artifact.gaussians),
            "focal_length_px": artifact.focal_length_px,
            "image_shape": list(artifact.image_shape),
        },
        path,
    )
    return path


def load_gaussian_artifact(path: Path) -> GaussianArtifact:
    """Load an internal request artifact without permitting arbitrary objects."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["version"] != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported Gaussian artifact version: {payload['version']}")
    tensors = payload["gaussians"]
    gaussians = Gaussians3D(
        mean_vectors=tensors["mean_vectors"],
        singular_values=tensors["singular_values"],
        quaternions=tensors["quaternions"],
        colors=tensors["colors"],
        opacities=tensors["opacities"],
        covariances=tensors["covariances"],
    )
    return GaussianArtifact(
        gaussians=gaussians,
        focal_length_px=float(payload["focal_length_px"]),
        image_shape=tuple(int(value) for value in payload["image_shape"]),
    )


def _default_splat_transform_prefix() -> list[str]:
    override = os.environ.get("SPLAT_TRANSFORM")
    if override:
        return shlex.split(override)
    return ["npx", "--yes", "--prefer-offline", SPLAT_TRANSFORM_PACKAGE]


def build_splat_transform_command(
    scene_ply: Path,
    output_sog: Path,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build the fixed, non-interactive SOG conversion command."""
    prefix = list(command_prefix) if command_prefix is not None else _default_splat_transform_prefix()
    return [
        *prefix,
        "--quiet",
        "--overwrite",
        str(scene_ply),
        "--filter-harmonics",
        "0",
        str(output_sog),
    ]


def build_viewer_template_command(
    scene_ply: Path,
    output_html: Path,
    viewer_settings: Path = VIEWER_SETTINGS,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build the one-time command that emits the SuperSplat viewer shell."""
    prefix = list(command_prefix) if command_prefix is not None else _default_splat_transform_prefix()
    return [
        *prefix,
        "--quiet",
        "--overwrite",
        "--unbundled",
        "--viewer-settings",
        str(viewer_settings),
        str(scene_ply),
        "--filter-harmonics",
        "0",
        str(output_html),
    ]


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"Could not bundle SuperSplat {description}; the viewer template changed."
        )
    return source.replace(old, new, 1)


def build_standalone_viewer(
    viewer_html: Path,
    viewer_assets_dir: Path,
    output_html: Path,
) -> Path:
    """Bundle a shared-asset SuperSplat viewer for single-file download."""
    scene_sog = viewer_html.with_suffix(".sog")
    source = viewer_html.read_text(encoding="utf-8")
    css = (viewer_assets_dir / "index.css").read_text(encoding="utf-8")
    javascript = (viewer_assets_dir / "index.js").read_text(encoding="utf-8")
    settings = json.loads((viewer_assets_dir / "settings.json").read_text(encoding="utf-8"))
    encoded_scene = base64.b64encode(scene_sog.read_bytes()).decode("ascii")
    shared_prefix = Path(os.path.relpath(viewer_assets_dir, viewer_html.parent)).as_posix()

    source = _replace_once(
        source,
        f'<link rel="stylesheet" href="{shared_prefix}/index.css">',
        f"<style>\n{css}\n        </style>",
        "stylesheet",
    )
    source = _replace_once(
        source,
        f"import {{ main }} from '{shared_prefix}/index.js';",
        javascript,
        "script",
    )
    source = _replace_once(
        source,
        "settings: fetch(settingsUrl).then(response => response.json())",
        f"settings: {json.dumps(settings, separators=(',', ':'), ensure_ascii=False)}",
        "settings",
    )
    source = _replace_once(
        source,
        f'fetch("{scene_sog.name}")',
        f'fetch("data:application/octet-stream;base64,{encoded_scene}")',
        "scene data",
    )
    output_html.write_text(source, encoding="utf-8")
    return output_html


def install_shared_viewer_assets(viewer_html: Path) -> Path:
    """Point an unbundled viewer at content-addressed shared static assets."""
    asset_paths = [viewer_html.parent / name for name in VIEWER_ASSET_FILENAMES]
    hasher = hashlib.sha256()
    for asset_path in asset_paths:
        hasher.update(asset_path.name.encode("utf-8"))
        hasher.update(asset_path.read_bytes())
    digest = hasher.hexdigest()[:16]
    shared_dir = viewer_html.parent.parent / "_viewer_assets" / digest
    shared_dir.mkdir(parents=True, exist_ok=True)
    for asset_path in asset_paths:
        destination = shared_dir / asset_path.name
        if not destination.exists():
            shutil.copyfile(asset_path, destination)

    shared_prefix = f"../_viewer_assets/{digest}"
    source = viewer_html.read_text(encoding="utf-8")
    source = _replace_once(
        source,
        "./index.css",
        f"{shared_prefix}/index.css",
        "shared stylesheet path",
    )
    source = _replace_once(
        source,
        "./index.js",
        f"{shared_prefix}/index.js",
        "shared script path",
    )
    source = _replace_once(
        source,
        "./settings.json",
        f"{shared_prefix}/settings.json",
        "shared settings path",
    )
    viewer_html.write_text(source, encoding="utf-8")
    return shared_dir


def _viewer_bootstrap_gaussians() -> Gaussians3D:
    """Create one tiny Gaussian used only to build the shared viewer shell."""
    return Gaussians3D(
        mean_vectors=torch.tensor([[[0.0, 0.0, 1.0]]]),
        singular_values=torch.tensor([[[0.001, 0.001, 0.001]]]),
        quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        colors=torch.tensor([[[0.5, 0.5, 0.5]]]),
        opacities=torch.tensor([[0.01]]),
    )


def prepare_viewer_template(
    output_root: Path,
    command_prefix: Sequence[str] | None = None,
) -> ViewerTemplate:
    """Generate and cache the viewer shell before the first user request."""
    template_dir = output_root / "_viewer_template"
    template_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_ply = template_dir / "template.ply"
    viewer_html = template_dir / "template.html"
    save_ply(
        gaussians=_viewer_bootstrap_gaussians(),
        f_px=1.0,
        image_shape=(1, 1),
        path=bootstrap_ply,
    )
    converter_environment = os.environ.copy()
    xdg_runtime_dir = template_dir / ".xdg-runtime"
    xdg_runtime_dir.mkdir(mode=0o700, exist_ok=True)
    converter_environment["XDG_RUNTIME_DIR"] = str(xdg_runtime_dir)
    subprocess.run(
        build_viewer_template_command(
            scene_ply=bootstrap_ply,
            output_html=viewer_html,
            command_prefix=command_prefix,
        ),
        check=True,
        env=converter_environment,
    )
    patch_supersplat_html_auto_rotate(viewer_html)
    patch_supersplat_html_viewer_bridge(
        viewer_html,
        viewer_script_path=template_dir / "index.js",
    )
    viewer_assets_dir = install_shared_viewer_assets(viewer_html)
    return ViewerTemplate(
        viewer_html=viewer_html,
        viewer_assets_dir=viewer_assets_dir,
    )


def create_request_viewer(
    viewer_template: ViewerTemplate,
    scene_sog: Path,
    viewer_html: Path,
) -> Path:
    """Create a small request page that reuses the prebuilt viewer shell."""
    template_scene = viewer_template.viewer_html.with_suffix(".sog")
    source = viewer_template.viewer_html.read_text(encoding="utf-8")
    source = _replace_once(
        source,
        f'fetch("{template_scene.name}")',
        f'fetch("{scene_sog.name}")',
        "scene data path",
    )
    viewer_html.write_text(source, encoding="utf-8")
    return viewer_html


def export_filtered_gaussian_ply(
    artifact_path: Path,
    output_dir: Path,
) -> Path:
    """Filter spatial outliers and export the resulting Gaussian PLY."""
    artifact = load_gaussian_artifact(artifact_path)
    gaussians = filter_final_gaussian_floaters(artifact.gaussians)

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_ply = output_dir / "scene.ply"
    save_ply(
        gaussians=gaussians,
        f_px=artifact.focal_length_px,
        image_shape=artifact.image_shape,
        path=scene_ply,
    )
    return scene_ply


def export_browser_viewer(
    scene_ply: Path,
    viewer_template: ViewerTemplate,
    command_prefix: Sequence[str] | None = None,
) -> BrowserViewerArtifacts:
    """Convert one PLY into the optimized browser viewer files."""
    output_dir = scene_ply.parent
    scene_sog = output_dir / "viewer.sog"
    viewer_html = output_dir / "viewer.html"
    converter_environment = os.environ.copy()
    xdg_runtime_dir = output_dir / ".xdg-runtime"
    xdg_runtime_dir.mkdir(mode=0o700, exist_ok=True)
    converter_environment["XDG_RUNTIME_DIR"] = str(xdg_runtime_dir)
    subprocess.run(
        build_splat_transform_command(
            scene_ply=scene_ply,
            output_sog=scene_sog,
            command_prefix=command_prefix,
        ),
        check=True,
        env=converter_environment,
    )
    create_request_viewer(
        viewer_template=viewer_template,
        scene_sog=scene_sog,
        viewer_html=viewer_html,
    )
    return BrowserViewerArtifacts(
        scene_sog=scene_sog,
        viewer_html=viewer_html,
    )


def export_standalone_viewer(
    viewer_html: Path,
    viewer_template: ViewerTemplate,
) -> Path:
    """Bundle one browser viewer into a directly downloadable HTML file."""
    return build_standalone_viewer(
        viewer_html=viewer_html,
        viewer_assets_dir=viewer_template.viewer_assets_dir,
        output_html=viewer_html.with_name("scene.html"),
    )


def export_gaussian_artifact(
    artifact_path: Path,
    output_dir: Path,
    viewer_template: ViewerTemplate,
    command_prefix: Sequence[str] | None = None,
) -> ExportedArtifacts:
    """Export filtered Gaussians and both viewer formats."""
    scene_ply = export_filtered_gaussian_ply(
        artifact_path=artifact_path,
        output_dir=output_dir,
    )
    browser_viewer = export_browser_viewer(
        scene_ply=scene_ply,
        viewer_template=viewer_template,
        command_prefix=command_prefix,
    )
    standalone_html = export_standalone_viewer(
        viewer_html=browser_viewer.viewer_html,
        viewer_template=viewer_template,
    )
    return ExportedArtifacts(
        scene_ply=scene_ply,
        scene_sog=browser_viewer.scene_sog,
        viewer_html=browser_viewer.viewer_html,
        standalone_html=standalone_html,
    )


def resolve_rgb_checkpoint() -> Path:
    """Resolve an override, reuse a repository checkpoint, or download it."""
    local_checkpoint = os.environ.get("INFINISPLAT_CHECKPOINT")
    if local_checkpoint:
        checkpoint_path = Path(local_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if LOCAL_RGB_CHECKPOINT.is_file():
        return LOCAL_RGB_CHECKPOINT

    return Path(
        hf_hub_download(
            repo_id=os.environ.get("INFINISPLAT_MODEL_REPO", MODEL_REPO_ID),
            filename=RGB_CHECKPOINT_FILE,
            revision=os.environ.get("INFINISPLAT_MODEL_REVISION", "main"),
        )
    )


class InfiniSplatRuntime:
    """One process-wide RGB encoder reused by all web requests."""

    def __init__(self, encoder: Encoder, device: torch.device) -> None:
        self.encoder = encoder
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | None = None,
        device: str | torch.device | None = None,
    ) -> "InfiniSplatRuntime":
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cfg = load_demo_config(RGB_EXPERIMENT)
        encoder = load_demo_encoder(
            cfg=cfg,
            checkpoint_path=checkpoint_path or resolve_rgb_checkpoint(),
            device=resolved_device,
        )
        return cls(encoder=encoder, device=resolved_device)

    @torch.inference_mode()
    def infer_to_artifact(self, image_path: Path, artifact_path: Path) -> Path:
        """Run one RGB reconstruction and persist CPU tensors for post-processing."""
        image_bundle = load_demo_image_bundle(image_path=image_path)
        encoder_output = run_single_image_inference(
            encoder=self.encoder,
            image=image_bundle.inference_image,
            intrinsics_px=image_bundle.inference_intrinsics.intrinsics_px,
            device=self.device,
        )
        _, height, width = image_bundle.inference_image.shape
        artifact = GaussianArtifact(
            gaussians=encoder_output["gaussians"].to("cpu"),
            focal_length_px=image_bundle.inference_intrinsics.focal_length_px,
            image_shape=(height, width),
        )
        return save_gaussian_artifact(artifact, artifact_path)
