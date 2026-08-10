"""Contains basic data structures and functionality for 3D Gaussians."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from src.utils import linalg
from src.utils.color_space import (
    encode_color_space,
    linearRGB2sRGB,
)

class Gaussians3D(NamedTuple):
    """Represents a collection of 3D Gaussians."""

    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    covariances: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "Gaussians3D":
        """Move Gaussians to device."""
        return Gaussians3D(
            mean_vectors=self.mean_vectors.to(device),
            singular_values=self.singular_values.to(device),
            quaternions=self.quaternions.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
            covariances=self.covariances.to(device) if self.covariances is not None else None,
        )


def get_unprojection_matrix(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute unprojection matrix to transform Gaussians to Euclidean space.

    Args:
        extrinsics: The 4x4 extrinsics matrix of the camera view.
        intrinsics: The 4x4 intrinsics matrix of the camera view.
        image_shape: The (width, height) of the input image.

    Returns:
        A 4x4 matrix to transform Gaussians from NDC space to Euclidean space.
    """
    device = intrinsics.device
    dtype = intrinsics.dtype
    image_width, image_height = image_shape

    if intrinsics.shape[-2:] == (3, 3):
        intrinsics_4x4 = torch.eye(4, device=device, dtype=dtype).expand(
            *intrinsics.shape[:-2], 4, 4
        ).clone()
        intrinsics_4x4[..., :3, :3] = intrinsics
        intrinsics = intrinsics_4x4

    ndc_matrix = torch.tensor(
        [
            [2.0 / image_width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / image_height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    # linalg.inv does not support low-precision dtypes (bf16/fp16); compute in fp32.
    # Keep the result in fp32 — geometry must not downcast.
    matrix = (ndc_matrix @ intrinsics @ extrinsics).float()
    return torch.linalg.inv(matrix)


def unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    """Unproject Gaussians from NDC space to world coordinates.

    Args:
        gaussians_ndc: Flattened per-view Gaussians with shape [B, V*N, ...].
        extrinsics: World-to-camera matrices with shape [B, V, 4, 4].
        intrinsics: Camera intrinsics with shape [B, V, 3, 3] or [B, V, 4, 4].
        image_shape: Image shape (width, height).
    Returns:
        World-space Gaussians with shape [B, V*N, ...].
    """
    num_views = extrinsics.shape[-3]
    num_gaussians = gaussians_ndc.mean_vectors.shape[1]
    if num_gaussians % num_views != 0:
        raise ValueError(
            f"Expected flattened Gaussian count {num_gaussians} to be divisible by num_views {num_views}."
        )

    gaussians_per_view = num_gaussians // num_views
    gaussians_grouped = Gaussians3D(
        mean_vectors=gaussians_ndc.mean_vectors.reshape(
            gaussians_ndc.mean_vectors.shape[0],
            num_views,
            gaussians_per_view,
            3,
        ),
        singular_values=gaussians_ndc.singular_values.reshape(
            gaussians_ndc.singular_values.shape[0],
            num_views,
            gaussians_per_view,
            3,
        ),
        quaternions=gaussians_ndc.quaternions.reshape(
            gaussians_ndc.quaternions.shape[0],
            num_views,
            gaussians_per_view,
            4,
        ),
        colors=gaussians_ndc.colors.reshape(
            gaussians_ndc.colors.shape[0],
            num_views,
            gaussians_per_view,
            3,
        ),
        opacities=gaussians_ndc.opacities.reshape(
            gaussians_ndc.opacities.shape[0],
            num_views,
            gaussians_per_view,
        ),
        covariances=gaussians_ndc.covariances.reshape(
            gaussians_ndc.covariances.shape[0],
            num_views,
            gaussians_per_view,
            3,
            3,
        ) if gaussians_ndc.covariances is not None else None,
    )

    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = apply_transform(
        gaussians_grouped,
        unprojection_matrix[..., :3, :],
    )
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors.reshape(
            gaussians.mean_vectors.shape[0],
            num_gaussians,
            3,
        ),
        singular_values=gaussians.singular_values.reshape(
            gaussians.singular_values.shape[0],
            num_gaussians,
            3,
        ),
        quaternions=gaussians.quaternions.reshape(
            gaussians.quaternions.shape[0],
            num_gaussians,
            4,
        ),
        colors=gaussians.colors.reshape(
            gaussians.colors.shape[0],
            num_gaussians,
            3,
        ),
        opacities=gaussians.opacities.reshape(
            gaussians.opacities.shape[0],
            num_gaussians,
        ),
        covariances=gaussians.covariances.reshape(
            gaussians.covariances.shape[0],
            num_gaussians,
            3, 3,
        ) if gaussians.covariances is not None else None,
    )


def apply_transform(
    gaussians: Gaussians3D,
    transform: torch.Tensor,
) -> Gaussians3D:
    """Apply an affine transformation to 3D Gaussians.

    Args:
        gaussians: The Gaussians to transform.
        transform: An affine transform with shape [..., 3, 4].
    Returns:
        The transformed Gaussians. World-space covariance matrices are stored in
        the ``covariances`` field and are fully differentiable. ``quaternions``
        and ``singular_values`` remain in NDC form because rendering consumes
        ``covariances`` directly. Export performs decomposition separately.
    """
    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians.mean_vectors @ transform_linear.transpose(-1, -2)
    mean_vectors = mean_vectors + transform_offset[..., None, :]

    # Differentiable covariance transform: M @ Sigma_ndc @ M^T
    covariance_matrices = (
        gaussians.covariances
        if gaussians.covariances is not None
        else compose_covariance_matrices(
            gaussians.quaternions,
            gaussians.singular_values,
        )
    )
    world_covariances = (
        transform_linear.unsqueeze(-3)
        @ covariance_matrices
        @ transform_linear.unsqueeze(-3).transpose(-1, -2)
    )

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=gaussians.singular_values,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
        covariances=world_covariances,
    )


def decompose_covariance_matrices(
    covariance_matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose 3D covariance matrices into quaternions and singular values.

    Args:
        covariance_matrices: The covariance matrices to decompose.

    Returns:
        Quaternion and singular values corresponding to the orientation and scales of
        the diagonalized matrix.

    Note:
        This operation is not differentiable.
    """
    eigval_eps = 1e-12
    device = covariance_matrices.device
    dtype = covariance_matrices.dtype
    batch_shape = covariance_matrices.shape[:-2]

    covariance_matrices = covariance_matrices.detach().cpu().to(torch.float64)
    covariance_matrices = covariance_matrices.reshape(-1, 3, 3)
    covariance_matrices = 0.5 * (
        covariance_matrices + covariance_matrices.transpose(-1, -2)
    )

    eigvals, eigvecs = torch.linalg.eigh(covariance_matrices)
    sort_idx = torch.argsort(eigvals, dim=-1, descending=True)
    eigvals = torch.gather(eigvals, -1, sort_idx)
    eigvecs = torch.gather(
        eigvecs,
        -1,
        sort_idx.unsqueeze(-2).expand(-1, 3, 3),
    )
    eigvals = eigvals.clamp_min(eigval_eps)

    det = torch.linalg.det(eigvecs)
    reflection_idx = torch.where(det < 0)[0]
    if reflection_idx.numel() > 0:
        eigvecs[reflection_idx, :, -1] *= -1

    rotations = eigvecs.reshape(batch_shape + (3, 3))
    singular_values = eigvals.sqrt().reshape(batch_shape + (3,))

    quaternions = linalg.quaternions_from_rotation_matrices(rotations)
    quaternions = quaternions.to(dtype=dtype, device=device)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    quaternions = canonicalize_quaternions(quaternions)
    singular_values = singular_values.to(dtype=dtype, device=device)
    return quaternions, singular_values


def compose_covariance_matrices(
    quaternions: torch.Tensor,
    singular_values: torch.Tensor,
) -> torch.Tensor:
    """Compose 3D covariance matrices into quaternions and singular values.

    Args:
        quaternions: The quaternions describing the principal basis.
        singular_values: The scales of the diagonalized matrix.

    Returns:
        The 3x3 covariance matrices.
    """
    device = quaternions.device
    rotations = linalg.rotation_matrices_from_quaternions(quaternions)
    diagonal_matrix = torch.eye(3, device=device, dtype=quaternions.dtype) * singular_values[..., :, None]
    return rotations @ diagonal_matrix.square() @ rotations.transpose(-1, -2)


def canonicalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Canonicalize quaternion signs for viewer-stable export.

    Args:
        quaternions: Quaternions in wxyz order with shape [..., 4].

    Returns:
        Quaternions with the largest-magnitude component forced to be non-negative.
    """
    largest_idx = quaternions.abs().argmax(dim=-1, keepdim=True)
    signs = torch.gather(quaternions, -1, largest_idx).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return quaternions * signs


def prepare_gaussians_for_ply_export(gaussians: Gaussians3D) -> Gaussians3D:
    """Convert Gaussians into SuperSplat-compatible q+s parameters for PLY export.

    Args:
        gaussians: Gaussians with shape [B, N, ...]. If ``covariances`` is
            available, it is treated as the source of truth and decomposed into a
            canonical world-space q+s representation.

    Returns:
        Gaussians with viewer-compatible world-space q+s in ``wxyz`` order.
    """
    if gaussians.covariances is not None:
        quaternions, singular_values = decompose_covariance_matrices(gaussians.covariances)
    else:
        singular_values = gaussians.singular_values.clamp_min(1e-8)
        quaternions = gaussians.quaternions
        quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        quaternions = canonicalize_quaternions(quaternions)

    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors,
        singular_values=singular_values.clamp_min(1e-8),
        quaternions=quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
        covariances=gaussians.covariances,
    )


def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to degree-0 spherical harmonics.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return (rgb - 0.5) / coeff_degree0


@torch.no_grad()
def save_ply(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
    path: Path,
) -> PlyData:
    """Save a predicted Gaussian3D to a ply file."""

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    gaussians = prepare_gaussians_for_ply_export(gaussians)

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)
    colors = convert_rgb_to_spherical_harmonics(
        linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

    attributes = torch.cat(
        (
            xyz,
            colors,
            opacity_logits,
            scale_logits,
            quaternions,
        ),
        dim=1,
    )
    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")

    image_height, image_width = image_shape

    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)
    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([encode_color_space("sRGB")]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    plydata.write(path)
    return plydata
