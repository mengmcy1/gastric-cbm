"""Triangle-mesh sampling utilities for InfiniSplat."""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

SAMPLE_KIND_TRIANGLE_VERTEX = 0
SAMPLE_KIND_EXTRA_FACE = 1
SAMPLE_KIND_PIXEL_CENTER = SAMPLE_KIND_TRIANGLE_VERTEX
SAMPLE_KIND_VERTEX = SAMPLE_KIND_TRIANGLE_VERTEX
SAMPLE_KIND_FACE = SAMPLE_KIND_EXTRA_FACE

_ANCHOR_GRID_STRIDE = 2
_ANCHOR_BUDGET_RATIO = 0.65
_IMAGE_DETAIL_WEIGHT = 0.7
_DEPTH_DETAIL_WEIGHT = 0.3
_DETAIL_SCORE_QUANTILE = 0.9


class SparseSamplingOutput(NamedTuple):
    """Sparse sampling coordinates plus per-sample density metadata.

    Args:
        coords_yx_ndc: Sparse sample coordinates in YX NDC order with shape [..., N, 2].
        sample_responsibility_area_metric: Metric surface area represented by each sample
            with shape [..., N].
        sample_kind: Sample source ids with shape [..., N]. `0` means
            triangle-vertex support and `1` means triangle-face support.
    """

    coords_yx_ndc: torch.Tensor
    sample_responsibility_area_metric: torch.Tensor
    sample_kind: torch.Tensor


class SurfaceMesh(NamedTuple):
    """Depth-induced triangle mesh used by vertex and face samplers.

    Args:
        depth_hw: Dense metric depth with shape [H, W].
        vertices_flat: Camera-space mesh vertices with shape [H*W, 3].
        valid_vertex_mask: Valid depth vertex mask with shape [H, W].
        faces: Pruned triangle face indices with shape [F, 3].
        face_count_initial: Number of candidate faces before pruning.
        face_count_after_prune: Number of faces after discontinuity pruning.
    """

    depth_hw: torch.Tensor
    vertices_flat: torch.Tensor
    valid_vertex_mask: torch.Tensor
    faces: torch.Tensor
    face_count_initial: int
    face_count_after_prune: int


def make_2d_uniform_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    query_coords = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        query_coords = query_coords.view(-1, query_coords.shape[-1])
    return query_coords


def _depth_to_vertices(D, fx, fy, cx, cy):
    """Project a depth map to camera-space vertices.

    Args:
        D: Depth map with shape [H, W].
        fx, fy, cx, cy: Pixel-space camera intrinsics.

    Returns:
        Camera-space vertices with shape [H, W, 3].
    """
    h, w = D.shape
    device = D.device
    js = torch.arange(w, device=device, dtype=torch.float32)
    is_ = torch.arange(h, device=device, dtype=torch.float32)
    jj, ii = torch.meshgrid(js, is_, indexing="xy")

    Z = D
    X = (jj - cx) / fx * Z
    Y = (ii - cy) / fy * Z
    return torch.stack([X, Y, Z], dim=-1)


def _build_faces(h, w, device):
    """Return triangle faces with shape [2 * (H - 1) * (W - 1), 3]."""
    idx = torch.arange(h * w, device=device).reshape(h, w)
    f1 = torch.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], dim=-1).reshape(-1, 3)
    f2 = torch.stack([idx[1:, 1:], idx[:-1, 1:], idx[1:, :-1]], dim=-1).reshape(-1, 3)
    return torch.cat([f1, f2], dim=0)


def _prune_faces(Vflat, faces, depth_ratio=1.05, max_edge=None, depth_ratio_far=1.10):
    """Prune triangles that likely cross depth discontinuities or are too large."""
    A = Vflat[faces[:, 0]]
    B = Vflat[faces[:, 1]]
    C = Vflat[faces[:, 2]]

    zA, zB, zC = A[:, 2], B[:, 2], C[:, 2]
    zmin = torch.min(torch.min(zA, zB), zC)
    zmax = torch.max(torch.max(zA, zB), zC)

    zmean = (zA + zB + zC) / 3.0
    log_z = torch.log10(zmean.clamp(min=1.0))
    alpha = torch.clamp(log_z / 2.0, 0.0, 1.0)
    adaptive_ratio = depth_ratio + (depth_ratio_far - depth_ratio) * alpha

    keep = (zmin > 0) & (zmax / torch.clamp(zmin, min=1e-9) < adaptive_ratio)

    if max_edge is not None:
        e0 = torch.norm(B - A, dim=1)
        e1 = torch.norm(C - B, dim=1)
        e2 = torch.norm(A - C, dim=1)
        keep &= torch.max(torch.max(e0, e1), e2) < max_edge

    return faces[keep]


def _faces_to_ij(faces, h, w):
    """Convert flattened face vertex indices to row/column coordinates."""
    i = faces // w
    j = faces % w
    return i, j


def _normalize_ij_coords(
    i_s: torch.Tensor,
    j_s: torch.Tensor,
    h: int,
    w: int,
    coord_norm: str,
) -> torch.Tensor:
    if coord_norm == "zero_one":
        x = (j_s + 0.5) / w
        y = (i_s + 0.5) / h
    else:
        x = 2.0 * ((j_s + 0.5) / w) - 1.0
        y = 2.0 * ((i_s + 0.5) / h) - 1.0
    return torch.stack([y, x], dim=-1)


def _compute_base_valid_mask(depth_hw: torch.Tensor) -> torch.Tensor:
    """Compute valid vertex mask shared by face filtering and diagnostics."""
    return torch.isfinite(depth_hw) & (depth_hw > 0)


def _compute_extra_face_weights(
    faces: torch.Tensor,
    Vflat: torch.Tensor,
    valid_pixel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute orig extra-face weights and metric areas.

    Args:
        faces: Triangle indices with shape [F, 3].
        Vflat: Camera-space vertices with shape [H*W, 3].
        valid_pixel_mask: Valid depth mask with shape [H, W].

    Returns:
        A tuple of face weights and metric face areas, each with shape [F].
    """
    A = Vflat[faces[:, 0]]
    B = Vflat[faces[:, 1]]
    C = Vflat[faces[:, 2]]

    cross_product = torch.cross(B - A, C - A, dim=-1)
    areas = 0.5 * torch.norm(cross_product, dim=-1)
    areas = torch.clamp(areas, min=0.0)

    valid_depths = Vflat[valid_pixel_mask.reshape(-1), 2]
    if valid_depths.numel() == 0:
        raise RuntimeError("No valid base depths available for extra-face weighting.")

    z_mean = (A[:, 2] + B[:, 2] + C[:, 2]) / 3.0
    z_ref = torch.median(valid_depths).clamp(min=1e-6)
    depth_scale = torch.clamp(z_mean / z_ref, min=0.25, max=4.0)
    weights = areas * depth_scale
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights, areas


def _build_surface_mesh(
    *,
    depth_hw: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_ratio: float,
    max_edge: Optional[float],
) -> SurfaceMesh:
    """Build and prune a depth-induced triangle mesh.

    Args:
        depth_hw: Dense metric depth with shape [H, W].
        fx, fy, cx, cy: Pixel-space camera intrinsics.
        depth_ratio: Near-depth discontinuity pruning ratio for faces.
        max_edge: Optional maximum 3D edge length for faces.

    Returns:
        Surface mesh containing valid vertices and pruned triangle faces.
    """
    h, w = depth_hw.shape
    device = depth_hw.device
    valid_vertex_mask = _compute_base_valid_mask(depth_hw=depth_hw)
    vertices = _depth_to_vertices(depth_hw, fx, fy, cx, cy)
    vertices_flat = vertices.reshape(-1, 3)
    faces = _build_faces(h, w, device)
    face_count_initial = int(faces.shape[0])

    if faces.numel() == 0:
        raise RuntimeError("No candidate faces remain.")

    faces = _prune_faces(vertices_flat, faces, depth_ratio=depth_ratio, max_edge=max_edge)
    face_count_after_prune = int(faces.shape[0])

    if faces.numel() == 0:
        raise RuntimeError(
            "No candidate faces remain after applying _prune_faces; relax depth_ratio or max_edge."
        )

    return SurfaceMesh(
        depth_hw=depth_hw,
        vertices_flat=vertices_flat,
        valid_vertex_mask=valid_vertex_mask,
        faces=faces,
        face_count_initial=face_count_initial,
        face_count_after_prune=face_count_after_prune,
    )


def _compute_mesh_supported_vertex_mask(mesh: SurfaceMesh) -> torch.Tensor:
    """Return valid vertices that belong to at least one retained triangle face.

    Args:
        mesh: Depth-induced surface mesh after face filtering and pruning.

    Returns:
        Boolean mask with shape [H, W]. A vertex is true only if it has valid
        depth and is referenced by at least one retained face.
    """
    supported_flat = torch.zeros(
        mesh.depth_hw.numel(),
        dtype=torch.bool,
        device=mesh.depth_hw.device,
    )
    if mesh.faces.numel() > 0:
        supported_flat[mesh.faces.reshape(-1)] = True
    return supported_flat.reshape_as(mesh.valid_vertex_mask) & mesh.valid_vertex_mask


def _deterministic_stratified_face_indices(
    weights: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.zeros((0,), dtype=torch.int64, device=weights.device)

    weight_sum = weights.sum()
    if not torch.isfinite(weight_sum) or weight_sum <= 0:
        raise RuntimeError("Invalid face weights; weights.sum() must be finite and positive.")

    cdf = torch.cumsum(weights / weight_sum, dim=0)
    cdf[-1] = 1.0
    positions = (
        torch.arange(num_samples, device=weights.device, dtype=torch.float32) + 0.5
    ) / float(num_samples)
    return torch.searchsorted(cdf, positions, right=False)


def _per_face_occurrence_index(face_indices: torch.Tensor) -> torch.Tensor:
    if face_indices.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=face_indices.device)

    order = torch.argsort(face_indices, stable=True)
    sorted_faces = face_indices[order]
    sorted_pos = torch.arange(sorted_faces.numel(), device=face_indices.device, dtype=torch.int64)
    group_start_mask = torch.ones_like(sorted_faces, dtype=torch.bool)
    group_start_mask[1:] = sorted_faces[1:] != sorted_faces[:-1]
    group_start_pos = torch.where(group_start_mask, sorted_pos, torch.zeros_like(sorted_pos))
    group_start_pos = torch.cummax(group_start_pos, dim=0).values
    local_sorted = sorted_pos - group_start_pos

    local = torch.empty_like(local_sorted)
    local[order] = local_sorted
    return local


def _deterministic_barycentric_samples(
    face_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if face_indices.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=face_indices.device)
        return empty, empty, empty

    local_idx = _per_face_occurrence_index(face_indices).to(torch.float32)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    psi = math.sqrt(2.0) - 1.0

    u = torch.frac(0.5 + (local_idx + 1.0) * phi)
    v = torch.frac(0.5 + (local_idx + 1.0) * psi)
    mask = u + v > 1.0
    u = torch.where(mask, 1.0 - u, u)
    v = torch.where(mask, 1.0 - v, v)

    w0 = 1.0 - u - v
    w1 = u
    w2 = v
    return w0, w1, w2


def _normalize_map_by_quantile(
    value_hw: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float = 0.9,
) -> torch.Tensor:
    """Normalize a scalar map by a valid-pixel quantile.

    Args:
        value_hw: Scalar map with shape [H, W].
        valid_mask: Boolean valid mask with shape [H, W].
        quantile: Quantile used as the normalization denominator.

    Returns:
        Normalized map clamped to [0, 1] with shape [H, W].
    """
    valid_values = value_hw[valid_mask]
    if valid_values.numel() == 0:
        return torch.zeros_like(value_hw, dtype=torch.float32)
    q = min(max(float(quantile), 1e-6), 1.0)
    denom = torch.quantile(valid_values.detach().float(), q).to(value_hw.dtype).clamp_min(1e-6)
    return (value_hw / denom).clamp(0.0, 1.0).to(torch.float32)


def compute_image_gradient_strength(
    image: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    quantile: float = 0.9,
) -> torch.Tensor:
    """Compute normalized Sobel RGB gradient for vertex-detail scoring.

    Args:
        image: Image tensor with shape [C, H, W] or [B, C, H, W].
        valid_mask: Optional boolean mask with shape [H, W] or [B, H, W].
        quantile: Per-image quantile used as the normalization denominator.

    Returns:
        Gradient strength clamped to [0, 1], with shape [H, W] for CHW input or
        [B, H, W] for BCHW input.
    """
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError("quantile must be in (0, 1].")

    squeeze_batch = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True
    elif image.ndim != 4:
        raise ValueError(f"image must have shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}.")

    if valid_mask is None:
        valid_mask = torch.ones(
            image.shape[0],
            image.shape[-2],
            image.shape[-1],
            dtype=torch.bool,
            device=image.device,
        )
    elif valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
    elif valid_mask.ndim != 3:
        raise ValueError(
            f"valid_mask must have shape [H,W] or [B,H,W], got {tuple(valid_mask.shape)}."
        )
    if valid_mask.shape[0] != image.shape[0] or valid_mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} must match image shape {tuple(image.shape)}."
        )
    valid_mask = valid_mask.to(device=image.device, dtype=torch.bool)

    if not image.is_floating_point():
        image = image.float()

    if image.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        gray = (image * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    else:
        gray = image.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(gray_padded, sobel_x)
    grad_y = F.conv2d(gray_padded, sobel_y)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12).squeeze(1)

    normalized = torch.stack(
        [
            _normalize_map_by_quantile(magnitude[index], valid_mask[index], quantile=quantile)
            for index in range(magnitude.shape[0])
        ],
        dim=0,
    )
    return normalized[0] if squeeze_batch else normalized


def compute_near_depth_score(depth_hw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute a weak foreground prior from metric depth.

    Args:
        depth_hw: Dense metric depth with shape [H, W].
        valid_mask: Boolean valid depth mask with shape [H, W].

    Returns:
        Near-depth score clamped to [0, 1] with shape [H, W].
    """
    valid_mask = valid_mask.to(device=depth_hw.device, dtype=torch.bool)
    valid_depth = depth_hw[valid_mask]
    if valid_depth.numel() == 0:
        return torch.zeros_like(depth_hw, dtype=torch.float32)
    near_ref = torch.median(valid_depth).to(depth_hw.dtype).clamp_min(1e-6)
    score = (near_ref / depth_hw.clamp(min=1e-6)).clamp(0.0, 4.0) / 4.0
    return torch.where(valid_mask, score, torch.zeros_like(score)).to(torch.float32)


def compute_vertex_detail_score(
    *,
    image_chw: Optional[torch.Tensor],
    depth_hw: torch.Tensor,
    valid_mask: torch.Tensor,
    image_weight: float,
    depth_weight: float,
    detail_quantile: float = 0.9,
) -> torch.Tensor:
    """Compute deterministic vertex-detail priorities.

    Args:
        image_chw: Optional RGB/features image with shape [C, H, W].
        depth_hw: Dense metric depth with shape [H, W].
        valid_mask: Boolean valid vertex mask with shape [H, W].
        image_weight: Weight for normalized image detail.
        depth_weight: Weight for weak near-depth score.
        detail_quantile: Quantile used to normalize image detail.

    Returns:
        Vertex detail score with shape [H, W]. Invalid vertices are `-inf`.
    """
    valid_mask = valid_mask.to(device=depth_hw.device, dtype=torch.bool)
    if image_chw is None or float(image_weight) == 0.0:
        rgb_grad = torch.zeros_like(depth_hw, dtype=torch.float32)
    else:
        rgb_grad = compute_image_gradient_strength(
            image_chw.to(device=depth_hw.device),
            valid_mask=valid_mask,
            quantile=detail_quantile,
        )
    near_depth = (
        torch.zeros_like(depth_hw, dtype=torch.float32)
        if float(depth_weight) == 0.0
        else compute_near_depth_score(depth_hw, valid_mask)
    )
    score = (
        float(image_weight) * rgb_grad
        + float(depth_weight) * near_depth
    )
    return torch.where(valid_mask, score, torch.full_like(score, -float("inf")))


def _make_vertex_output_from_indices(
    *,
    depth_hw: torch.Tensor,
    fx: float,
    fy: float,
    coord_norm: str,
    selected: torch.Tensor,
) -> SparseSamplingOutput:
    """Build mesh-vertex samples from flattened vertex indices.

    Args:
        depth_hw: Dense metric depth with shape `[H, W]`.
        fx: Pixel-space focal length in x.
        fy: Pixel-space focal length in y.
        coord_norm: Coordinate normalization mode.
        selected: Flattened vertex indices with shape [N].

    Returns:
        Mesh-vertex sparse samples with shape [N, 2].
    """
    device = depth_hw.device
    h, w = depth_hw.shape
    selected = selected.to(device=device, dtype=torch.long)
    if selected.numel() == 0:
        return SparseSamplingOutput(
            coords_yx_ndc=torch.zeros((0, 2), dtype=torch.float32, device=device),
            sample_responsibility_area_metric=torch.zeros((0,), dtype=torch.float32, device=device),
            sample_kind=torch.zeros((0,), dtype=torch.long, device=device),
        )

    i_s = (selected // w).to(torch.float32)
    j_s = (selected % w).to(torch.float32)
    coords = _normalize_ij_coords(i_s, j_s, h=h, w=w, coord_norm=coord_norm).to(torch.float32)

    selected_depth = depth_hw.reshape(-1)[selected].clamp(min=1e-6).to(torch.float32)
    focal_area = max(float(fx) * float(fy), 1e-6)
    responsibility_area = selected_depth.square() / focal_area
    return SparseSamplingOutput(
        coords_yx_ndc=coords,
        sample_responsibility_area_metric=responsibility_area.to(torch.float32),
        sample_kind=torch.full(
            (coords.shape[0],),
            SAMPLE_KIND_VERTEX,
            dtype=torch.long,
            device=device,
        ),
    )


def _select_anchor_vertices(
    *,
    valid_mask: torch.Tensor,
    priority_hw: torch.Tensor,
    stride: int,
    deterministic: bool,
) -> torch.Tensor:
    """Select high-priority mesh vertices from a coarse image grid.

    Args:
        valid_mask: Boolean valid vertex mask with shape [H, W].
        priority_hw: Priority map with shape [H, W].
        stride: Coarse grid cell size.
        deterministic: Whether to use deterministic priority selection.

    Returns:
        Flattened vertex indices with shape [N].
    """
    if stride <= 0:
        raise ValueError("Anchor grid stride must be positive.")
    device = valid_mask.device
    h, w = valid_mask.shape
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return valid_indices

    row = valid_indices // w
    col = valid_indices % w
    cell_w = math.ceil(w / float(stride))
    cell_id = (row // stride) * cell_w + (col // stride)
    priority = priority_hw.reshape(-1).to(device=device, dtype=torch.float32)[valid_indices]
    if not deterministic:
        priority = priority + torch.rand_like(priority) * 1e-4

    # Stable lexicographic order: cell id ascending, priority descending, index ascending.
    order_by_index = torch.argsort(valid_indices, stable=True)
    valid_indices = valid_indices[order_by_index]
    cell_id = cell_id[order_by_index]
    priority = priority[order_by_index]
    order_by_priority = torch.argsort(priority, descending=True, stable=True)
    valid_indices = valid_indices[order_by_priority]
    cell_id = cell_id[order_by_priority]
    order_by_cell = torch.argsort(cell_id, stable=True)
    selected = valid_indices[order_by_cell]
    sorted_cell = cell_id[order_by_cell]
    first_in_cell = torch.ones_like(sorted_cell, dtype=torch.bool)
    first_in_cell[1:] = sorted_cell[1:] != sorted_cell[:-1]
    return selected[first_in_cell]


def _select_vertex_detail_indices(
    *,
    valid_mask: torch.Tensor,
    priority_hw: torch.Tensor,
    already_selected: torch.Tensor,
    num_samples: int,
    deterministic: bool,
) -> torch.Tensor:
    """Select extra high-priority mesh vertices without duplicating scaffold vertices.

    Args:
        valid_mask: Boolean valid vertex mask with shape [H, W].
        priority_hw: Priority map with shape [H, W].
        already_selected: Flattened scaffold indices with shape [S].
        num_samples: Requested detail vertex count.
        deterministic: Whether to use deterministic top-k selection.

    Returns:
        Flattened detail vertex indices with shape [D].
    """
    device = valid_mask.device
    if num_samples <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)

    valid_flat = valid_mask.reshape(-1)
    priority = priority_hw.reshape(-1).to(device=device, dtype=torch.float32)
    priority = torch.where(valid_flat, priority, torch.full_like(priority, -float("inf")))
    if already_selected.numel() > 0:
        priority = priority.clone()
        priority[already_selected.to(device=device, dtype=torch.long)] = -float("inf")
    available = torch.isfinite(priority)
    k = min(int(num_samples), int(available.sum().item()))
    if k <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    if deterministic:
        return torch.topk(priority, k=k, largest=True).indices

    available_indices = torch.nonzero(available, as_tuple=False).squeeze(-1)
    probs = torch.softmax(priority[available_indices], dim=0)
    sampled = torch.multinomial(probs, num_samples=k, replacement=False)
    return available_indices[sampled]


def _sample_extra_faces_from_mesh(
    *,
    mesh: SurfaceMesh,
    num_samples: int,
    coord_norm: str,
    deterministic: bool,
) -> SparseSamplingOutput:
    """Sample triangle-face interior supports using the original extra-face weights.

    Args:
        mesh: Depth-induced surface mesh.
        num_samples: Number of face-interior supports to return.
        coord_norm: Coordinate normalization mode.
        deterministic: Whether to use deterministic stratified face and barycentric samples.
    Returns:
        Triangle-face sparse samples with coordinates of shape [N, 2].
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    device = mesh.depth_hw.device
    if num_samples == 0:
        return SparseSamplingOutput(
            coords_yx_ndc=torch.zeros((0, 2), dtype=torch.float32, device=device),
            sample_responsibility_area_metric=torch.zeros((0,), dtype=torch.float32, device=device),
            sample_kind=torch.zeros((0,), dtype=torch.long, device=device),
        )

    weights, areas = _compute_extra_face_weights(
        faces=mesh.faces,
        Vflat=mesh.vertices_flat,
        valid_pixel_mask=mesh.valid_vertex_mask,
    )
    total_weight = weights.sum()
    if not torch.isfinite(total_weight) or total_weight <= 0:
        raise RuntimeError("Invalid total extra-face weights after filtering/pruning; check depth values.")
    probs = weights / total_weight

    if deterministic:
        tri_idx = _deterministic_stratified_face_indices(
            weights=weights,
            num_samples=num_samples,
        )
        selected_faces = mesh.faces[tri_idx]
        w0, w1, w2 = _deterministic_barycentric_samples(tri_idx)
    else:
        tri_idx = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        selected_faces = mesh.faces[tri_idx]
        u = torch.rand(num_samples, device=device)
        v = torch.rand(num_samples, device=device)
        mask = u + v > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        w0 = 1.0 - u - v
        w1 = u
        w2 = v

    h, w = mesh.depth_hw.shape
    face_i, face_j = _faces_to_ij(selected_faces, h, w)
    i0, i1, i2 = face_i[:, 0].float(), face_i[:, 1].float(), face_i[:, 2].float()
    j0, j1, j2 = face_j[:, 0].float(), face_j[:, 1].float(), face_j[:, 2].float()

    i_s = w0 * i0 + w1 * i1 + w2 * i2
    j_s = w0 * j0 + w1 * j1 + w2 * j2

    face_sample_count = torch.bincount(tri_idx, minlength=mesh.faces.shape[0]).to(torch.float32)
    responsibility_area_metric = areas[tri_idx] / face_sample_count[tri_idx].clamp_min(1.0)

    coords = _normalize_ij_coords(
        i_s=i_s,
        j_s=j_s,
        h=h,
        w=w,
        coord_norm=coord_norm,
    ).to(torch.float32)
    return SparseSamplingOutput(
        coords_yx_ndc=coords,
        sample_responsibility_area_metric=responsibility_area_metric.to(torch.float32),
        sample_kind=torch.full(
            (coords.shape[0],),
            SAMPLE_KIND_FACE,
            dtype=torch.long,
            device=device,
        ),
    )


def make_sparse_surface_samples(
    depth_hw: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sample_point_num: int,
    image_chw: Optional[torch.Tensor] = None,
    coord_norm: str = "minus_one_to_one",
    depth_ratio: float = 1.05,
    max_edge: Optional[float] = None,
) -> SparseSamplingOutput:
    """Sample a fixed budget from triangle vertices and triangle interiors.

    Args:
        depth_hw: Dense metric depth with shape [H, W].
        fx, fy, cx, cy: Pixel-space camera intrinsics.
        sample_point_num: Target total number of sparse supports.
        image_chw: Optional image tensor with shape [C, H, W] used for vertex-detail scoring.
        coord_norm: Coordinate normalization, either `minus_one_to_one` or `zero_one`.
        depth_ratio: Near-depth discontinuity pruning ratio.
        max_edge: Optional maximum 3D edge length for pruning faces.

    Returns:
        Sparse sampling output with vertex supports first and triangle-face interior
        supports second. The output has exactly `sample_point_num` rows when at least
        one valid face remains for the face budget.
    """
    if sample_point_num < 0:
        raise ValueError("sample_point_num must be non-negative.")
    if _ANCHOR_GRID_STRIDE <= 0:
        raise ValueError("Anchor grid stride must be positive.")
    if image_chw is not None and image_chw.shape[-2:] != depth_hw.shape:
        raise ValueError(
            f"image_chw spatial shape {tuple(image_chw.shape[-2:])} must match "
            f"depth_hw shape {tuple(depth_hw.shape)}."
        )

    device = depth_hw.device
    if sample_point_num == 0:
        return SparseSamplingOutput(
            coords_yx_ndc=torch.zeros((0, 2), dtype=torch.float32, device=device),
            sample_responsibility_area_metric=torch.zeros((0,), dtype=torch.float32, device=device),
            sample_kind=torch.zeros((0,), dtype=torch.long, device=device),
        )

    mesh = _build_surface_mesh(
        depth_hw=depth_hw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_ratio=depth_ratio,
        max_edge=max_edge,
    )
    valid_mask = _compute_mesh_supported_vertex_mask(mesh)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return SparseSamplingOutput(
            coords_yx_ndc=torch.zeros((0, 2), dtype=torch.float32, device=device),
            sample_responsibility_area_metric=torch.zeros((0,), dtype=torch.float32, device=device),
            sample_kind=torch.zeros((0,), dtype=torch.long, device=device),
        )

    priority = compute_vertex_detail_score(
        image_chw=image_chw,
        depth_hw=depth_hw,
        valid_mask=valid_mask,
        image_weight=_IMAGE_DETAIL_WEIGHT,
        depth_weight=_DEPTH_DETAIL_WEIGHT,
        detail_quantile=_DETAIL_SCORE_QUANTILE,
    )
    scaffold = _select_anchor_vertices(
        valid_mask=valid_mask,
        priority_hw=priority,
        stride=_ANCHOR_GRID_STRIDE,
        deterministic=True,
    )

    anchor_budget_ratio = min(max(float(_ANCHOR_BUDGET_RATIO), 0.0), 1.0)
    vertex_target = min(
        max(int(scaffold.numel()), int(round(float(sample_point_num) * anchor_budget_ratio))),
        valid_count,
        int(sample_point_num),
    )
    if scaffold.numel() > vertex_target:
        scaffold_priority = priority.reshape(-1)[scaffold]
        order = torch.argsort(scaffold_priority, descending=True, stable=True)
        scaffold = scaffold[order[:vertex_target]]

    detail = _select_vertex_detail_indices(
        valid_mask=valid_mask,
        priority_hw=priority,
        already_selected=scaffold,
        num_samples=vertex_target - int(scaffold.numel()),
        deterministic=True,
    )
    vertex_indices = torch.cat([scaffold, detail], dim=0)
    vertex_output = _make_vertex_output_from_indices(
        depth_hw=depth_hw,
        fx=fx,
        fy=fy,
        coord_norm=coord_norm,
        selected=vertex_indices,
    )
    face_budget = int(sample_point_num) - int(vertex_output.coords_yx_ndc.shape[0])

    if face_budget <= 0:
        return vertex_output

    face_output = _sample_extra_faces_from_mesh(
        mesh=mesh,
        num_samples=face_budget,
        coord_norm=coord_norm,
        deterministic=True,
    )
    return SparseSamplingOutput(
        coords_yx_ndc=torch.cat(
            [vertex_output.coords_yx_ndc, face_output.coords_yx_ndc],
            dim=0,
        ),
        sample_responsibility_area_metric=torch.cat(
            [
                vertex_output.sample_responsibility_area_metric,
                face_output.sample_responsibility_area_metric,
            ],
            dim=0,
        ),
        sample_kind=torch.cat(
            [
                vertex_output.sample_kind,
                face_output.sample_kind,
            ],
            dim=0,
        )
    )
