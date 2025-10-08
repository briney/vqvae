"""Local frame utilities and alignment helpers.

This module purposefully contains only lightweight tensor operations so that it
can be reused by both the data pipeline (to build per-edge frames) and the loss
functions (for alignment aware reconstruction losses).  The functions operate on
PyTorch tensors because the rest of the project is implemented in PyTorch and
automatic differentiation through some of the primitives – most notably the
Kabsch alignment routine – is occasionally useful when computing auxiliary
metrics during training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


def _normalize(vec: Tensor, eps: float = 1e-8) -> Tensor:
    """Return safely normalised vectors.

    Args:
        vec: Tensor whose final dimension stores vector components.
        eps: Minimum norm used to avoid division by zero.

    Returns:
        Tensor with the same shape as ``vec`` containing unit-length vectors.
    """

    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return vec / norm


def _orthonormal_vector(vec: Tensor) -> Tensor:
    """Return an arbitrary vector orthogonal to ``vec``.

    Args:
        vec: Tensor of shape ``(..., 3)`` storing target directions.

    Returns:
        Tensor of shape ``(..., 3)`` with orthonormal companion vectors.
    """

    batch_shape = vec.shape[:-1]
    out = torch.zeros_like(vec)
    abs_vec = torch.abs(vec)
    # Choose the axis with the smallest absolute value to avoid degeneracy.
    indices = torch.argmin(abs_vec, dim=-1, keepdim=True)
    out.scatter_(-1, indices, 1.0)
    out = out - (out * vec).sum(dim=-1, keepdim=True) * vec
    return _normalize(out)


def build_local_frames(
    ca_positions: Tensor,
    edge_index: Tensor,
    *,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Construct a right-handed orthonormal frame for each directed edge.

    Args:
        ca_positions: Tensor of shape ``(L, 3)`` containing Cα coordinates.
        edge_index: Long tensor of shape ``(2, E)`` indicating directed edges.
        mask: Optional boolean mask of shape ``(L,)`` marking valid residues.
            Frames touching invalid residues are replaced with identities.
        eps: Numerical stability constant used during normalisation.

    Returns:
        Tensor of shape ``(E, 3, 3)`` where each slice is a rotation matrix with
        tangent, normal, and binormal vectors as columns.

    Raises:
        ValueError: If input tensors have unexpected shapes.
    """

    if ca_positions.ndim != 2 or ca_positions.size(-1) != 3:
        raise ValueError("ca_positions must be of shape (L, 3)")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape (2, E)")

    src, dst = edge_index
    if src.numel() == 0:
        return torch.empty((0, 3, 3), device=ca_positions.device, dtype=ca_positions.dtype)

    edge_vec = ca_positions[dst] - ca_positions[src]
    tangent = _normalize(edge_vec, eps)

    # Compute neighbourhood averages for bias vectors.
    num_nodes = ca_positions.shape[0]
    device = ca_positions.device
    dtype = ca_positions.dtype

    neighbour_sum = torch.zeros((num_nodes, 3), device=device, dtype=dtype)
    neighbour_count = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
    neighbour_unit = tangent
    neighbour_sum.index_add_(0, src, neighbour_unit)
    neighbour_count.index_add_(0, src, torch.ones_like(neighbour_count[src]))

    # Prepare fallback axes once to avoid repeated allocations in the loop.
    fallback_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    fallback_alt_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

    bias_vectors = neighbour_sum[src] - neighbour_unit
    denom = torch.clamp(neighbour_count[src] - 1.0, min=1.0)
    bias_vectors = bias_vectors / denom

    # Project the bias into the plane orthogonal to the tangent.
    proj = bias_vectors - (bias_vectors * tangent).sum(dim=-1, keepdim=True) * tangent

    needs_fallback = torch.linalg.norm(proj, dim=-1, keepdim=True) <= eps
    if needs_fallback.any():
        fallback = fallback_axis.expand_as(proj)
        fallback = fallback - (fallback * tangent).sum(dim=-1, keepdim=True) * tangent
        still_degenerate = torch.linalg.norm(fallback, dim=-1, keepdim=True) <= eps
        if still_degenerate.any():
            fallback2 = fallback_alt_axis.expand_as(proj)
            fallback2 = fallback2 - (fallback2 * tangent).sum(dim=-1, keepdim=True) * tangent
            fallback = torch.where(still_degenerate, fallback2, fallback)
        proj = torch.where(needs_fallback, fallback, proj)

    normal = _normalize(proj, eps)
    binormal = _normalize(torch.cross(tangent, normal, dim=-1), eps)

    normal = _normalize(torch.cross(binormal, tangent, dim=-1), eps)
    binormal = torch.cross(tangent, normal, dim=-1)

    frames = torch.stack((tangent, normal, binormal), dim=-1)

    if mask is not None:
        mask = mask.to(torch.bool)
        valid = mask[src] & mask[dst]
        identity = torch.eye(3, device=device, dtype=dtype).expand_as(frames)
        frames = torch.where(valid.view(-1, 1, 1), frames, identity)

    return frames


def kabsch_align(
    src: Tensor,
    dst: Tensor,
    *,
    mask: Optional[Tensor] = None,
    allow_reflections: bool = False,
    return_aligned: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Compute the optimal rigid transform aligning ``src`` onto ``dst``.

    Args:
        src: Tensor of shape ``(N, 3)`` describing source coordinates.
        dst: Tensor of shape ``(N, 3)`` describing target coordinates.
        mask: Optional boolean tensor of shape ``(N,)`` indicating valid
            correspondences.
        allow_reflections: Allow reflection components in the optimal rotation.
        return_aligned: When ``True`` also return the aligned source points.

    Returns:
        Tuple ``(rotation, translation, aligned)`` where ``rotation`` has shape
        ``(3, 3)``, ``translation`` has shape ``(3,)``, and ``aligned`` is either
        ``None`` or the aligned source coordinates of shape ``(N, 3)``.

    Raises:
        ValueError: If shapes are incompatible or fewer than three valid points
            are provided when ``mask`` is used.
    """

    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    if src.ndim != 2 or src.size(-1) != 3:
        raise ValueError("src and dst must be of shape (N, 3)")

    if mask is not None:
        if mask.shape != src.shape[:1]:
            raise ValueError("mask must have shape (N,)")
        mask = mask.to(torch.bool)
        if mask.sum() < 3:
            raise ValueError("At least three points are required for alignment")
        src_sel = src[mask]
        dst_sel = dst[mask]
    else:
        src_sel = src
        dst_sel = dst

    src_mean = src_sel.mean(dim=0)
    dst_mean = dst_sel.mean(dim=0)

    src_centered = src_sel - src_mean
    dst_centered = dst_sel - dst_mean

    cov = src_centered.transpose(0, 1) @ dst_centered

    svd_input = cov
    if cov.dtype in (torch.float16, torch.bfloat16):
        svd_input = cov.to(torch.float32)

    u, _, vh = torch.linalg.svd(svd_input, full_matrices=False)
    rotation = u @ vh

    if not allow_reflections:
        # ``torch.linalg.det`` does not currently support the low precision
        # dtypes that ``kabsch_align`` needs to handle (``float16`` and
        # ``bfloat16``) on all backends.  Mixed precision training pipelines can
        # therefore fail when the autocaster requests one of those types.
        # Promote to ``float32`` for the determinant check which is only used to
        # decide whether to flip the last singular vector, and cast back later if
        # required.
        det = torch.linalg.det(rotation.to(torch.float32))
        if det < 0:
            vh = vh.clone()
            vh[-1, :] *= -1
            rotation = u @ vh

    if rotation.dtype != src_sel.dtype:
        rotation = rotation.to(src_sel.dtype)

    translation = dst_mean - src_mean @ rotation

    aligned: Optional[Tensor]
    if return_aligned:
        aligned = (src @ rotation) + translation
    else:
        aligned = None

    return rotation, translation, aligned


__all__ = ["build_local_frames", "kabsch_align"]
