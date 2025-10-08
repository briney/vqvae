"""Shared SE(3)-equivariant helper utilities used by the models."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor


def safe_norm(vec: Tensor, *, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """Return the L2 norm of ``vec`` clamped away from zero.

    Args:
        vec: Tensor containing vectors along dimension ``dim``.
        dim: Dimension containing the vector components.
        keepdim: Retain ``dim`` in the returned tensor when ``True``.
        eps: Minimum norm used for clamping to avoid division by zero.

    Returns:
        Tensor containing the clamped norms.
    """

    norm = torch.linalg.norm(vec, dim=dim, keepdim=keepdim)
    return torch.clamp(norm, min=eps)


def unit(vec: Tensor, *, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Return a safely normalised copy of ``vec``.

    Args:
        vec: Tensor containing vectors to normalise.
        dim: Dimension containing the vector components.
        eps: Minimum norm used when normalising.

    Returns:
        Tensor of unit vectors.
    """

    return vec / safe_norm(vec, dim=dim, keepdim=True, eps=eps)


def vector_linear(vectors: Tensor, weight: Tensor) -> Tensor:
    """Apply a linear map over vector channels while preserving equivariance.

    Args:
        vectors: Tensor of shape ``(..., C_in, 3)``.
        weight: Projection matrix of shape ``(C_out, C_in)``.

    Returns:
        Tensor of shape ``(..., C_out, 3)``.

    Raises:
        ValueError: If tensor dimensions are incompatible.
    """

    if vectors.size(-1) != 3:
        raise ValueError("vectors must store 3D components in the last dimension")
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D matrix")
    if vectors.size(-2) != weight.size(1):
        raise ValueError(
            f"Mismatched channel dimensions: expected {weight.size(1)}, "
            f"got {vectors.size(-2)}"
        )

    vector_dtype = vectors.dtype
    weight_dtype = weight.dtype

    if vector_dtype != weight_dtype:
        vectors = vectors.to(weight_dtype)
        result = torch.einsum("oi,...ic->...oc", weight, vectors)
        return result.to(vector_dtype)

    return torch.einsum("oi,...ic->...oc", weight, vectors)


def apply_gating(update: Tensor, gate: Tensor) -> Tensor:
    """Apply row-wise gates to vector features.

    Args:
        update: Tensor of shape ``(..., C, 3)`` containing vector features.
        gate: Tensor broadcastable to ``update[..., C]``.

    Returns:
        Tensor with gated vector features.
    """

    if gate.ndim != update.ndim - 1:
        raise ValueError("Gate dimensionality must match update tensor")

    expanded_gate = gate.unsqueeze(-1)
    return update * expanded_gate


def broadcast_param(param: Tensor, target_dims: Iterable[int]) -> Tensor:
    """Broadcast ``param`` across ``target_dims`` trailing dimensions.

    Args:
        param: Tensor to reshape.
        target_dims: Number of leading singleton dimensions to prepend.

    Returns:
        Reshaped tensor with ``target_dims`` singleton axes followed by ``param``.
    """

    shape = [1] * target_dims
    return param.view(*shape, -1)


def scalarize(vectors: Tensor, frames: Tensor) -> Tensor:
    """Project vector features onto local frames to obtain scalars.

    Args:
        vectors: Tensor of shape ``(E, C, 3)`` storing vector channels.
        frames: Tensor of shape ``(E, 3, 3)`` storing orthonormal frames.

    Returns:
        Tensor of shape ``(E, C * 3)`` containing scalar projections.

    Raises:
        ValueError: If input shapes are incompatible.
    """

    if vectors.size(-1) != 3:
        raise ValueError("scalarize expects vectors with 3D components")
    if frames.ndim != 3 or frames.size(-1) != 3 or frames.size(-2) != 3:
        raise ValueError("frames must have shape (E, 3, 3)")

    if vectors.size(-2) == 0:
        return vectors.new_zeros(vectors.shape[:-2] + (0,))

    projected = torch.einsum("eij,ecj->eci", frames.to(vectors.dtype), vectors)
    return projected.reshape(vectors.shape[0], -1)


def vectorize(
    vectors: Tensor,
    weight: Optional[Tensor],
    *,
    gate: Optional[Tensor] = None,
    vector_gate: bool = True,
    enable_e3_equivariance: bool = True,
) -> Tensor:
    """Reconstruct full-resolution vectors from downsampled channels.

    Args:
        vectors: Tensor containing downsampled vector channels.
        weight: Optional projection matrix with shape ``(C_out, C_in)``.
        gate: Optional multiplicative gates applied channel-wise.
        vector_gate: Apply ``gate`` to the reconstructed vectors when ``True``.
        enable_e3_equivariance: Disable the equivariant pathway when ``False``.

    Returns:
        Tensor of reconstructed vectors with shape ``(..., C_out, 3)``.
    """

    if weight is None or weight.numel() == 0:
        out_channels = weight.size(0) if weight is not None else 0
        shape = vectors.shape[:-2] + (out_channels, 3)
        return vectors.new_zeros(shape)

    if not enable_e3_equivariance:
        shape = vectors.shape[:-2] + (weight.size(0), 3)
        return vectors.new_zeros(shape)

    lifted = vector_linear(vectors, weight)
    if gate is not None and vector_gate:
        lifted = apply_gating(lifted, gate)
    return lifted


__all__ = [
    "safe_norm",
    "unit",
    "vector_linear",
    "apply_gating",
    "broadcast_param",
    "scalarize",
    "vectorize",
]
