"""Shared SE(3)-equivariant helper utilities used by the models."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def safe_norm(vec: Tensor, *, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """Return the L2 norm of ``vec`` clamped away from zero.

    Parameters
    ----------
    vec:
        Tensor containing vectors along dimension ``dim``.
    dim:
        Dimension containing the vector components.
    keepdim:
        Whether to retain ``dim`` in the returned tensor.
    eps:
        Minimum norm used for clamping.  This avoids divisions by zero when
        normalising vectors in downstream modules.
    """

    norm = torch.linalg.norm(vec, dim=dim, keepdim=keepdim)
    return torch.clamp(norm, min=eps)


def unit(vec: Tensor, *, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Return a safely normalised copy of ``vec``."""

    return vec / safe_norm(vec, dim=dim, keepdim=True, eps=eps)


def vector_linear(vectors: Tensor, weight: Tensor) -> Tensor:
    """Apply a linear map over vector channels while preserving equivariance.

    The function expects ``vectors`` to have shape ``(..., C_in, 3)`` and applies
    the ``(C_out, C_in)`` matrix ``weight`` across the channel dimension without
    mixing spatial components.  The return tensor has shape ``(..., C_out, 3)``.
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

    return torch.einsum("oi,...ic->...oc", weight, vectors)


def apply_gating(update: Tensor, gate: Tensor) -> Tensor:
    """Apply row-wise gates to vector features."""

    if gate.ndim != update.ndim - 1:
        raise ValueError("Gate dimensionality must match update tensor")

    expanded_gate = gate.unsqueeze(-1)
    return update * expanded_gate


def broadcast_param(param: Tensor, target_dims: Iterable[int]) -> Tensor:
    """Broadcast ``param`` across ``target_dims`` trailing dimensions."""

    shape = [1] * target_dims
    return param.view(*shape, -1)


__all__ = ["safe_norm", "unit", "vector_linear", "apply_gating", "broadcast_param"]
