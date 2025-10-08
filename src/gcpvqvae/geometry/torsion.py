"""Torsion angle computations for protein backbones."""

from __future__ import annotations

from typing import Dict

import torch

Tensor = torch.Tensor


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Return the signed dihedral angle for four sets of points.

    Args:
        p0: Tensor of shape ``(..., 3)`` with the first positions.
        p1: Tensor matching ``p0`` with the second positions.
        p2: Tensor matching ``p0`` with the third positions.
        p3: Tensor matching ``p0`` with the fourth positions.

    Returns:
        Tensor of dihedral angles in radians with shape ``(...)``.
    """

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
    b1_norm = torch.clamp(b1_norm, min=1e-8)
    b1_unit = b1 / b1_norm

    v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum(dim=-1)

    angle = torch.atan2(y, x)
    return angle


def backbone_torsions(backbone: Tensor) -> Dict[str, Tensor]:
    """Compute φ, ψ, and ω torsions for a backbone tensor.

    Args:
        backbone: Tensor of shape ``(L, 3, 3)`` containing ``(N, CA, C)`` atom
            coordinates.

    Returns:
        Dictionary mapping ``phi``, ``psi``, and ``omega`` to tensors of shape
        ``(L,)`` with torsion angles in radians. Missing values are filled with
        zeros.

    Raises:
        ValueError: If ``backbone`` does not have shape ``(L, 3, 3)``.
    """

    if backbone.ndim != 3 or backbone.shape[1:] != (3, 3):
        raise ValueError("backbone must have shape (L, 3, 3)")

    n = backbone[:, 0, :]
    ca = backbone[:, 1, :]
    c = backbone[:, 2, :]

    L = backbone.shape[0]
    dtype = backbone.dtype
    device = backbone.device

    phi = torch.zeros((L,), dtype=dtype, device=device)
    psi = torch.zeros((L,), dtype=dtype, device=device)
    omega = torch.zeros((L,), dtype=dtype, device=device)

    if L >= 2:
        phi[1:] = _dihedral(c[:-1], n[1:], ca[1:], c[1:])
        psi[:-1] = _dihedral(n[:-1], ca[:-1], c[:-1], n[1:])
        omega[:-1] = _dihedral(ca[:-1], c[:-1], n[1:], ca[1:])

    return {"phi": phi, "psi": psi, "omega": omega}


__all__ = ["backbone_torsions"]
