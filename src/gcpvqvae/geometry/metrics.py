"""Geometric metrics and evaluation utilities."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

Tensor = torch.Tensor


def rmsd(coords_a: Tensor, coords_b: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
    """Compute the root-mean-square deviation between two coordinate sets.

    Parameters
    ----------
    coords_a, coords_b:
        Tensors containing matching coordinates.  The tensors can either be of
        shape ``(..., 3)`` or ``(..., 3, 3)`` (for three backbone atoms).  The
        function operates element-wise on the leading dimensions.
    mask:
        Optional boolean mask whose ``True`` entries mark valid residues.  The
        mask is broadcast against the leading dimensions of the coordinate
        tensors.
    """

    if coords_a.shape != coords_b.shape:
        raise ValueError("coords_a and coords_b must have identical shapes")

    diff = coords_a - coords_b
    if diff.size(-2) == 3 and diff.ndim >= 3:
        # Flatten atom dimension.
        diff = diff.reshape(*diff.shape[:-2], -1)
    else:
        diff = diff.reshape(*diff.shape[:-1], -1)

    if mask is not None:
        mask = mask.to(diff.dtype)
        mask = mask.reshape(*mask.shape, *([1] * (diff.ndim - mask.ndim)))
        diff = diff * mask
        valid = mask.sum()
        denom = torch.clamp(valid * diff.shape[-1], min=1.0)
    else:
        denom = float(diff.numel())

    mse = (diff**2).sum() / denom
    return torch.sqrt(mse)


def tm_score(
    coords_a: Tensor,
    coords_b: Tensor,
    *,
    mask: Optional[Tensor] = None,
    length_scale: Optional[float] = None,
) -> Tensor:
    """Compute an approximate TM-score between two aligned structures.

    The implementation follows the standard TM-score formulation operating on
    the per-residue Cα coordinates.  The caller is expected to align the
    structures beforehand (for example via :func:`gcpvqvae.geometry.frames.kabsch_align`).
    """

    if coords_a.shape != coords_b.shape:
        raise ValueError("coords_a and coords_b must have identical shapes")
    if coords_a.size(-2) != 3:
        raise ValueError("Inputs must contain backbone atoms with shape (..., 3, 3)")

    ca_a = coords_a[..., 1, :]
    ca_b = coords_b[..., 1, :]
    diff = torch.linalg.norm(ca_a - ca_b, dim=-1)

    if mask is not None:
        mask = mask.to(torch.bool)
        diff = diff[mask]

    L = diff.numel()
    if L == 0:
        return torch.tensor(0.0, dtype=coords_a.dtype, device=coords_a.device)

    if length_scale is None:
        effective_L = max(L, 19)
        length_scale = 1.24 * (effective_L - 15) ** (1 / 3) - 1.8
    length_scale = max(float(length_scale), 0.5)
    length_scale_tensor = torch.tensor(length_scale, dtype=coords_a.dtype, device=coords_a.device)

    score = (1.0 / L) * torch.sum(1.0 / (1.0 + (diff / length_scale_tensor) ** 2))
    return score


def gdt_ts(
    coords_a: Tensor,
    coords_b: Tensor,
    *,
    mask: Optional[Tensor] = None,
    thresholds: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
) -> Tensor:
    """Compute the Global Distance Test Total Score (GDT-TS).

    The metric measures the fraction of residues whose Cα atoms fall within the
    specified distance thresholds after the structures have been aligned.
    """

    if coords_a.shape != coords_b.shape:
        raise ValueError("coords_a and coords_b must have identical shapes")
    if coords_a.size(-2) != 3:
        raise ValueError("Inputs must contain backbone atoms with shape (..., 3, 3)")
    if not thresholds:
        raise ValueError("At least one distance threshold is required")

    ca_a = coords_a[..., 1, :]
    ca_b = coords_b[..., 1, :]
    diff = torch.linalg.norm(ca_a - ca_b, dim=-1)

    if mask is not None:
        mask = mask.to(torch.bool)
        diff = diff[mask]

    L = diff.numel()
    if L == 0:
        return torch.tensor(0.0, dtype=coords_a.dtype, device=coords_a.device)

    scores = []
    for threshold in thresholds:
        threshold = float(threshold)
        within = (diff <= threshold).sum().to(coords_a.dtype)
        scores.append(within / float(L))

    return torch.mean(torch.stack(scores))


__all__ = ["rmsd", "tm_score", "gdt_ts"]
