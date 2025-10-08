"""Geometry-aware reconstruction losses used by GCP-VQVAE."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from gcpvqvae.geometry.frames import kabsch_align


def _flatten_backbone(coords: Tensor) -> Tensor:
    """Reshape backbone tensors for pairwise operations.

    Args:
        coords: Coordinate tensor with shape ``(B, L, 3, 3)``.

    Returns:
        Tensor reshaped to ``(B, L*3, 3)``.
    """
    if coords.ndim != 4 or coords.size(-2) != 3 or coords.size(-1) != 3:
        raise ValueError("Backbone tensors must have shape (B, L, 3, 3)")
    return coords.view(coords.size(0), coords.size(1) * 3, 3)


def _broadcast_mask(mask: Optional[Tensor], length: int) -> Optional[Tensor]:
    """Expand residue masks across atoms to match :func:`_flatten_backbone`.

    Args:
        mask: Optional residue mask ``(B, L)``.
        length: Number of residues ``L``.

    Returns:
        Mask broadcast to ``(B, L*3)`` or ``None`` when no mask is provided.
    """
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError("Mask must have shape (B, L)")
    return mask.unsqueeze(-1).expand(-1, -1, 3).reshape(mask.shape[0], length * 3)


def _zero_loss_like(tensor: Tensor) -> Tensor:
    """Return a zero scalar that preserves gradients for ``tensor``."""

    zero = torch.zeros((), device=tensor.device, dtype=tensor.dtype)
    if tensor.requires_grad:
        zero = zero.requires_grad_()
    return zero


def aligned_mse(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute :math:`(P - T_\\text{aln})^2` averaged over valid atoms.

    Args:
        pred: Predicted coordinates with shape ``(B, L, 3, 3)``.
        target: Target coordinates with shape ``(B, L, 3, 3)``.
        mask: Optional boolean mask ``(B, L)`` selecting valid residues.

    Returns:
        Scalar tensor containing the aligned MSE loss.
    """

    pred_flat = _flatten_backbone(pred)
    target_flat = _flatten_backbone(target)

    mask_flat = _broadcast_mask(mask, pred.size(1))
    if mask_flat is not None:
        mask_flat = mask_flat.to(device=pred.device, dtype=torch.bool)

    losses = []
    for i in range(pred_flat.size(0)):
        if mask_flat is not None:
            mask_i = mask_flat[i]
            if mask_i.sum() < 3:
                losses.append(_zero_loss_like(pred_flat[i]))
                continue
        else:
            mask_i = None

        try:
            _, _, target_aligned = kabsch_align(
                target_flat[i],
                pred_flat[i],
                mask=mask_i,
                allow_reflections=False,
                return_aligned=True,
            )
        except ValueError:
            target_aligned = target_flat[i]

        diff = pred_flat[i] - target_aligned
        if mask_i is not None:
            diff = diff[mask_i]

        if diff.numel() == 0:
            losses.append(_zero_loss_like(pred_flat[i]))
            continue

        losses.append(diff.pow(2).mean())

    return torch.stack(losses).mean()


def backbone_distance_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    clamp_distance: float = 25.0,
) -> Tensor:
    """Pairwise distance loss on flattened backbone atom coordinates.

    Args:
        pred: Predicted coordinates ``(B, L, 3, 3)``.
        target: Target coordinates ``(B, L, 3, 3)``.
        mask: Optional residue mask ``(B, L)``.
        clamp_distance: Maximum distance (Å) used when comparing pairs.

    Returns:
        Scalar tensor containing the pairwise distance loss.
    """

    pred_flat = _flatten_backbone(pred)
    target_flat = _flatten_backbone(target)

    mask_flat = _broadcast_mask(mask, pred.size(1))
    if mask_flat is not None:
        mask_flat = mask_flat.to(device=pred.device, dtype=torch.bool)

    losses = []
    for i in range(pred_flat.size(0)):
        if mask_flat is not None:
            mask_i = mask_flat[i]
            pred_i = pred_flat[i, mask_i]
            target_i = target_flat[i, mask_i]
        else:
            pred_i = pred_flat[i]
            target_i = target_flat[i]

        if pred_i.size(0) < 2:
            losses.append(_zero_loss_like(pred_flat[i]))
            continue

        pred_dist = torch.cdist(pred_i, pred_i)
        target_dist = torch.cdist(target_i, target_i)

        pred_dist = torch.clamp(pred_dist, max=clamp_distance)
        target_dist = torch.clamp(target_dist, max=clamp_distance)

        losses.append((pred_dist - target_dist).pow(2).mean())

    return torch.stack(losses).mean()


def _backbone_vectors(coords: Tensor) -> Tensor:
    """Construct canonical backbone direction vectors for each residue.

    Args:
        coords: Coordinate tensor ``(B, L, 3, 3)``.

    Returns:
        Tensor ``(B, L, 6, 3)`` containing direction vectors per residue.
    """
    n = coords[:, :, 0, :]
    ca = coords[:, :, 1, :]
    c = coords[:, :, 2, :]

    v1 = ca - n
    v2 = c - ca
    v3 = torch.zeros_like(v1)
    v3[:, :-1] = n[:, 1:] - c[:, :-1]

    def _normalise(vec: Tensor) -> Tensor:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        return vec / torch.clamp(norm, min=1e-8)

    v1 = _normalise(v1)
    v2 = _normalise(v2)
    v3 = _normalise(v3)
    v4 = _normalise(-torch.cross(v1, v2, dim=-1))
    v5 = _normalise(torch.cross(v3, v1, dim=-1))
    v6 = _normalise(torch.cross(v2, v3, dim=-1))

    return torch.stack((v1, v2, v3, v4, v5, v6), dim=2)


def backbone_direction_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    clamp_value: float = 20.0,
) -> Tensor:
    """Loss on backbone direction signatures using pairwise dot products.

    Args:
        pred: Predicted coordinates ``(B, L, 3, 3)``.
        target: Target coordinates ``(B, L, 3, 3)``.
        mask: Optional residue mask ``(B, L)``.
        clamp_value: Clamp applied to dot products for numerical stability.

    Returns:
        Scalar tensor containing the direction signature loss.
    """

    pred_vectors = _backbone_vectors(pred)
    target_vectors = _backbone_vectors(target)

    losses = []
    if mask is not None:
        mask = mask.to(device=pred.device, dtype=torch.bool)

    for i in range(pred_vectors.size(0)):
        if mask is not None:
            valid = mask[i]
            pv = pred_vectors[i, valid]
            tv = target_vectors[i, valid]
        else:
            pv = pred_vectors[i]
            tv = target_vectors[i]

        if pv.size(0) < 2:
            losses.append(_zero_loss_like(pred_vectors[i]))
            continue

        pred_dot = torch.einsum("icd,jcd->ijc", pv, pv)
        target_dot = torch.einsum("icd,jcd->ijc", tv, tv)

        pred_dot = torch.clamp(pred_dot, min=-clamp_value, max=clamp_value)
        target_dot = torch.clamp(target_dot, min=-clamp_value, max=clamp_value)

        losses.append((pred_dot - target_dot).pow(2).mean())

    return torch.stack(losses).mean()


def reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    weights: Tuple[float, float, float] = (5e-3, 1e-2, 5e-2),
    return_components: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]] | Tensor:
    """Compute the weighted reconstruction loss following Algorithm 2.

    Args:
        pred: Predicted coordinates ``(B, L, 3, 3)``.
        target: Target coordinates ``(B, L, 3, 3)``.
        mask: Optional residue mask ``(B, L)``.
        weights: Tuple of weights for aligned MSE, distance, and direction losses.
        return_components: Return individual loss components when ``True``.

    Returns:
        Total reconstruction loss, optionally with component breakdown.
    """

    l_mse = aligned_mse(pred, target, mask=mask)
    l_dist = backbone_distance_loss(pred, target, mask=mask)
    l_dir = backbone_direction_loss(pred, target, mask=mask)

    total = weights[0] * l_mse + weights[1] * l_dist + weights[2] * l_dir

    if return_components:
        components = {
            "aligned_mse": l_mse,
            "distance": l_dist,
            "direction": l_dir,
            "total": total,
        }
        return total, components

    return total


__all__ = [
    "aligned_mse",
    "backbone_distance_loss",
    "backbone_direction_loss",
    "reconstruction_loss",
]
