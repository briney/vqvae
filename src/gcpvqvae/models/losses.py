"""Geometry-aware reconstruction losses used by GCP-VQVAE."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from gcpvqvae.geometry.frames import kabsch_align


def _flatten_backbone(coords: Tensor) -> Tensor:
    if coords.ndim != 4 or coords.size(-2) != 3 or coords.size(-1) != 3:
        raise ValueError("Backbone tensors must have shape (B, L, 3, 3)")
    return coords.view(coords.size(0), coords.size(1) * 3, 3)


def _broadcast_mask(mask: Optional[Tensor], length: int) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError("Mask must have shape (B, L)")
    return mask.unsqueeze(-1).expand(-1, -1, 3).reshape(mask.shape[0], length * 3)


def aligned_mse(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute the aligned MSE between predicted and target backbones."""

    pred_flat = _flatten_backbone(pred)
    target_flat = _flatten_backbone(target)

    mask_flat = _broadcast_mask(mask, pred.size(1))

    losses = []
    for i in range(pred_flat.size(0)):
        mask_i: Optional[Tensor]
        if mask_flat is not None:
            mask_i = mask_flat[i]
            valid = mask_i.to(torch.bool)
            if valid.sum() < 3:
                losses.append(torch.zeros((), device=pred.device, dtype=pred.dtype))
                continue
        else:
            mask_i = None

        try:
            rotation, translation, _ = kabsch_align(
                pred_flat[i],
                target_flat[i],
                mask=mask_i,
                allow_reflections=False,
                return_aligned=False,
            )
        except ValueError:
            rotation = torch.eye(3, device=pred.device, dtype=pred.dtype)
            translation = torch.zeros(3, device=pred.device, dtype=pred.dtype)

        aligned = pred_flat[i] @ rotation + translation
        diff = aligned - target_flat[i]
        if mask_i is not None:
            diff = diff * mask_i.unsqueeze(-1)
            denom = torch.clamp(mask_i.sum(), min=1.0) * 1.0
        else:
            denom = diff.shape[0]
        losses.append((diff.square().sum(dim=-1).sum()) / denom)

    return torch.stack(losses).mean()


def distance_matrix_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    clamp_distance: float = 5.0,
) -> Tensor:
    """Pairwise distance loss on flattened residue representations."""

    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    losses = []
    for i in range(pred_flat.size(0)):
        if mask is not None:
            valid = mask[i].to(torch.bool)
            pred_i = pred_flat[i, valid]
            target_i = target_flat[i, valid]
        else:
            pred_i = pred_flat[i]
            target_i = target_flat[i]

        if pred_i.size(0) < 2:
            losses.append(torch.zeros((), device=pred.device, dtype=pred.dtype))
            continue

        pred_dist = torch.cdist(pred_i, pred_i)
        target_dist = torch.cdist(target_i, target_i)

        pred_dist = torch.clamp(pred_dist, max=clamp_distance)
        target_dist = torch.clamp(target_dist, max=clamp_distance)

        losses.append((pred_dist - target_dist).pow(2).mean())

    return torch.stack(losses).mean()


def _backbone_vectors(coords: Tensor) -> Tensor:
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


def direction_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Loss on backbone direction signatures."""

    pred_vectors = _backbone_vectors(pred)
    target_vectors = _backbone_vectors(target)

    losses = []
    for i in range(pred_vectors.size(0)):
        if mask is not None:
            valid = mask[i].to(torch.bool)
            pv = pred_vectors[i, valid]
            tv = target_vectors[i, valid]
        else:
            pv = pred_vectors[i]
            tv = target_vectors[i]

        if pv.size(0) < 2:
            losses.append(torch.zeros((), device=pred.device, dtype=pred.dtype))
            continue

        pred_dot = torch.einsum("icd,jcd->ijc", pv, pv)
        target_dot = torch.einsum("icd,jcd->ijc", tv, tv)
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
    """Compute the weighted reconstruction loss following Algorithm 2."""

    l_mse = aligned_mse(pred, target, mask=mask)
    l_dist = distance_matrix_loss(pred, target, mask=mask)
    l_dir = direction_loss(pred, target, mask=mask)

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
    "distance_matrix_loss",
    "direction_loss",
    "reconstruction_loss",
]
