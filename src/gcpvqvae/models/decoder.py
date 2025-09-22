"""Rotation-based decoder mapping latents to backbone coordinates."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _normalize(vec: Tensor, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    return vec / torch.clamp(norm, min=eps)


class RotationDecoder(nn.Module):
    """6D rotation decoder head operating on transformer outputs.

    The module implements the rigid-body update described in Algorithm 1 of the
    GCP-VQVAE manuscript.  Given a stream of latent vectors it predicts per-step
    rotation and translation updates which are accumulated to reconstruct the
    backbone coordinates from an idealised local template.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        translation_scale: float = 1.0,
        template: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.translation_scale = translation_scale
        self.proj = nn.Linear(in_dim, 9, bias=True)

        if template is None:
            # Idealised backbone geometry (Å).  The coordinates are centred so
            # that the Cα atom sits at the origin which improves numerical
            # stability when composing transforms.
            template = torch.tensor(
                [
                    [-0.525, 0.000, 0.000],  # N
                    [0.000, 0.000, 0.000],   # Cα
                    [0.626, 0.000, 0.000],   # C
                ]
            )
        self.register_buffer("template", template, persistent=False)

    def forward(
        self,
        latents: Tensor,
        *,
        mask: Optional[Tensor] = None,
        init_pose: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Decode latent representations into backbone coordinates.

        Parameters
        ----------
        latents:
            Tensor of shape ``(batch, length, in_dim)``.
        mask:
            Optional boolean mask of shape ``(batch, length)``.  Masked positions
            are ignored and their coordinates are filled with zeros.
        init_pose:
            Optional tuple ``(R, t)`` providing the initial rigid transform for
            each sequence in the batch.  ``R`` has shape ``(batch, 3, 3)`` and
            ``t`` has shape ``(batch, 3)``.
        """

        if latents.ndim != 3:
            raise ValueError("latents must have shape (batch, length, dim)")

        batch, length, _ = latents.shape
        device = latents.device
        dtype = latents.dtype

        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=device)
        else:
            mask = mask.to(torch.bool)

        if init_pose is None:
            R = torch.eye(3, device=device, dtype=dtype).expand(batch, 3, 3).clone()
            t = torch.zeros((batch, 3), device=device, dtype=dtype)
        else:
            R, t = init_pose
            if R.shape != (batch, 3, 3) or t.shape != (batch, 3):
                raise ValueError("Initial pose must match batch dimensions")
            R = R.clone()
            t = t.clone()

        out_coords = []

        projected = self.proj(latents)
        translation, vec_a, vec_b = torch.split(projected, 3, dim=-1)

        for idx in range(length):
            active = mask[:, idx].view(batch, 1)
            if not active.any():
                out_coords.append(torch.zeros((batch, 3, 3), device=device, dtype=dtype))
                continue

            a = _normalize(vec_a[:, idx, :], eps=1e-6)
            b_raw = vec_b[:, idx, :]
            b = b_raw - (a * b_raw).sum(dim=-1, keepdim=True) * a
            needs_fallback = torch.linalg.norm(b, dim=-1, keepdim=True) < 1e-6
            if needs_fallback.any():
                fallback = torch.zeros((batch, 3), device=device, dtype=dtype)
                fallback[:, 1] = 1.0
                fallback = fallback - (fallback * a).sum(dim=-1, keepdim=True) * a
                b = torch.where(needs_fallback, fallback, b)
            b = _normalize(b, eps=1e-6)
            c = torch.cross(a, b, dim=-1)
            c = _normalize(c, eps=1e-6)
            b = torch.cross(c, a, dim=-1)

            R_local = torch.stack((a, b, c), dim=-1)  # (batch, 3, 3)

            t_local = self.translation_scale * translation[:, idx, :]

            R_candidate = R @ R_local
            t_candidate = (R @ t_local.unsqueeze(-1)).squeeze(-1) + t

            coords = torch.einsum("bij,aj->bai", R_candidate, self.template) + t_candidate.unsqueeze(-2)

            R = torch.where(active.view(batch, 1, 1), R_candidate, R)
            t = torch.where(active, t_candidate, t)
            coords = torch.where(active.view(batch, 1, 1), coords, torch.zeros_like(coords))
            out_coords.append(coords)

        stacked = torch.stack(out_coords, dim=1)
        final_pose = (R, t)

        return stacked, final_pose


__all__ = ["RotationDecoder"]
