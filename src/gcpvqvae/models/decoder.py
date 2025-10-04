"""Structure decoding head producing rigid frames from latent features."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _normalize(vec: Tensor, eps: float = 1e-5) -> Tensor:
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    return vec / torch.clamp(norm, min=eps)


def _gram_schmidt(vec_a: Tensor, vec_b: Tensor, eps: float = 1e-5) -> Tensor:
    """Construct an orthonormal frame using Gram–Schmidt orthogonalisation."""

    u1 = _normalize(vec_a, eps)
    proj = (u1 * vec_b).sum(dim=-1, keepdim=True) * u1
    u2 = vec_b - proj

    needs_fallback = torch.linalg.norm(u2, dim=-1, keepdim=True) < eps
    if needs_fallback.any():
        fallback = torch.zeros_like(u2)
        fallback[..., 1] = 1.0
        fallback = fallback - (fallback * u1).sum(dim=-1, keepdim=True) * u1
        u2 = torch.where(needs_fallback, fallback, u2)

    u2 = _normalize(u2, eps)
    u3 = torch.cross(u1, u2, dim=-1)
    u3 = _normalize(u3, eps)
    u2 = torch.cross(u3, u1, dim=-1)

    return torch.stack((u1, u2, u3), dim=-1)


class Dim6RotStructureHead(nn.Module):
    """Decode latent representations into rigid frames and backbone coordinates."""

    def __init__(
        self,
        in_dim: int,
        *,
        template: Optional[Tensor] = None,
        decoder_output_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.output_scale = decoder_output_scaling_factor

        self.linear = nn.Linear(in_dim, in_dim, bias=True)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(in_dim)
        self.out_proj = nn.Linear(in_dim, 9, bias=True)

        if template is None:
            template = torch.tensor(
                [
                    [0.5256, 1.3612, 0.0],  # Cα
                    [0.0, 0.0, 0.0],       # N
                    [-1.5251, 0.0, 0.0],   # C
                ]
            )
        self.register_buffer("template", template, persistent=False)
        self.register_buffer(
            "reorder_index", torch.tensor([1, 0, 2], dtype=torch.long), persistent=False
        )

    def _decode_params(
        self,
        params: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if params.ndim != 3 or params.size(-1) != 9:
            raise ValueError("params must have shape (batch, length, 9)")

        batch, length, _ = params.shape
        device = params.device
        dtype = params.dtype

        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=device)
        else:
            mask = mask.to(torch.bool)

        translation, hint_a, hint_b = torch.split(params, 3, dim=-1)
        hint_a = hint_a + translation
        hint_b = hint_b + translation

        rotations = _gram_schmidt(hint_a, hint_b)

        template = self.template.to(device=device, dtype=dtype).transpose(0, 1)
        rotated = torch.matmul(rotations, template).transpose(-1, -2)
        coords = rotated + translation.unsqueeze(-2)

        # Reorder to the conventional (N, Cα, C) atom ordering expected downstream.
        coords = coords.index_select(-2, self.reorder_index.to(device=device))

        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        coords = torch.where(mask_expanded, coords, torch.zeros_like(coords))

        identity = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        rotations = torch.where(mask_expanded, rotations, identity)
        translation = torch.where(mask.unsqueeze(-1), translation, torch.zeros_like(translation))

        flat = coords.reshape(batch, length, 9)
        scaled = flat * self.output_scale

        aux = {
            "rotations": rotations,
            "translations": translation,
            "coordinates": coords,
            "mask": mask,
        }
        return scaled, aux

    def forward(
        self,
        latents: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if latents.ndim != 3 or latents.size(-1) != self.in_dim:
            raise ValueError("latents must have shape (batch, length, in_dim)")

        hidden = self.linear(latents)
        hidden = self.activation(hidden)
        hidden = self.norm(hidden)
        params = self.out_proj(hidden)

        return self._decode_params(params, mask=mask)


__all__ = ["Dim6RotStructureHead"]
