"""Wrapper around :mod:`vector_quantize_pytorch` for project-specific usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

try:  # pragma: no cover - import guarded for clearer error messages
    from vector_quantize_pytorch import VectorQuantize
    from vector_quantize_pytorch import vector_quantize_pytorch as vqp
except ImportError as exc:  # pragma: no cover - tested indirectly via wrapper
    raise ImportError(
        "vector-quantize-pytorch is required. Install v1.23.2 or add it to your"
        " environment dependencies."
    ) from exc


@dataclass
class VectorQuantizerOptions:
    """Options controlling the behaviour of :class:`VectorQuantizer`."""

    kmeans_init: bool = True
    kmeans_iters: int = 10
    stochastic_sample_codes: bool = True
    sample_codebook_temp: float = 1.0
    orthogonal_reg_active_codes_only: bool = True
    return_zeros_for_masked_padding: bool = True


class VectorQuantizer(nn.Module):
    """Thin wrapper around :class:`~vector_quantize_pytorch.VectorQuantize`."""

    def __init__(
        self,
        num_codes: int,
        dim: int,
        *,
        beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        rotation_trick: bool = True,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_max_codes: Optional[int] = 512,
        options: Optional[VectorQuantizerOptions] = None,
    ) -> None:
        super().__init__()
        if num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        opts = options or VectorQuantizerOptions()

        self.num_codes = num_codes
        self.dim = dim
        self.commitment_weight = beta
        self.orthogonal_reg_weight = orthogonal_reg_weight

        self._vq = VectorQuantize(
            dim=dim,
            codebook_size=num_codes,
            commitment_weight=beta,
            decay=decay,
            eps=epsilon,
            kmeans_init=opts.kmeans_init,
            kmeans_iters=opts.kmeans_iters,
            stochastic_sample_codes=opts.stochastic_sample_codes,
            sample_codebook_temp=opts.sample_codebook_temp,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=opts.orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
            rotation_trick=rotation_trick,
            return_zeros_for_masked_padding=opts.return_zeros_for_masked_padding,
        )

    def forward(
        self,
        latents: Tensor,
        *,
        mask: Optional[Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Quantise latent embeddings via vector quantisation.

        Args:
            latents: Tensor of shape ``(B, L, D)`` with embedding dimension ``D``.
            mask: Optional boolean mask selecting valid positions.
            return_metrics: Include auxiliary loss metrics when ``True``.

        Returns:
            Quantised embeddings, code indices, and total loss. When
            ``return_metrics`` is ``True`` the final element is a dictionary of
            loss components instead of a scalar loss tensor.
        """

        if latents.size(-1) != self.dim:
            raise ValueError(f"Expected latent dim {self.dim}, got {latents.size(-1)}")

        mask_tensor = mask.to(torch.bool) if mask is not None else None
        quantized, indices, total_loss, breakdown = self._vq(
            latents,
            mask=mask_tensor,
            return_loss_breakdown=True,
        )

        indices = indices.to(torch.long)
        if indices.ndim > latents.ndim - 1:
            indices = indices.squeeze(-1)

        metrics = self._build_loss_dict(total_loss, breakdown)
        metrics["perplexity"] = self._perplexity(indices, mask_tensor, latents.dtype)

        if return_metrics:
            return quantized, indices, metrics["total"], metrics
        return quantized, indices, metrics["total"]

    def freeze_codebook(self) -> None:
        """Prevent further codebook updates."""

        self._vq.freeze_codebook = True

    def unfreeze_codebook(self) -> None:
        """Re-enable codebook updates."""

        self._vq.freeze_codebook = False

    def commit_pending_codebook(self) -> None:
        """Retained for backwards compatibility; no-op with library backend."""

        return None

    def get_output_from_indices(self, indices: Tensor) -> Tensor:
        """Decode quantised embeddings given integer ``indices``."""

        return self._vq.get_output_from_indices(indices)

    # ------------------------------------------------------------------ helpers
    def _build_loss_dict(
        self, total_loss: Tensor, breakdown: vqp.LossBreakdown
    ) -> Dict[str, Tensor]:
        """Assemble a dictionary of loss components from the backend output."""
        commit = breakdown.commitment * self.commitment_weight
        orth = breakdown.orthogonal_reg * self.orthogonal_reg_weight
        codebook = total_loss - commit - orth
        losses: Dict[str, Tensor] = {
            "total": total_loss,
            "commitment": commit,
            "codebook": codebook,
            "orthogonality": orth,
        }
        if hasattr(breakdown, "codebook_diversity"):
            losses["codebook_diversity"] = (
                breakdown.codebook_diversity * self._vq.codebook_diversity_loss_weight
            )
        if hasattr(breakdown, "inplace_optimize"):
            losses["inplace_optimize"] = breakdown.inplace_optimize
        return losses

    def _perplexity(
        self,
        indices: Tensor,
        mask: Optional[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        """Compute codebook perplexity over valid indices."""
        if mask is None:
            mask = indices >= 0
        else:
            mask = mask & (indices >= 0)

        if not mask.any():
            return torch.zeros((), device=indices.device, dtype=dtype)

        flat = indices[mask]
        counts = torch.bincount(flat, minlength=self.num_codes).to(dtype=torch.float32)
        probs = counts / counts.sum()
        nonzero = probs > 0
        entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
        perplexity = torch.exp(entropy)
        return perplexity.to(dtype=dtype)


__all__ = ["VectorQuantizer", "VectorQuantizerOptions"]
