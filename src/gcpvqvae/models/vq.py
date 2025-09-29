"""Vector quantization layers and codebook management."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _safe_normalize(vec: Tensor, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    return vec / torch.clamp(norm, min=eps)


def _orthogonal_basis(vec: Tensor) -> Tensor:
    """Return vectors orthogonal to ``vec`` (batched)."""

    if vec.ndim != 2:
        raise ValueError("vec must be of shape (N, D)")
    n, d = vec.shape
    if n == 0:
        return vec
    basis = torch.zeros_like(vec)
    # Select the axis with smallest magnitude to avoid degeneracy.
    indices = torch.argmin(torch.abs(vec), dim=-1)
    basis[torch.arange(n, device=vec.device), indices] = 1.0
    basis = basis - (basis * vec).sum(dim=-1, keepdim=True) * vec
    return _safe_normalize(basis)


def _rotation_trick_gradient(grad: Tensor, inputs: Tensor, codes: Tensor, eps: float) -> Tensor:
    """Apply the rotation-trick Jacobian to ``grad``.

    The operation is vectorised over the leading dimension of the tensors and
    works for arbitrary latent dimensionality.  For degenerate inputs (e.g.
    zero-norm latents) the fallback reduces to simple rescaling which matches
    the behaviour of the straight-through estimator.
    """

    if grad.numel() == 0:
        return grad

    z_norm = torch.linalg.norm(inputs, dim=-1, keepdim=True)
    c_norm = torch.linalg.norm(codes, dim=-1, keepdim=True)

    scale = c_norm / torch.clamp(z_norm, min=eps)

    a_hat = torch.where(z_norm > eps, inputs / torch.clamp(z_norm, min=eps), _orthogonal_basis(codes))
    b_hat = torch.where(c_norm > eps, codes / torch.clamp(c_norm, min=eps), _orthogonal_basis(inputs))

    cos_theta = torch.clamp((a_hat * b_hat).sum(dim=-1, keepdim=True), -1.0, 1.0)
    proj = b_hat - cos_theta * a_hat
    sin_theta = torch.linalg.norm(proj, dim=-1, keepdim=True)

    u = torch.where(sin_theta > eps, proj / torch.clamp(sin_theta, min=eps), _orthogonal_basis(a_hat))

    dot_ga = (grad * a_hat).sum(dim=-1, keepdim=True)
    dot_gu = (grad * u).sum(dim=-1, keepdim=True)

    term1 = sin_theta * (dot_gu * a_hat - dot_ga * u)
    term2 = (cos_theta - 1.0) * (dot_ga * a_hat + dot_gu * u)

    rotated_grad = grad + term1 + term2
    return scale * rotated_grad


class _RotateTrickPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, quantized: Tensor, mask: Tensor, eps: float) -> Tensor:
        ctx.save_for_backward(inputs, quantized, mask)
        ctx.eps = eps
        return quantized

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        inputs, quantized, mask = ctx.saved_tensors
        mask = mask.to(torch.bool)

        grad_inputs = torch.zeros_like(inputs)
        if mask.any():
            grad = grad_output[mask]
            inp = inputs[mask]
            codes = quantized[mask]
            grad_inputs[mask] = _rotation_trick_gradient(grad, inp, codes, ctx.eps)

        return grad_inputs, None, None, None


class VectorQuantizer(nn.Module):
    """Vector quantisation module with EMA codebook updates."""

    def __init__(
        self,
        num_codes: int,
        dim: int,
        *,
        beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_iters: int = 0,
        rotation_trick: bool = True,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_max_codes: int = 512,
    ) -> None:
        super().__init__()

        if num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = epsilon
        self.kmeans_iters = kmeans_iters
        self.rotation_trick = rotation_trick
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self.embedding = nn.Parameter(torch.randn(num_codes, dim))

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_codebook", torch.zeros(num_codes, dim))
        self.register_buffer("usage", torch.zeros(num_codes))

        self._initialised = False

    @staticmethod
    def _zero_loss_like(tensor: Tensor) -> Tensor:
        zero = torch.zeros((), device=tensor.device, dtype=tensor.dtype)
        if tensor.requires_grad:
            zero = zero.requires_grad_()
        return zero

    def _initialise_codebook(self, samples: Tensor) -> None:
        if self._initialised or samples.numel() == 0:
            return

        num_samples = samples.shape[0]
        if num_samples < self.num_codes:
            repeats = (self.num_codes + num_samples - 1) // num_samples
            samples = samples.repeat(repeats, 1)
        centroids = samples[: self.num_codes].clone()

        for _ in range(self.kmeans_iters):
            distances = (
                torch.sum(samples**2, dim=1, keepdim=True)
                + torch.sum(centroids**2, dim=1)
                - 2.0 * samples @ centroids.t()
            )
            assignment = distances.argmin(dim=1)
            for k in range(self.num_codes):
                mask = assignment == k
                if mask.any():
                    centroids[k] = samples[mask].mean(dim=0)

        with torch.no_grad():
            self.embedding.copy_(centroids[: self.num_codes])
        self._initialised = True

    def forward(
        self,
        latents: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if latents.size(-1) != self.dim:
            raise ValueError(f"Expected latent dim {self.dim}, got {latents.size(-1)}")

        original_shape = latents.shape
        flat_latents = latents.reshape(-1, self.dim)

        if mask is not None:
            mask = mask.to(torch.bool).reshape(-1)
        else:
            mask = torch.ones(flat_latents.shape[0], dtype=torch.bool, device=latents.device)

        valid_latents = flat_latents[mask]

        if self.training and self.kmeans_iters > 0 and not self._initialised:
            self._initialise_codebook(valid_latents.detach())

        if valid_latents.numel() == 0:
            quantized = torch.zeros_like(flat_latents)
            indices = torch.full((flat_latents.shape[0],), -1, dtype=torch.long, device=latents.device)
            zero_loss = self._zero_loss_like(flat_latents)
            losses = {
                "commitment": self.beta * zero_loss,
                "codebook": zero_loss,
                "orthogonality": zero_loss,
                "perplexity": torch.tensor(0.0, device=latents.device, dtype=latents.dtype),
            }
            quantized = quantized.reshape(original_shape)
            indices = indices.reshape(original_shape[:-1])
            return quantized, indices, losses

        distances = (
            torch.sum(valid_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1)
            - 2.0 * valid_latents @ self.embedding.t()
        )
        assignment = distances.argmin(dim=1)
        quantized_valid = self.embedding.index_select(0, assignment)

        full_quantized = torch.zeros_like(flat_latents)
        full_quantized[mask] = quantized_valid

        indices = torch.full((flat_latents.shape[0],), -1, dtype=torch.long, device=latents.device)
        indices[mask] = assignment

        full_quantized = torch.where(mask.unsqueeze(-1), full_quantized, flat_latents)

        diff = flat_latents - full_quantized.detach()
        commitment = (diff[mask] ** 2).mean()
        codebook_loss = ((flat_latents.detach() - full_quantized)[mask] ** 2).mean()

        if self.orthogonal_reg_weight > 0:
            num_codes = min(self.orthogonal_reg_max_codes, self.num_codes)
            codes = self.embedding[:num_codes]
            gram = codes @ codes.t()
            identity = torch.eye(num_codes, device=latents.device, dtype=latents.dtype)
            orth_loss = ((gram - identity) ** 2).mean()
        else:
            orth_loss = self._zero_loss_like(self.embedding)

        if self.training and self.decay < 1.0:
            with torch.no_grad():
                one_hot = F.one_hot(assignment, num_classes=self.num_codes).to(latents.dtype)
                cluster_size = one_hot.sum(dim=0)
                embed_sum = one_hot.t() @ valid_latents

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
                self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

                cluster_size = self.ema_cluster_size + self.eps
                n = cluster_size.sum()
                cluster_size = cluster_size / (n + self.num_codes * self.eps) * n
                updated = self.ema_codebook / cluster_size.unsqueeze(1)
                self.embedding.copy_(updated)

        one_hot = F.one_hot(assignment, num_classes=self.num_codes).to(latents.dtype)
        avg_probs = one_hot.mean(dim=0)
        nonzero = avg_probs > 0
        if nonzero.any():
            perplexity = torch.exp(-(avg_probs[nonzero] * torch.log(avg_probs[nonzero])).sum())
        else:
            perplexity = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        self.usage.copy_(avg_probs)

        if self.rotation_trick:
            full_quantized = _RotateTrickPass.apply(flat_latents, full_quantized, mask, self.eps)
        else:
            full_quantized = flat_latents + (full_quantized - flat_latents).detach()

        quantized = full_quantized.reshape(original_shape)
        indices = indices.reshape(original_shape[:-1])

        losses = {
            "commitment": self.beta * commitment,
            "codebook": codebook_loss,
            "orthogonality": self.orthogonal_reg_weight * orth_loss,
            "perplexity": perplexity,
        }

        return quantized, indices, losses


__all__ = ["VectorQuantizer"]
