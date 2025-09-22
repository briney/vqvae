"""Geometric metrics and evaluation utilities."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from gcpvqvae.geometry.frames import kabsch


def rmsd(x_pred, x_true, mask):
    """
    Computes Root Mean Squared Deviation after optimal alignment.
    """
    # Flatten the atom dimension for Kabsch
    P = x_pred.reshape(x_pred.shape[0], -1, 3)
    Q = x_true.reshape(x_true.shape[0], -1, 3)

    # The mask needs to be repeated for each atom (N, CA, C)
    mask_flat = mask.repeat_interleave(3, dim=1)

    R, t = kabsch(P, Q, mask=mask_flat)

    # Align P to Q
    P_aligned = torch.einsum('bij,bkj->bki', R, P) + t.unsqueeze(1)

    # Compute masked squared error
    error = torch.sum((P_aligned - Q.reshape(Q.shape[0], -1, 3))**2, dim=-1)
    error_masked = torch.sum(error * mask_flat)

    # Compute RMSD
    num_points = torch.sum(mask_flat)
    mean_error = error_masked / (num_points + 1e-8)

    return torch.sqrt(mean_error)


def codebook_perplexity(indices: torch.Tensor) -> torch.Tensor:
    """
    Computes the perplexity of the codebook usage.
    A higher perplexity indicates that more codes are being used.
    """
    # Count the frequency of each index
    counts = torch.bincount(indices.flatten())
    # Compute the probability distribution
    probs = counts / (indices.numel() + 1e-8)
    # Compute entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    # Perplexity is the exponential of the entropy
    perplexity = torch.exp(entropy)
    return perplexity


def tm_score(coords_a, coords_b):
    """Approximate TM-score for backbones (stub)."""
    raise NotImplementedError("tm_score is not yet implemented")
