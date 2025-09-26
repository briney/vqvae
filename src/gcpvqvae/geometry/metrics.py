"""Geometric metrics and evaluation utilities."""

from __future__ import annotations

import torch
from torch.nn import functional as F
import numpy as np

from gcpvqvae.geometry.frames import kabsch


def rmsd(x_pred, x_true, mask):
    """
    Computes Root Mean Squared Deviation after optimal alignment.
    This implementation loops over the batch.
    """
    batch_size = x_pred.shape[0]
    total_error = 0.0
    total_points = 0.0

    for i in range(batch_size):
        p_i = x_pred[i]
        q_i = x_true[i]
        m_i = mask[i]

        p_flat = p_i.reshape(-1, 3)
        q_flat = q_i.reshape(-1, 3)
        m_flat = m_i.repeat_interleave(3)

        R, t = kabsch(p_flat, q_flat, mask=m_flat)
        p_aligned = (R @ p_flat.T).T + t

        error = torch.sum((p_aligned - q_flat)**2, dim=-1)
        error_masked = torch.sum(error * m_flat)

        total_error += error_masked
        total_points += torch.sum(m_flat)

    mean_error = total_error / total_points.clamp(min=1e-8)
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


def tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the TM-score between two structures.
    Expects single, unbatched structures.
    """
    if mask is None:
        mask = torch.ones(pred_coords.shape[0], device=pred_coords.device)

    # Flatten coordinates for Kabsch
    P_flat = pred_coords.reshape(-1, 3)
    Q_flat = true_coords.reshape(-1, 3)
    mask_flat = mask.repeat_interleave(3)

    # Align coordinates using Kabsch
    R, t = kabsch(P_flat, Q_flat, mask=mask_flat)
    P_aligned_flat = (R @ P_flat.T).T + t
    aligned_coords = P_aligned_flat.reshape_as(pred_coords)

    # Use only C-alpha atoms for TM-score calculation
    ca_pred = aligned_coords[:, 1]
    ca_true = true_coords[:, 1]
    L_target = true_coords.shape[0]

    # Heuristic from TM-score paper, with a fix for short sequences
    len_for_d0 = max(0, L_target - 15)
    d0 = 1.24 * np.power(len_for_d0, 1.0 / 3.0) - 1.8
    if d0 < 0.5:
        d0 = 0.5

    # Pairwise distances between C-alpha atoms
    d_i = torch.norm(ca_pred - ca_true, dim=-1)

    # TM-score formula
    score = 1.0 / (1.0 + (d_i / d0) ** 2)
    tm = torch.sum(mask * score) / L_target

    return tm


def gdt_ts(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    cutoffs: list[float] | None = None,
) -> torch.Tensor:
    """
    Computes the Global Distance Test (GDT-TS) score.
    Expects single, unbatched structures.
    """
    if cutoffs is None:
        cutoffs = [1.0, 2.0, 4.0, 8.0]
    if mask is None:
        mask = torch.ones(pred_coords.shape[0], device=pred_coords.device)

    # Flatten coordinates for Kabsch
    P_flat = pred_coords.reshape(-1, 3)
    Q_flat = true_coords.reshape(-1, 3)
    mask_flat = mask.repeat_interleave(3)

    # Align coordinates
    R, t = kabsch(P_flat, Q_flat, mask=mask_flat)
    P_aligned_flat = (R @ P_flat.T).T + t
    aligned_coords = P_aligned_flat.reshape_as(pred_coords)

    # Use only C-alpha atoms for GDT
    ca_pred = aligned_coords[:, 1]
    ca_true = true_coords[:, 1]
    L = torch.sum(mask)

    # Pairwise distances
    distances = torch.norm(ca_pred - ca_true, dim=-1)

    # Calculate percentage of residues within each cutoff
    scores = []
    for cutoff in cutoffs:
        score = torch.sum((distances <= cutoff) * mask) / L
        scores.append(score)

    return torch.mean(torch.tensor(scores))