"""
Loss functions for geometry-aware training of protein structure models.
This includes aligned MSE, pairwise distance loss, and backbone direction loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from gcpvqvae.geometry.frames import kabsch


def _build_backbone_vectors(coords: torch.Tensor):
    """Computes 6 backbone unit vectors for the direction loss."""
    n, ca, c = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]

    # Pad for C_i -> N_{i+1} vector
    c_pad = torch.cat([c[:, :-1], torch.zeros_like(c[:, :1])], dim=1)
    n_pad = torch.cat([n[:, 1:], torch.zeros_like(n[:, :1])], dim=1)

    v1 = ca - n
    v2 = c - ca
    v3 = n_pad - c_pad

    v4 = -torch.cross(v1, v2, dim=-1)
    v5 = torch.cross(v3, v1, dim=-1)
    v6 = torch.cross(v2, v3, dim=-1)

    # Normalize all vectors
    vectors = torch.stack([
        F.normalize(v, dim=-1) for v in [v1, v2, v3, v4, v5, v6]
    ], dim=2)
    return vectors


def aligned_mse(x_pred, x_true, mask):
    """
    Computes Mean Squared Error after optimal alignment via Kabsch.
    """
    B = x_pred.shape[0]
    P_flat = x_pred.reshape(B, -1, 3)
    Q_flat = x_true.reshape(B, -1, 3)
    mask_flat = mask.repeat_interleave(3, dim=1)

    P_aligned_list = []
    for i in range(B):
        R, t = kabsch(P_flat[i], Q_flat[i], mask=mask_flat[i])
        # Align P to Q
        P_aligned_i = (R @ P_flat[i].T).T + t
        P_aligned_list.append(P_aligned_i)

    P_aligned = torch.stack(P_aligned_list, dim=0)
    x_pred_aligned = P_aligned.reshape_as(x_pred)

    # Compute masked MSE
    error = torch.sum((x_pred_aligned - x_true)**2, dim=(-1, -2))
    loss = torch.sum(error * mask) / (torch.sum(mask) * 9 + 1e-8)
    return loss


def backbone_distance_loss(x_pred, x_true, mask, clamp_dist=5.0):
    """
    Computes MSE on pairwise distances of flattened residue coordinates.
    """
    B, L, _, _ = x_pred.shape

    # Flatten to [B, L, 9]
    x_pred_flat = x_pred.reshape(B, L, 9)
    x_true_flat = x_true.reshape(B, L, 9)

    # Compute pairwise distance matrices
    dist_pred = torch.cdist(x_pred_flat, x_pred_flat)
    dist_true = torch.cdist(x_true_flat, x_true_flat)

    # Clamp distances
    dist_pred = torch.clamp(dist_pred, max=clamp_dist)
    dist_true = torch.clamp(dist_true, max=clamp_dist)

    # Compute loss on the upper triangle (to avoid double counting)
    error = (dist_pred - dist_true)**2

    # Create a mask for the pairwise matrix
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    loss = torch.sum(error * pair_mask) / (torch.sum(pair_mask) + 1e-8)
    return loss


def backbone_direction_loss(x_pred, x_true, mask, clamp_dot=20.0):
    """
    Computes MSE on pairwise dot products of backbone unit vectors.
    """
    # Get the 6 backbone vectors for pred and true
    vec_pred = _build_backbone_vectors(x_pred) # [B, L, 6, 3]
    vec_true = _build_backbone_vectors(x_true) # [B, L, 6, 3]

    # Compute pairwise dot products
    # Reshape to [B, L, 18]
    vec_pred_flat = vec_pred.reshape(vec_pred.shape[0], vec_pred.shape[1], -1)
    vec_true_flat = vec_true.reshape(vec_true.shape[0], vec_true.shape[1], -1)

    # dot_pred[i, j] = vec_pred[i] . vec_pred[j]
    dots_pred = vec_pred_flat @ vec_pred_flat.transpose(-1, -2)
    dots_true = vec_true_flat @ vec_true_flat.transpose(-1, -2)

    # Clamp
    dots_pred = torch.clamp(dots_pred, max=clamp_dot)
    dots_true = torch.clamp(dots_true, max=clamp_dot)

    # Compute loss
    error = (dots_pred - dots_true)**2
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    loss = torch.sum(error * pair_mask) / (torch.sum(pair_mask) + 1e-8)
    return loss


class ReconstructionLoss(nn.Module):
    """
    Combines all geometric reconstruction losses with specified weights.
    """
    def __init__(self, lambda_mse=1.0, lambda_dist=1.0, lambda_dir=1.0):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_dist = lambda_dist
        self.lambda_dir = lambda_dir

    def forward(self, pred_coords, true_coords, mask):
        l_mse = aligned_mse(pred_coords, true_coords, mask)
        l_dist = backbone_distance_loss(pred_coords, true_coords, mask)
        l_dir = backbone_direction_loss(pred_coords, true_coords, mask)

        total_loss = (
            self.lambda_mse * l_mse +
            self.lambda_dist * l_dist +
            self.lambda_dir * l_dir
        )

        return {
            "loss_rec": total_loss,
            "loss_mse": l_mse,
            "loss_dist": l_dist,
            "loss_dir": l_dir,
        }
