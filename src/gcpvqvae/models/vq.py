"""
Vector quantization layers with rotation-trick autograd and EMA updates.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class StraightThroughVQ(Function):
    """
    A simplified VQ autograd function that uses a straight-through estimator
    for the gradients, while matching the norm of the quantized vector.
    This is a dimension-agnostic version of the "rotation trick".
    """
    @staticmethod
    def forward(ctx, z, codebook):
        # Find nearest neighbors
        distances = torch.cdist(z, codebook)
        indices = torch.argmin(distances, dim=-1)
        z_q = F.embedding(indices, codebook)

        # Save for backward
        ctx.save_for_backward(z, z_q)
        return z_q, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        z, z_q = ctx.saved_tensors

        # Scale the gradient by the ratio of norms
        z_norm = torch.linalg.norm(z, dim=-1, keepdim=True)
        z_q_norm = torch.linalg.norm(z_q, dim=-1, keepdim=True)

        # Jacobian is a simple scalar
        Jk = z_q_norm / (z_norm + 1e-8)

        grad_z = grad_output * Jk
        return grad_z, None


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA updates, k-means init, and rotation-trick autograd.
    """
    def __init__(self, K, d_vq, beta_commit=0.25, ema_decay=0.99, orth_weight=1.0):
        super().__init__()
        self.K = K
        self.d_vq = d_vq
        self.beta_commit = beta_commit
        self.ema_decay = ema_decay
        self.orth_weight = orth_weight

        self.codebook = nn.Embedding(K, d_vq)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)

        self.register_buffer("ema_cluster_size", torch.zeros(K))
        self.register_buffer("ema_dw", self.codebook.weight.clone())
        self.register_buffer("initialized", torch.tensor(False))

    def _kmeans_init(self, z, iters=10):
        """Initialize codebook with k-means."""
        with torch.no_grad():
            # Select K random points as initial centroids
            if z.shape[0] < self.K:
                # Sample with replacement if we don't have enough points
                indices = torch.randint(0, z.shape[0], (self.K,))
            else:
                indices = torch.randperm(z.shape[0])[:self.K]
            self.codebook.weight.data.copy_(z[indices])

            for _ in range(iters):
                # Assign clusters
                distances = torch.cdist(z, self.codebook.weight)
                indices = torch.argmin(distances, dim=-1)

                # Update centroids
                for k in range(self.K):
                    cluster_points = z[indices == k]
                    if cluster_points.shape[0] > 0:
                        self.codebook.weight.data[k] = cluster_points.mean(dim=0)

            self.initialized.data.fill_(True)

    def forward(self, z):
        B, L, D = z.shape
        z_flat = z.reshape(-1, self.d_vq)

        if self.training and not self.initialized:
            self._kmeans_init(z_flat)

        z_q, indices = StraightThroughVQ.apply(z_flat, self.codebook.weight)

        loss_code = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_flat, z_q.detach())

        # EMA update
        if self.training:
            with torch.no_grad():
                # Update EMA cluster size
                one_hot_indices = F.one_hot(indices, num_classes=self.K).float()
                self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * one_hot_indices.sum(0)
                )

                # Update EMA codebook vectors
                dw = one_hot_indices.T @ z_flat
                self.ema_dw.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * dw)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed_cluster_size = (
                    (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
                )

                # Update codebook
                self.codebook.weight.data.copy_(self.ema_dw / smoothed_cluster_size.unsqueeze(1))

        # Orthogonality regularizer
        active_codes = self.codebook.weight[self.ema_cluster_size > 0]
        if active_codes.shape[0] > 1:
            orth_matrix = active_codes @ active_codes.T
            identity = torch.eye(orth_matrix.shape[0], device=z.device)
            loss_orth = self.orth_weight * F.mse_loss(orth_matrix, identity)
        else:
            loss_orth = torch.tensor(0.0, device=z.device)

        z_q = z_q.view(B, L, D)
        indices = indices.view(B, L)

        return {
            "z_q": z_q,
            "indices": indices,
            "loss_code": loss_code,
            "loss_commit": self.beta_commit * loss_commit,
            "loss_orth": loss_orth,
        }
