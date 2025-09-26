"""
Rotation-based decoder head that maps latent embeddings from a transformer
to backbone coordinates via a series of rigid transformations.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from gcpvqvae.models.gcpcore import Linear


def _gram_schmidt(a, b):
    """
    Orthonormalizes a and b to produce a right-handed rotation matrix.
    This is a numerically stable version of the Gram-Schmidt process.
    """
    # Normalize a
    a_norm = F.normalize(a, dim=-1)

    # Project b onto the plane orthogonal to a_norm and normalize
    b_ortho = b - torch.sum(a_norm * b, dim=-1, keepdim=True) * a_norm
    b_norm = F.normalize(b_ortho, dim=-1)

    # Compute the cross product to get the third basis vector
    c_norm = torch.cross(a_norm, b_norm, dim=-1)

    # Stack into a rotation matrix
    R = torch.stack([a_norm, b_norm, c_norm], dim=1)
    return R


class Rigid6DHead(nn.Module):
    """
    Decodes a sequence of embeddings into backbone coordinates using the
    6D rotation head method described in the workplan (Algorithm 1).
    """
    def __init__(self, d_model: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

        # Projection from transformer hidden state to the 9 numbers for 6D transform
        self.proj_6d = Linear(d_model, 9)

        # Idealized local backbone template (N, CA, C)
        # Based on standard bond lengths and angles. Placed in the XY plane.
        # This is a simplified representation.
        template = torch.tensor([
            [1.46, 0.0, 0.0],   # N
            [0.0, 0.0, 0.0],   # CA at origin
            [0.0, 1.52, 0.0],   # C
        ], dtype=torch.float32)
        self.register_buffer("local_template", template)

    def forward(self, H: torch.Tensor, g0: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Args:
            H: Hidden states from the decoder transformer of shape [B, L, d_model].
            g0: An optional initial pose (R, t) to start from. If None, starts
                from identity. R shape [B, 3, 3], t shape [B, 3].

        Returns:
            A tensor of predicted coordinates of shape [B, L, 3, 3].
        """
        if H.ndim == 2:
            H = H.unsqueeze(0)

        B, L, _ = H.shape

        # Initialize the running pose for each item in the batch
        if g0 is None:
            R_running = torch.eye(3, device=H.device).unsqueeze(0).repeat(B, 1, 1)
            t_running = torch.zeros(B, 3, device=H.device)
        else:
            R_running, t_running = g0

        # Project all hidden states to 6D representations at once
        params_6d = self.proj_6d(H) # [B, L, 9]

        all_coords = []
        # Iteratively apply the rigid updates for each residue in the sequence
        for i in range(L):
            params_i = params_6d[:, i, :]

            # Split into two 3D vectors and a translation
            a = params_i[:, 0:3]
            b = params_i[:, 3:6]
            t_update = params_i[:, 6:9]

            # Get the rotation for this step
            R_update = _gram_schmidt(a, b)

            # Scale translation
            t_update = self.alpha * t_update

            # Store old pose for use in translation update
            R_old = R_running

            # Compose with the running pose: g_new = g_old @ g_update
            # R_new = R_old @ R_update
            # t_new = t_old + R_old @ t_update
            R_running = torch.bmm(R_old, R_update)
            t_running = torch.bmm(R_old, t_update.unsqueeze(-1)).squeeze(-1) + t_running

            # Apply the new pose to the local template to get world coordinates
            coords_i = torch.einsum('bij,kj->bik', R_running, self.local_template) + t_running.unsqueeze(1)
            all_coords.append(coords_i)

        out = torch.stack(all_coords, dim=1)

        return out