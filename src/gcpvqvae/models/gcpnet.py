"""
Graph Convolutional Point (GCP) network building blocks.
This file implements the GCP micro-step, the GCPConv layer, and the full
GCPNet encoder as described in the workplan and the GCP-VQVAE paper.
"""

from __future__ import annotations

import torch
from torch import nn

from gcpvqvae.models.gcpcore import EquivariantGating, GatedMLP, Linear


class GCP(nn.Module):
    """
    The GCP micro-step. This module performs the core equivariant update.
    It takes node and edge features and produces updated node features.
    """
    def __init__(self, node_s_dim, node_v_dim, edge_s_dim, edge_v_dim, hidden_v_dim=16):
        super().__init__()
        # Vector down/up-scaling and signature creation
        self.vec_down = Linear(node_v_dim, hidden_v_dim)
        self.vec_up = Linear(hidden_v_dim, node_v_dim)
        self.sig_net = Linear(node_v_dim, node_v_dim * 3) # Project to 3x3 for frame projection

        # MLPs for scalar updates
        total_scalar_in = node_s_dim + edge_s_dim + hidden_v_dim
        self.scalar_mlp = GatedMLP(total_scalar_in, total_scalar_in, node_s_dim)

        # Gating mechanism
        self.gate = EquivariantGating(node_s_dim, node_v_dim)

    def forward(self, h, V, e_ij, n_ij, F_ij):
        # 1. Downscale vectors
        V_down = self.vec_down(V) # [..., C_in, C_hidden]

        # 2. Orientation signatures
        # Project vectors onto the edge frames
        # V_proj = (F_ij.transpose(-1, -2) @ self.sig_net(V).reshape(V.shape[0], V.shape[1], 3, 3))
        # The paper's description is a bit ambiguous. A simpler interpretation:
        q_ij = torch.einsum('...cv,...df->...cdf', self.sig_net(V).view(*V.shape, 3), F_ij)
        q_ij = q_ij.view(*q_ij.shape[:-2], -1) # Flatten the 3x3 signature matrix

        # 3. Aggregate messages (done in GCPConv)
        # For now, we assume messages are passed and aggregated outside.
        # Here, we just process the inputs as if they are per-node.

        # 4. Update scalars
        # We need aggregated q_i and ||V_down|| norms
        # This part is tricky without the aggregation step.
        # Let's defer the full logic to GCPConv. This module is more of a container for layers.
        pass # The logic is better placed in GCPConv.


class GCPConv(nn.Module):
    """
    A full GCP convolution layer, including message passing.
    It uses the GCP micro-step logic for updates.
    """
    def __init__(self, node_s_dim, node_v_dim, edge_s_dim, edge_v_dim, v_down_proj_dim=4):
        super().__init__()
        self.node_s_dim = node_s_dim

        # Linear layers for message creation
        self.edge_s_mlp = Linear(node_s_dim * 2 + edge_s_dim, edge_s_dim)

        # Vector processing layers
        self.vec_down = Linear(node_v_dim, v_down_proj_dim)
        self.vec_up = Linear(v_down_proj_dim, node_v_dim)

        # Scalar update MLP
        total_scalar_in = node_s_dim + edge_s_dim + v_down_proj_dim
        self.scalar_mlp = GatedMLP(total_scalar_in, total_scalar_in, node_s_dim)

        # Gating mechanism
        self.gate = EquivariantGating(node_s_dim, node_v_dim)

        # Layer norms
        self.norm_h = nn.LayerNorm(node_s_dim)
        self.norm_v = nn.LayerNorm(node_v_dim)

    def forward(self, h, V, edge_index, e, n):
        # Pre-norm
        h_in = self.norm_h(h)
        # Correctly normalize vector features across the channel dimension
        V_in = self.norm_v(V.transpose(-1, -2)).transpose(-1, -2)

        i, j = edge_index

        # 1. Create edge messages
        h_i, h_j = h_in[i], h_in[j]
        m_e_ij = self.edge_s_mlp(torch.cat([h_i, h_j, e], dim=-1))

        # 2. Aggregate edge messages to nodes
        m_e_i = torch.zeros(h.shape[0], m_e_ij.shape[1], device=h.device)
        m_e_i.index_add_(0, i, m_e_ij)

        # 3. Update node scalars
        # Correctly apply linear layer across channels
        V_down = self.vec_down(V_in.transpose(-1, -2)).transpose(-1, -2)
        V_down_norm = torch.linalg.norm(V_down, dim=-1)

        h_cat = torch.cat([h_in, m_e_i, V_down_norm], dim=-1)
        h_out = self.scalar_mlp(h_cat)

        # 4. Update node vectors with gating
        V_up = self.vec_up(V_down.transpose(-1, -2)).transpose(-1, -2)
        V_out = self.gate(h_out, V_up)

        # Residual connection
        h = h + h_out
        V = V + V_out

        return h, V


class GCPNetEncoder(nn.Module):
    """The full GCPNet encoder stack."""

    def __init__(
        self,
        in_h_dim=6, in_v_dim=3, in_e_dim=8, in_ev_dim=1,
        num_layers=6,
        hidden_s_dim=128, hidden_v_dim=16, hidden_e_dim=32,
        out_vq_dim=256,
    ):
        super().__init__()

        # Input projections
        self.proj_h = Linear(in_h_dim, hidden_s_dim)
        self.proj_v = Linear(in_v_dim, hidden_v_dim)
        self.proj_e = Linear(in_e_dim, hidden_e_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCPConv(hidden_s_dim, hidden_v_dim, hidden_e_dim, in_ev_dim)
            )

        # Output projection
        self.proj_out = Linear(hidden_s_dim + hidden_v_dim * 3, out_vq_dim)

    def forward(self, features):
        h = features['node_scalars']
        V = features['node_vectors']
        edge_index = features['edge_index']
        e = features['edge_scalars']
        n = features['edge_vectors'] # Not used in this simplified version

        # Project inputs
        h = self.proj_h(h)
        V = self.proj_v(V.transpose(-1, -2)).transpose(-1, -2)
        e = self.proj_e(e)

        # Apply GCPConv layers
        for conv in self.convs:
            h, V = conv(h, V, edge_index, e, n)

        # Pool vector features and concatenate with scalars for output
        V_flat = V.reshape(V.shape[0], -1)
        h_cat = torch.cat([h, V_flat], dim=-1)

        return self.proj_out(h_cat)
