"""Shared equivariant utility functions and modules for GCP layers."""

from __future__ import annotations

import torch
from torch import nn


class Linear(nn.Linear):
    """
    A linear layer with no bias, as specified in the workplan for all
    projection and FFN layers in the transformers and GCPNet.
    """
    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)


class GatedMLP(nn.Module):
    """
    A Gated Multi-Layer Perceptron used for scalar feature processing.
    This architecture is common in GNNs for stable and expressive updates.
    It uses a residual connection and a gating mechanism.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = Linear(in_features, hidden_features)
        self.gate = nn.Sequential(
            Linear(in_features, hidden_features),
            nn.Sigmoid()
        )
        self.layer2 = Linear(hidden_features, out_features)
        self.residual = Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        gated_x = self.layer1(x) * self.gate(x)
        return self.layer2(gated_x) + self.residual(x)


class EquivariantGating(nn.Module):
    """
    Performs the row-wise gating of vector features, a core component of
    SE(3)-equivariant networks like GCPNet.

    The gate is computed from scalar features and applied to each row
    (3D vector) of the vector features, preserving equivariance.
    """
    def __init__(self, scalar_features, vector_channels):
        """
        Args:
            scalar_features: The number of input scalar features used to compute the gate.
            vector_channels: The number of vector channels (rows) in the input vector features.
        """
        super().__init__()
        self.gate_mlp = nn.Sequential(
            Linear(scalar_features, vector_channels),
            nn.Sigmoid()
        )

    def forward(self, scalars, vectors):
        """
        Args:
            scalars: A tensor of scalar features [..., scalar_features].
            vectors: A tensor of vector features [..., vector_channels, 3].

        Returns:
            The gated vector features.
        """
        # Compute gates from scalar features.
        # The output shape will be [..., vector_channels].
        gates = self.gate_mlp(scalars)

        # Add a dimension for broadcasting and apply the gate.
        # [..., vector_channels] -> [..., vector_channels, 1]
        # This allows multiplying with the vectors of shape [..., vector_channels, 3].
        gated_vectors = vectors * gates.unsqueeze(-1)

        return gated_vectors
