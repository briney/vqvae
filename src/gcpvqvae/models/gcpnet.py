"""Graph Convolutional Point (GCP) network building blocks."""

from __future__ import annotations

import torch
from torch import nn


class GCPConv(nn.Module):
    """Placeholder for the GCP convolution layer."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("GCPConv needs implementation")

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class GCPNetEncoder(nn.Module):
    """Placeholder for the GCPNet encoder stack."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("GCPNetEncoder needs implementation")

    def forward(self, *args, **kwargs):
        raise NotImplementedError
