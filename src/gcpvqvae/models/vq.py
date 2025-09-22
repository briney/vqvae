"""Vector quantization layers and codebook management."""

from __future__ import annotations

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """Placeholder VQ module with EMA updates."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("VectorQuantizer needs implementation")

    def forward(self, *args, **kwargs):
        raise NotImplementedError
