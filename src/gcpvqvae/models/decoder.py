"""Rotation-based decoder mapping latents to backbone coordinates."""

from __future__ import annotations

import torch
from torch import nn


class RotationDecoder(nn.Module):
    """Placeholder 6D rotation decoder head."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("RotationDecoder needs implementation")

    def forward(self, *args, **kwargs):
        raise NotImplementedError
