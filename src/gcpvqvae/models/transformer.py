"""Transformer backbone with rotary embeddings and grouped query attention."""

from __future__ import annotations

import torch
from torch import nn


class GCPTokensTransformer(nn.Module):
    """Placeholder transformer operating on latent tokens."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("GCPTokensTransformer needs implementation")

    def forward(self, *args, **kwargs):
        raise NotImplementedError
