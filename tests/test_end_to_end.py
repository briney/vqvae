"""End-to-end integration smoke tests."""

from __future__ import annotations

import torch

from gcpvqvae.models.decoder import RotationDecoder
from gcpvqvae.models.vq import VectorQuantizer


def test_vq_decoder_pipeline_runs() -> None:
    batch, length, dim = 2, 4, 3
    vq = VectorQuantizer(num_codes=4, dim=dim, beta=0.1, decay=0.9, rotation_trick=True)
    decoder = RotationDecoder(dim, translation_scale=0.5)

    latents = torch.randn(batch, length, dim, requires_grad=True)
    quantized, indices, losses = vq(latents)
    coords, _ = decoder(quantized)

    assert coords.shape == (batch, length, 3, 3)
    assert indices.shape == (batch, length)
    total_loss = losses["commitment"] + losses["codebook"]
    total_loss.backward()
    assert torch.isfinite(latents.grad).all()
