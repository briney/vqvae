"""Unit tests for vector quantisation components."""

from __future__ import annotations

import torch

from gcpvqvae.models.vq import VectorQuantizer


def test_vector_quantizer_assigns_nearest_codes() -> None:
    vq = VectorQuantizer(num_codes=3, dim=3, beta=0.25, decay=1.0, rotation_trick=True)
    with torch.no_grad():
        vq.embedding.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    latents = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.2, 0.7],
        ],
        requires_grad=True,
        dtype=torch.float32,
    )

    quantized, indices, losses = vq(latents)

    assert indices.tolist() == [0, 1, 2]
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(quantized, expected)
    assert {"commitment", "codebook", "orthogonality", "perplexity"} <= set(losses.keys())

    loss = quantized.sum() + losses["commitment"] + losses["codebook"]
    loss.backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


def test_vector_quantizer_supports_masks() -> None:
    vq = VectorQuantizer(num_codes=2, dim=2, beta=0.25, decay=1.0, rotation_trick=False)
    with torch.no_grad():
        vq.embedding.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    latents = torch.tensor([[0.9, 0.1], [0.0, 1.2]], requires_grad=True)
    mask = torch.tensor([True, False])

    quantized, indices, _ = vq(latents, mask=mask)

    assert indices.tolist() == [0, -1]
    assert torch.allclose(quantized[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(quantized[1], latents[1])  # straight-through when masked
