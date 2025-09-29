"""Unit tests for vector quantisation components."""

from __future__ import annotations

import pytest
import torch

from gcpvqvae.models.vq import VectorQuantizer, _rotation_trick_gradient


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
    assert torch.all(indices >= 0)
    assert torch.all(indices < vq.num_codes)
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


def test_rotation_trick_gradient_matches_closed_form() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(num_codes=4, dim=3, beta=0.25, decay=1.0, rotation_trick=True)
    with torch.no_grad():
        vq.embedding.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]))

    latents = torch.tensor(
        [
            [0.8, 0.3, 0.1],
            [0.0, 0.9, 0.4],
            [0.2, 0.1, 0.9],
            [-0.7, 0.2, 0.1],
        ],
        requires_grad=True,
        dtype=torch.float32,
    )

    quantized, _indices, _ = vq(latents)
    upstream = torch.randn_like(quantized)
    quantized.backward(upstream)

    expected = _rotation_trick_gradient(upstream, latents.detach(), quantized.detach(), vq.eps)
    assert torch.allclose(latents.grad, expected, atol=1e-5)


def test_vector_quantizer_perplexity_increases_after_updates() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(num_codes=4, dim=2, beta=0.1, decay=0.8, rotation_trick=False, kmeans_iters=0)
    with torch.no_grad():
        vq.embedding.zero_()

    uniform = torch.zeros(12, 2)
    vq.train()
    _, _, losses_before = vq(uniform)

    cluster_a = torch.tensor([[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]])
    cluster_b = torch.tensor([[-1.0, 0.2], [-0.9, -0.2], [-1.2, 0.0]])
    cluster_c = torch.tensor([[0.0, 1.0], [0.2, 1.1], [-0.1, 0.8]])

    for _ in range(30):
        latents = torch.cat([cluster_a, cluster_b, cluster_c], dim=0)
        noise = 0.05 * torch.randn_like(latents)
        vq(latents + noise)

    vq.eval()
    _, _, losses_after = vq(torch.cat([cluster_a, cluster_b, cluster_c], dim=0))

    assert losses_before["perplexity"].item() == pytest.approx(1.0, abs=1e-5)
    assert losses_after["perplexity"].item() > 2.0


def test_vector_quantizer_supports_orthogonal_regularisation_with_ema() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(
        num_codes=8,
        dim=4,
        beta=0.25,
        decay=0.95,
        rotation_trick=False,
        orthogonal_reg_weight=1.0,
    )

    latents = torch.randn(2, 16, 4, requires_grad=True)
    vq.train()
    _, _, losses = vq(latents)
    total = losses["commitment"] + losses["codebook"] + losses["orthogonality"]

    total.backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()

    # Applying the EMA update should not raise and should clear the pending tensor.
    vq.commit_pending_codebook()
    assert vq._pending_codebook is None
