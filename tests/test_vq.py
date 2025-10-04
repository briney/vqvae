"""Unit tests for the vector-quantisation wrapper."""

from __future__ import annotations

import torch

from gcpvqvae.models.vq import VectorQuantizer, VectorQuantizerOptions


def _make_vq(dim: int, num_codes: int) -> VectorQuantizer:
    options = VectorQuantizerOptions(
        kmeans_init=False,
        kmeans_iters=1,
        stochastic_sample_codes=False,
        sample_codebook_temp=1.0,
    )
    return VectorQuantizer(
        num_codes=num_codes,
        dim=dim,
        beta=0.25,
        decay=0.99,
        epsilon=1e-5,
        rotation_trick=True,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_max_codes=num_codes,
        options=options,
    )


def test_vector_quantizer_forward_shapes() -> None:
    batch, length, dim, num_codes = 2, 4, 3, 8
    torch.manual_seed(0)
    vq = _make_vq(dim, num_codes)

    latents = torch.randn(batch, length, dim, requires_grad=True)
    quantized, indices, loss, metrics = vq(latents, return_metrics=True)

    assert quantized.shape == (batch, length, dim)
    assert indices.shape == (batch, length)
    assert {"commitment", "codebook", "orthogonality", "perplexity"} <= set(metrics)

    loss.backward()
    assert torch.isfinite(latents.grad).all()


def test_vector_quantizer_supports_masks() -> None:
    batch, length, dim, num_codes = 1, 6, 4, 16
    torch.manual_seed(1)
    vq = _make_vq(dim, num_codes)

    latents = torch.randn(batch, length, dim, requires_grad=True)
    mask = torch.tensor([[True, True, False, True, False, False]])

    quantized, indices, loss, metrics = vq(latents, mask=mask, return_metrics=True)

    assert quantized.shape == (batch, length, dim)
    assert torch.equal(indices[~mask], torch.full((3,), -1, dtype=torch.long))
    assert metrics["perplexity"].item() >= 1.0


def test_get_output_from_indices_matches_quantized_vectors() -> None:
    batch, length, dim, num_codes = 2, 5, 3, 32
    torch.manual_seed(2)
    vq = _make_vq(dim, num_codes)
    vq.eval()

    latents = torch.randn(batch, length, dim)
    with torch.no_grad():
        quantized, indices, _ = vq(latents)

    recovered = vq.get_output_from_indices(indices.reshape(-1))
    assert recovered.shape == (batch * length, dim)
    assert torch.allclose(recovered.view_as(quantized), quantized, atol=1e-5)
