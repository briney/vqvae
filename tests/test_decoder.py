"""Unit tests for the rotation decoder."""

from __future__ import annotations

import torch

from gcpvqvae.models.decoder import RotationDecoder


def _identity_decoder(in_dim: int) -> RotationDecoder:
    decoder = RotationDecoder(in_dim, translation_scale=1.0)
    with torch.no_grad():
        decoder.proj.weight.zero_()
        decoder.proj.bias.zero_()
        decoder.proj.weight.copy_(torch.eye(9, in_dim))
    return decoder


def test_rotation_decoder_identity_frame() -> None:
    decoder = _identity_decoder(9)
    latents = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
    coords, (R, t) = decoder(latents)

    template = decoder.template
    assert torch.allclose(coords[0, 0], template, atol=1e-6)
    assert torch.allclose(R, torch.eye(3))
    assert torch.allclose(t, torch.zeros(3))


def test_rotation_decoder_accumulates_translations() -> None:
    decoder = _identity_decoder(9)
    latents = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    coords, (R, t) = decoder(latents)

    ca_positions = coords[0, :, 1, :]
    assert torch.allclose(ca_positions[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(ca_positions[1], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(t, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(R, torch.eye(3))


def test_rotation_decoder_respects_mask() -> None:
    decoder = _identity_decoder(9)
    latents = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False]])

    coords, (R, t) = decoder(latents, mask=mask)

    ca_positions = coords[0, :, 1, :]
    assert torch.allclose(ca_positions[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(ca_positions[1], torch.zeros(3))
    assert torch.allclose(t, torch.tensor([1.0, 0.0, 0.0]))
