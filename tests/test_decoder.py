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


def test_rotation_decoder_produces_valid_rotations() -> None:
    torch.manual_seed(0)
    decoder = RotationDecoder(9)

    latents = torch.randn(2, 5, 9)
    _, (R, _t) = decoder(latents)

    identity = torch.eye(3).expand(R.shape[0], 3, 3)
    rt_r = torch.matmul(R.transpose(-1, -2), R)
    assert torch.allclose(rt_r, identity, atol=1e-5)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)

    basis = torch.eye(3)
    rotated = torch.matmul(R, basis)
    norms = torch.linalg.norm(rotated, dim=-2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
