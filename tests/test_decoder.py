"""Unit tests for the rotation decoder."""

from __future__ import annotations

import torch

from gcpvqvae.models.decoder import Dim6RotStructureHead


def test_structure_head_identity_template() -> None:
    head = Dim6RotStructureHead(9, decoder_output_scaling_factor=1.0)
    params = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    flat, aux = head._decode_params(params)

    coords = aux["coordinates"]
    template = head.template.index_select(0, torch.tensor([1, 0, 2]))
    assert flat.shape == (1, 1, 9)
    assert torch.allclose(coords[0, 0], template, atol=1e-6)


def test_structure_head_respects_mask() -> None:
    head = Dim6RotStructureHead(9, decoder_output_scaling_factor=1.0)
    params = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.5, 0.1, -0.2, -0.3, 0.4, 0.2],
                [2.0, -1.0, 0.5, 0.3, -0.2, 0.1, 0.4, -0.3, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False]])

    flat, aux = head._decode_params(params, mask=mask)

    coords = aux["coordinates"]
    rotations = aux["rotations"]
    translations = aux["translations"]

    assert torch.all(flat[0, 1] == 0.0)
    assert torch.all(coords[0, 1] == 0.0)
    assert torch.allclose(rotations[0, 1], torch.eye(3), atol=1e-6)
    assert torch.all(translations[0, 1] == 0.0)


def test_structure_head_scaling_factor() -> None:
    head = Dim6RotStructureHead(9, decoder_output_scaling_factor=2.5)
    params = torch.zeros(1, 1, 9)

    flat, aux = head._decode_params(params)
    coords = aux["coordinates"].reshape(1, 1, 9)

    assert torch.allclose(flat, coords * 2.5, atol=1e-6)


def test_structure_head_produces_orthonormal_rotations() -> None:
    torch.manual_seed(0)
    head = Dim6RotStructureHead(16)
    params = torch.randn(3, 7, 9)

    _, aux = head._decode_params(params)
    rotations = aux["rotations"].reshape(-1, 3, 3)

    identity = torch.eye(3)
    rt_r = torch.matmul(rotations.transpose(-1, -2), rotations)
    assert torch.allclose(rt_r, identity.expand_as(rt_r), atol=1e-5)
    det = torch.linalg.det(rotations)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)
