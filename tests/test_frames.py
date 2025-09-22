"""Unit tests for local frame construction and alignment helpers."""

from __future__ import annotations

import math

import torch

from gcpvqvae.geometry.frames import build_local_frames, kabsch_align


def _random_rotation(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / torch.linalg.norm(axis)
    angle = torch.rand(1, device=device, dtype=dtype) * 2 * math.pi
    K = torch.tensor(
        [
            [0.0, -axis[2].item(), axis[1].item()],
            [axis[2].item(), 0.0, -axis[0].item()],
            [-axis[1].item(), axis[0].item(), 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    R = torch.eye(3, device=device, dtype=dtype) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


def test_build_local_frames_right_handed() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [2.1, 0.3, 0.5],
            [3.1, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    frames = build_local_frames(coords, edge_index)

    assert frames.shape == (3, 3, 3)
    tangent = frames[:, :, 0]
    diff = coords[edge_index[1]] - coords[edge_index[0]]
    diff = diff / torch.linalg.norm(diff, dim=-1, keepdim=True)
    assert torch.allclose(tangent, diff, atol=1e-5)

    orthogonality = torch.matmul(frames.transpose(-1, -2), frames)
    identity = torch.eye(3).expand_as(orthogonality)
    assert torch.allclose(orthogonality, identity, atol=1e-5)

    det = torch.linalg.det(frames)
    assert torch.all(det > 0.99)


def test_kabsch_align_recovers_transform() -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    points = torch.randn(10, 3, device=device, dtype=dtype)
    rotation = _random_rotation(device, dtype)
    translation = torch.tensor([0.4, -1.2, 0.7], device=device, dtype=dtype)

    transformed = (points @ rotation) + translation

    R, t, aligned = kabsch_align(points, transformed, return_aligned=True)

    assert torch.allclose(R, rotation, atol=1e-6)
    assert torch.allclose(t, translation, atol=1e-6)
    assert aligned is not None and torch.allclose(aligned, transformed, atol=1e-6)
