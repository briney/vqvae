"""Unit tests for local frame construction and alignment helpers."""

from __future__ import annotations

import math

import pytest
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


def test_build_local_frames_equivariant_under_rigid_transform() -> None:
    torch.manual_seed(0)
    coords = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

    original = build_local_frames(coords, edge_index)

    rotation = _random_rotation(device=torch.device("cpu"), dtype=torch.float64).to(torch.float32)
    translation = torch.tensor([0.4, -0.7, 1.1])

    transformed = coords @ rotation + translation
    transformed_frames = build_local_frames(transformed, edge_index)

    relation = torch.matmul(transformed_frames, original.transpose(-1, -2))
    ortho = torch.matmul(relation.transpose(-1, -2), relation)
    identity = torch.eye(3).expand_as(ortho)
    assert torch.allclose(ortho, identity, atol=1e-5)
    assert torch.all(torch.linalg.det(relation) > 0.99)

    orthogonality = torch.matmul(transformed_frames.transpose(-1, -2), transformed_frames)
    identity = torch.eye(3).expand_as(orthogonality)
    assert torch.allclose(orthogonality, identity, atol=1e-5)
    assert torch.all(torch.linalg.det(transformed_frames) > 0.99)


def _set_seed() -> None:
    torch.manual_seed(0)


def test_kabsch_align_identity() -> None:
    _set_seed()
    device = torch.device("cpu")
    dtype = torch.float64
    points = torch.randn(10, 3, device=device, dtype=dtype)

    R, t, aligned = kabsch_align(points, points, return_aligned=True)

    assert torch.allclose(R, torch.eye(3, dtype=dtype), atol=1e-7)
    assert torch.allclose(t, torch.zeros(3, dtype=dtype), atol=1e-7)
    assert aligned is not None and torch.allclose(aligned, points, atol=1e-7)


def test_kabsch_align_recovers_transform() -> None:
    _set_seed()
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


def test_kabsch_align_pure_translation() -> None:
    _set_seed()
    device = torch.device("cpu")
    dtype = torch.float64
    points = torch.randn(7, 3, device=device, dtype=dtype)
    translation = torch.tensor([-0.2, 0.5, 1.3], device=device, dtype=dtype)

    transformed = points + translation
    R, t, aligned = kabsch_align(points, transformed, return_aligned=True)

    assert torch.allclose(R, torch.eye(3, dtype=dtype), atol=1e-7)
    assert torch.allclose(t, translation, atol=1e-7)
    assert aligned is not None and torch.allclose(aligned, transformed, atol=1e-7)


def test_kabsch_align_supports_masks() -> None:
    _set_seed()
    device = torch.device("cpu")
    dtype = torch.float64
    points = torch.randn(12, 3, device=device, dtype=dtype)
    rotation = _random_rotation(device, dtype)
    translation = torch.tensor([0.2, -0.8, 1.1], device=device, dtype=dtype)

    transformed = (points @ rotation) + translation

    mask = torch.tensor([True, True, True, False, True, False, True, True, False, True, False, True])

    R, t, aligned = kabsch_align(points, transformed, mask=mask, return_aligned=True)

    assert torch.allclose(R, rotation, atol=1e-6)
    assert torch.allclose(t, translation, atol=1e-6)
    assert aligned is not None and torch.allclose(aligned, transformed, atol=1e-6)


def test_kabsch_align_requires_three_points() -> None:
    _set_seed()
    points = torch.randn(5, 3, dtype=torch.float32)
    transformed = points.clone()
    mask = torch.tensor([True, True, False, False, False])

    with pytest.raises(ValueError):
        kabsch_align(points, transformed, mask=mask)


def test_kabsch_alignment_with_reflection_toggle() -> None:
    _set_seed()
    device = torch.device("cpu")
    dtype = torch.float64
    points = torch.randn(9, 3, device=device, dtype=dtype)

    reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype))
    reflected = points @ reflection

    R_forced, _, aligned_forced = kabsch_align(points, reflected, return_aligned=True)
    R_free, _, aligned_free = kabsch_align(points, reflected, allow_reflections=True, return_aligned=True)

    assert aligned_free is not None and torch.allclose(aligned_free, reflected, atol=1e-6)
    assert torch.allclose(R_free, reflection, atol=1e-6)

    assert torch.linalg.det(R_forced) > 0.0
    assert aligned_forced is not None
    forced_rmsd = torch.sqrt(torch.mean((aligned_forced - reflected) ** 2))
    free_rmsd = torch.sqrt(torch.mean((aligned_free - reflected) ** 2))
    assert forced_rmsd > free_rmsd * 10


def test_kabsch_align_float32_accuracy() -> None:
    _set_seed()
    device = torch.device("cpu")
    dtype = torch.float32
    points = torch.randn(15, 3, device=device, dtype=dtype)
    rotation = _random_rotation(device, torch.float64).to(dtype)
    translation = torch.tensor([-0.3, 0.6, -1.4], device=device, dtype=dtype)

    transformed = (points @ rotation) + translation
    R, t, aligned = kabsch_align(points, transformed, return_aligned=True)

    assert torch.allclose(R, rotation, atol=1e-5)
    assert torch.allclose(t, translation, atol=1e-5)
    assert aligned is not None and torch.allclose(aligned, transformed, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_kabsch_align_low_precision_dtypes(dtype: torch.dtype) -> None:
    _set_seed()
    device = torch.device("cpu")
    points = torch.randn(12, 3, device=device, dtype=dtype)
    rotation = _random_rotation(device, torch.float32).to(dtype)
    translation = torch.tensor([0.25, -0.4, 0.9], device=device, dtype=dtype)

    transformed = (points @ rotation) + translation
    R, t, aligned = kabsch_align(points, transformed, return_aligned=True)

    assert R.dtype == dtype
    assert t.dtype == dtype
    assert aligned is not None and aligned.dtype == dtype

    expected_rotation = rotation.to(torch.float32)
    expected_translation = translation.to(torch.float32)
    assert torch.allclose(R.to(torch.float32), expected_rotation, atol=5e-2, rtol=5e-2)
    assert torch.allclose(t.to(torch.float32), expected_translation, atol=5e-2, rtol=5e-2)
    assert torch.allclose(aligned.to(torch.float32), transformed.to(torch.float32), atol=5e-2, rtol=5e-2)
