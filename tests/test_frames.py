"""Unit tests for frame construction and geometric utilities."""

import pytest
import torch

from gcpvqvae.geometry.frames import build_edge_frames, centralize, kabsch


@pytest.fixture
def dummy_coords():
    """Returns a dummy coordinate tensor of shape [10, 3, 3]."""
    return torch.randn(10, 3, 3)


def test_centralize(dummy_coords):
    """Tests that centralization works correctly."""
    coords_cen, centroid = centralize(dummy_coords)
    # The centroid of the C-alpha atoms of the centralized coords should be near zero.
    new_centroid = torch.mean(coords_cen[:, 1], dim=0)
    assert torch.allclose(new_centroid, torch.zeros(3), atol=1e-6)
    assert coords_cen.shape == dummy_coords.shape


def test_kabsch_identity():
    """Tests Kabsch alignment on identical point clouds."""
    p = torch.randn(20, 3)
    R_kabsch, t_kabsch = kabsch(p, p)
    assert torch.allclose(R_kabsch, torch.eye(3), atol=1e-6)
    assert torch.allclose(t_kabsch, torch.zeros(3), atol=1e-6)


def test_kabsch_known_transform():
    """Tests Kabsch with a known rotation and translation."""
    p = torch.randn(20, 3)

    # Create a known rotation matrix (45 degrees around z-axis)
    angle = torch.tensor(torch.pi / 4)
    R_true = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ])
    t_true = torch.tensor([1.0, 2.0, -3.0])

    # Transform p to get q
    q = (R_true @ p.T).T + t_true

    R_kabsch, t_kabsch = kabsch(p, q)

    assert torch.allclose(R_kabsch, R_true, atol=1e-6)
    assert torch.allclose(t_kabsch, t_true, atol=1e-6)


def test_kabsch_no_reflection():
    """Tests that Kabsch corrects for reflections if not allowed."""
    p = torch.randn(20, 3)

    # Create a reflection matrix
    R_reflect = torch.eye(3)
    R_reflect[2, 2] = -1.0

    q = (R_reflect @ p.T).T

    R_kabsch, _ = kabsch(p, q, allow_reflections=False)

    # The determinant should be +1, not -1
    assert torch.linalg.det(R_kabsch) > 0


def test_build_edge_frames(dummy_coords):
    """Tests that edge frames are orthonormal and right-handed."""
    num_nodes = dummy_coords.shape[0]
    # Create a simple edge index (every node connected to the next)
    edge_index = torch.stack([
        torch.arange(num_nodes - 1),
        torch.arange(1, num_nodes)
    ], dim=0)

    frames = build_edge_frames(edge_index, dummy_coords)

    # Test shape
    assert frames.shape == (num_nodes - 1, 3, 3)

    # Test orthonormality: F @ F.T should be close to identity
    identity = torch.eye(3).unsqueeze(0).repeat(frames.shape[0], 1, 1)
    product = torch.bmm(frames, frames.transpose(1, 2))
    assert torch.allclose(product, identity, atol=1e-5)

    # Test right-handedness: determinant should be +1
    determinants = torch.linalg.det(frames)
    assert torch.allclose(determinants, torch.ones(frames.shape[0]), atol=1e-6)
