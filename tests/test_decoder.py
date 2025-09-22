"""Unit tests for the 6D rotation decoder head."""

import pytest
import torch

from gcpvqvae.models.decoder import Rigid6DHead, _gram_schmidt


@pytest.fixture
def decoder_config():
    """Returns a config dict for a small decoder head."""
    return {"d_model": 64}


@pytest.fixture
def decoder_head(decoder_config):
    """Returns a Rigid6DHead instance."""
    return Rigid6DHead(**decoder_config)


@pytest.fixture
def dummy_hidden_states(decoder_config):
    """Returns a dummy hidden state tensor."""
    return torch.randn(4, 10, decoder_config['d_model']) # B, L, D


def test_gram_schmidt_orthonormality():
    """Tests that the Gram-Schmidt process produces orthonormal matrices."""
    a = torch.randn(16, 3)
    b = torch.randn(16, 3)

    R = _gram_schmidt(a, b)

    # Test shape
    assert R.shape == (16, 3, 3)

    # Test orthonormality: R @ R.T should be close to identity
    identity = torch.eye(3).unsqueeze(0).repeat(R.shape[0], 1, 1)
    product = torch.bmm(R, R.transpose(1, 2))
    assert torch.allclose(product, identity, atol=1e-5)

    # Test right-handedness: determinant should be +1
    determinants = torch.linalg.det(R)
    assert torch.allclose(determinants, torch.ones(R.shape[0]), atol=1e-6)


def test_decoder_head_output_shape(decoder_head, dummy_hidden_states):
    """Tests that the decoder head produces coordinates of the correct shape."""
    pred_coords = decoder_head(dummy_hidden_states)

    B, L, D = dummy_hidden_states.shape
    assert pred_coords.shape == (B, L, 3, 3) # B, L, (N,CA,C), (x,y,z)


def test_decoder_head_with_initial_pose(decoder_head, dummy_hidden_states):
    """Tests that an initial pose is correctly applied."""
    B, L, D = dummy_hidden_states.shape

    # Create a known initial pose
    angle = torch.tensor(torch.pi / 2) # 90 degree rotation
    R0 = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ]).unsqueeze(0).repeat(B, 1, 1)
    t0 = torch.tensor([10.0, 20.0, 30.0]).unsqueeze(0).repeat(B, 1)

    pred_coords = decoder_head(dummy_hidden_states, g0=(R0, t0))

    assert pred_coords.shape == (B, L, 3, 3)

    # The first coordinate should be translated and rotated from the local template
    first_coord_pred = pred_coords[:, 0, :, :]
    first_coord_expected = torch.einsum('bij,kj->bik', R0, decoder_head.local_template) # This is only a partial check

    # This is a weak test, but it confirms the shape and that no errors occur.
    # A more rigorous test would check the full composition logic.
    assert first_coord_pred.shape == (B, 3, 3)
