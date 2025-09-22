"""Unit tests for the VectorQuantizer module."""

import pytest
import torch

from gcpvqvae.models.vq import VectorQuantizer


@pytest.fixture
def vq_config():
    """Returns a config dict for a small VQ module."""
    return {
        "K": 64,
        "d_vq": 32,
    }


@pytest.fixture
def vq_module(vq_config):
    """Returns a VectorQuantizer instance."""
    return VectorQuantizer(**vq_config)


@pytest.fixture
def dummy_latents(vq_config):
    """Returns a dummy latent tensor."""
    return torch.randn(4, 10, vq_config['d_vq']) # B, L, D


def test_vq_forward_pass(vq_module, dummy_latents):
    """Tests the basic forward pass of the VQ layer."""
    vq_module.eval() # Ensure k-means is not triggered
    vq_module.initialized.data.fill_(True) # Pretend it's initialized

    output = vq_module(dummy_latents)

    assert "z_q" in output
    assert "indices" in output
    assert "loss_code" in output

    z_q = output['z_q']
    indices = output['indices']

    assert z_q.shape == dummy_latents.shape
    assert indices.shape == dummy_latents.shape[:-1] # Should be [B, L]
    assert indices.min() >= 0
    assert indices.max() < vq_module.K


def test_vq_kmeans_init(vq_module, dummy_latents):
    """Tests that k-means initialization is triggered correctly."""
    assert not vq_module.initialized.item()

    vq_module.train() # Set to training mode
    _ = vq_module(dummy_latents)

    assert vq_module.initialized.item()


def test_vq_backward_pass(vq_module, dummy_latents):
    """Tests that gradients flow through the rotation-trick autograd function."""
    vq_module.train() # Must be in training mode for k-means init

    latents_with_grad = dummy_latents.clone().requires_grad_()

    output = vq_module(latents_with_grad)
    z_q = output['z_q']

    # Create a dummy loss and backpropagate
    loss = z_q.mean()
    loss.backward()

    assert latents_with_grad.grad is not None
    assert latents_with_grad.grad.shape == dummy_latents.shape
    # Check that gradients are not all zero
    assert torch.any(latents_with_grad.grad != 0)
