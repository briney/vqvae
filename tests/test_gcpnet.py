"""Tests for the minimal GCPNet building blocks."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gcpvqvae.models.gcpnet import GCPConv


def test_gcpconv_handles_dtype_mismatch() -> None:
    """The convolution should accept bfloat16 accumulators and float32 sources."""

    conv = GCPConv()
    node_scalars = torch.zeros(3, dtype=torch.bfloat16)
    dst = torch.tensor([0, 2, 1, 2], dtype=torch.long)
    edge_scalars = torch.tensor([1.0, -2.0, 0.5, 3.5], dtype=torch.float32)

    aggregated = conv(node_scalars, dst, edge_scalars)

    expected = node_scalars.clone()
    expected.index_add_(0, dst, edge_scalars.to(node_scalars.dtype))

    assert aggregated.dtype == node_scalars.dtype
    assert torch.equal(aggregated, expected)


def test_gcpconv_backpropagates_through_cast() -> None:
    """Casting the source tensor should not break gradients."""

    conv = GCPConv()
    node_scalars = torch.zeros(3, dtype=torch.bfloat16)
    dst = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    edge_scalars = torch.tensor([0.25, 0.5, -1.0, 1.0], dtype=torch.float32)
    edge_scalars.requires_grad_(True)

    aggregated = conv(node_scalars, dst, edge_scalars)
    loss = aggregated.to(torch.float32).sum()
    loss.backward()

    assert edge_scalars.grad is not None
    assert torch.allclose(edge_scalars.grad, torch.ones_like(edge_scalars))
