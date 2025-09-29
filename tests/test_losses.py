import torch

from gcpvqvae.models.losses import (
    aligned_mse,
    backbone_direction_loss,
    backbone_distance_loss,
    reconstruction_loss,
)


def _dummy_backbone(batch_size: int = 1, length: int = 1) -> torch.Tensor:
    coords = torch.zeros(batch_size, length, 3, 3, requires_grad=True)
    coords.data.uniform_(-1.0, 1.0)
    return coords


def test_aligned_mse_supports_fully_masked_batch() -> None:
    pred = _dummy_backbone()
    target = torch.zeros_like(pred)
    mask = torch.zeros(pred.shape[:2], dtype=torch.bool)

    loss = aligned_mse(pred, target, mask=mask)

    assert loss.requires_grad
    loss.backward()


def test_backbone_distance_loss_supports_fully_masked_batch() -> None:
    pred = _dummy_backbone()
    target = torch.zeros_like(pred)
    mask = torch.zeros(pred.shape[:2], dtype=torch.bool)

    loss = backbone_distance_loss(pred, target, mask=mask)

    assert loss.requires_grad
    loss.backward()


def test_backbone_direction_loss_supports_short_backbones() -> None:
    pred = _dummy_backbone()
    target = torch.zeros_like(pred)
    mask = torch.zeros(pred.shape[:2], dtype=torch.bool)

    loss = backbone_direction_loss(pred, target, mask=mask)

    assert loss.requires_grad
    loss.backward()


def test_reconstruction_loss_requires_grad_with_empty_mask() -> None:
    pred = _dummy_backbone()
    target = torch.zeros_like(pred)
    mask = torch.zeros(pred.shape[:2], dtype=torch.bool)

    loss = reconstruction_loss(pred, target, mask=mask)

    assert loss.requires_grad
    loss.backward()
