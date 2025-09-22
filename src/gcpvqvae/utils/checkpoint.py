"""Checkpoint loading and saving utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    config: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Saves a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        step: The current training step.
        config: The configuration dictionary used for this run.
        path: The path to save the checkpoint to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    torch.save(state, path)
    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    path: str | Path,
    device: str,
) -> int:
    """
    Loads a training checkpoint to resume training.

    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        path: The path to the checkpoint file.
        device: The device to map the loaded tensors to.

    Returns:
        The step number to resume training from.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_step = ckpt['step']

    print(f"Loaded checkpoint from {path} at step {start_step}")
    return start_step
