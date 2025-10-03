"""Checkpoint helpers wrapping :func:`torch.save` and :func:`torch.load`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Persist ``state`` to ``path`` creating parent directories as needed."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path_obj)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """Load and return a previously saved training state."""

    # ``torch.load`` defaulted ``weights_only`` to ``False`` prior to PyTorch 2.6,
    # which allowed arbitrary Python objects (such as OmegaConf ``DictConfig``
    # instances) to be deserialised.  The reference checkpoints for the GCPNet
    # encoder store their configuration in this format, so we explicitly request
    # the legacy behaviour to preserve compatibility with those files.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


__all__ = ["save_checkpoint", "load_checkpoint"]
