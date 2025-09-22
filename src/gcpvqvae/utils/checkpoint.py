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


def load_checkpoint(path: str | Path, *, map_location: Optional[str | torch.device] = None) -> dict[str, Any]:
    """Load and return a previously saved training state."""

    return torch.load(Path(path), map_location=map_location)


__all__ = ["save_checkpoint", "load_checkpoint"]
