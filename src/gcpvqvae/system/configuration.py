"""Shared utilities for working with training and evaluation configuration."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from gcpvqvae.models.gcpvqvae import GCPVQVAEConfig


def update_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    """Recursively apply ``updates`` to a dataclass ``instance``."""

    if not dataclasses.is_dataclass(instance) or not isinstance(updates, dict):
        return instance

    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            setattr(instance, key, update_dataclass(current, value))
        else:
            setattr(instance, key, value)
    return instance


def build_model_config(raw: Optional[Dict[str, Any]]) -> GCPVQVAEConfig:
    """Construct a :class:`GCPVQVAEConfig` from a raw dictionary."""

    config = GCPVQVAEConfig()
    if raw:
        for key, value in raw.items():
            if not hasattr(config, key):
                continue
            current = getattr(config, key)
            if dataclasses.is_dataclass(current):
                setattr(config, key, update_dataclass(current, value))
            else:
                setattr(config, key, value)
        config.rotation.input_dim = None
        config.__post_init__()
    return config


__all__ = ["build_model_config", "update_dataclass"]

