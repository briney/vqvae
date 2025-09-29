"""Shared utilities for working with training and evaluation configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

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


def compose_overrides(config_path: Path, overrides: Iterable[str]) -> Dict[str, Any]:
    """Load ``config_path`` and apply Hydra-style ``overrides``.

    Parameters
    ----------
    config_path:
        Base configuration file to load.
    overrides:
        Iterable of override strings following Hydra's ``key=value`` syntax.

    Returns
    -------
    Dict[str, Any]
        A plain Python dictionary with all overrides applied and interpolations
        resolved.
    """

    config_path = Path(config_path)
    if not config_path.exists():  # pragma: no cover - handled by CLI before calling
        raise FileNotFoundError(config_path)

    overrides = list(overrides)
    config_dir = config_path.parent.resolve()
    config_name = config_path.stem

    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        global_hydra.clear()

    with initialize_config_dir(
        config_dir=str(config_dir), job_name="gpcvq_cli", version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides)

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):  # pragma: no cover - Hydra guarantees mapping
        raise TypeError("Hydra compose did not return a mapping")
    return cast(Dict[str, Any], container)


__all__ = ["build_model_config", "compose_overrides", "update_dataclass"]

