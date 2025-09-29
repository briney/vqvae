"""Shared utilities for working with training and evaluation configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast

try:  # Optional dependency – available in full installations
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
except ModuleNotFoundError:  # pragma: no cover - exercised when hydra is absent
    compose = None  # type: ignore[assignment]
    initialize_config_dir = None  # type: ignore[assignment]
    GlobalHydra = None  # type: ignore[assignment]

try:  # OmegaConf ships with hydra-core but we guard against it missing too
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - exercised when hydra is absent
    OmegaConf = None  # type: ignore[assignment]

import yaml

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


def _apply_override(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("Override key must not be empty")

    cursor = target
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise TypeError(
                f"Cannot override '{dotted_key}': '{part}' does not contain nested keys"
            )
        cursor = current

    cursor[parts[-1]] = value


def _parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}': expected key=value syntax")
    key, raw_value = override.split("=", 1)
    # ``yaml.safe_load`` gives us a reasonable approximation of Hydra's coercion rules.
    value = yaml.safe_load(raw_value)
    return key, value


def _fallback_compose(config_path: Path, overrides: Iterable[str]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    if not isinstance(base, dict):
        raise TypeError("Configuration file must contain a mapping")

    for override in overrides:
        key, value = _parse_override(override)
        _apply_override(base, key, value)

    return base


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

    if compose is None or initialize_config_dir is None or GlobalHydra is None:
        return _fallback_compose(config_path, overrides)

    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        global_hydra.clear()

    with initialize_config_dir(
        config_dir=str(config_dir), job_name="gpcvq_cli", version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides)

    if OmegaConf is None:  # pragma: no cover - hydra guarantees OmegaConf
        raise ModuleNotFoundError(
            "OmegaConf is required when hydra-core is installed; reinstall hydra-core"
        )

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):  # pragma: no cover - Hydra guarantees mapping
        raise TypeError("Hydra compose did not return a mapping")
    return cast(Dict[str, Any], container)


__all__ = ["build_model_config", "compose_overrides", "update_dataclass"]

