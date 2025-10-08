"""Top-level package for the GCP-VQVAE project."""

from importlib import import_module
from typing import Any

_MODEL_EXPORTS = {
    "GCPVQVAE",
    "GCPVQVAEConfig",
    "VectorQuantizerConfig",
    "RotationHeadConfig",
    "DataPipelineConfig",
}

__all__ = ["cli", *_MODEL_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        module = import_module(".models.gcpvqvae", package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MODEL_EXPORTS)
