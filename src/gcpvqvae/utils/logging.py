"""Minimal logging helpers used by the training harness."""

from __future__ import annotations

import logging


def get_logger(name: str = "gcpvqvae", level: int = logging.INFO) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    Args:
        name: Logger name used for retrieval.
        level: Logging level applied to the logger.

    Returns:
        Logger configured with a single stream handler and a concise formatter.
        Subsequent calls reuse the same handler to avoid duplicate messages.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler: logging.Handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
