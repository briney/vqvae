"""Minimal logging helpers used by the training harness."""

from __future__ import annotations

import logging


def get_logger(name: str = "gcpvqvae", level: int = logging.INFO) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    The helper ensures that multiple invocations do not attach duplicate
    handlers which would otherwise result in repeated log lines when modules
    import the utility from different places.  The default configuration keeps
    the output compact and console friendly which suits both interactive runs
    and automated tests.
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
