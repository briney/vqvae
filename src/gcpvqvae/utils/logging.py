"""Logging utilities integrating console output and optional TensorBoard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# A simple flag to avoid a hard dependency on tensorboard.
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    A simple logger for training metrics.
    Logs to the console and optionally to TensorBoard if available.
    """
    def __init__(self, log_dir: str | Path | None = None):
        self.writer = None
        if TENSORBOARD_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: dict[str, Any], step: int, prefix: str = ""):
        """
        Logs a dictionary of metrics to the console and TensorBoard.

        Args:
            metrics: A dictionary of metric names to values.
            step: The current training step.
            prefix: An optional prefix for the metric names (e.g., 'train/', 'val/').
        """
        log_str = f"Step: {step}"
        for key, value in metrics.items():
            log_str += f" | {prefix}{key}: {value:.4f}"
            if self.writer:
                self.writer.add_scalar(f"{prefix}{key}", value, step)

        print(log_str)

    def close(self):
        """Closes the TensorBoard writer."""
        if self.writer:
            self.writer.close()
