"""Compatibility wrapper for the reference preprocessing workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .preprocessing import preprocess_dataset


def preprocess_reference_dataset(
    input_root: Path,
    output_dir: Path,
    *,
    max_len: Optional[int] = None,
    min_len: Optional[int] = None,
    max_workers: Optional[int] = None,
    use_cif: bool = False,
    file_index: bool = True,
    gap_threshold: Optional[float] = None,
):
    """Temporarily back the reference CLI with the legacy implementation."""

    # The legacy preprocessing pipeline does not yet implement the reference-only
    # options. They are accepted for forward compatibility but ignored for now.
    # ``max_len`` maps onto the existing ``length_cap`` parameter.
    kwargs = {}
    if max_len is not None:
        kwargs["length_cap"] = max_len

    return preprocess_dataset(
        input_root,
        output_dir,
        overwrite=False,
        progress=True,
        **kwargs,
    )


__all__ = ["preprocess_reference_dataset"]
