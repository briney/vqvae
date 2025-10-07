"""Utilities for preprocessing backbone datasets for fast reuse."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .dataset import (
    BackboneDataset,
    PREPROCESSED_MANIFEST,
    PREPROCESSED_SAMPLES_DIR,
    PREPROCESSED_VERSION,
)

try:  # pragma: no cover - tqdm is optional
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: _to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_cpu(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu(item) for item in data)
    return data


def preprocess_dataset(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    chain_ids: Optional[Sequence[str]] = None,
    length_cap: int = 2048,
    k: int = 16,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Materialise a :class:`BackboneDataset` to disk for reuse.

    Parameters
    ----------
    input_root:
        Directory or file containing the raw structure data.
    output_dir:
        Directory where the preprocessed representation should be stored.
    chain_ids:
        Optional iterable of chain identifiers to retain.
    length_cap:
        Maximum chain length to load from the raw data.
    k:
        Neighbourhood size used when computing geometric features.
    overwrite:
        Whether to delete any existing data under ``output_dir``.
    progress:
        Show a progress bar while writing samples.

    Returns
    -------
    Path
        The path to the manifest describing the processed dataset.
    """

    output_path = Path(output_dir)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_path} already exists. Pass overwrite=True to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = BackboneDataset(
        input_root,
        chain_ids=chain_ids,
        length_cap=length_cap,
        k=k,
        cache=True,
        progress=progress,
    )

    samples_dir = output_path / PREPROCESSED_SAMPLES_DIR
    samples_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, object]] = []
    iterator = range(len(dataset))
    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(total=len(dataset), desc="Saving preprocessed samples")

    try:
        for index in iterator:
            sample = dataset[index]
            cpu_sample = _to_cpu(sample)
            file_name = f"{index:08d}.pt"
            sample_path = samples_dir / file_name
            torch.save(cpu_sample, sample_path)

            metadata = cpu_sample.get("metadata", {})
            if isinstance(metadata, dict):
                source_path = metadata.get("path")
                chain_id = metadata.get("chain_id")
                sequence = metadata.get("sequence")
            else:
                source_path = None
                chain_id = None
                sequence = None

            mask = cpu_sample.get("mask")
            length = None
            if isinstance(mask, torch.Tensor):
                length = int(mask.to(torch.bool).sum().item())
            entries.append(
                {
                    "file": str(Path(PREPROCESSED_SAMPLES_DIR) / file_name),
                    "source_path": source_path,
                    "chain_id": chain_id,
                    "sequence": sequence,
                    "length": length,
                }
            )

            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    manifest = {
        "version": PREPROCESSED_VERSION,
        "source": str(Path(input_root).resolve()),
        "length_cap": dataset.length_cap,
        "k": dataset.k,
        "chain_ids": sorted(set(chain_ids)) if chain_ids else None,
        "num_samples": len(entries),
        "entries": entries,
    }

    manifest_path = output_path / PREPROCESSED_MANIFEST
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest_path


__all__ = ["preprocess_dataset"]
