"""Compatibility wrapper for the reference preprocessing workflow."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import gemmi
import numpy as np

from gcpvqvae.data.mmcif import THREE_TO_ONE

from .reference.preprocessing import (
    PreprocessedChain,
    _load_structure,
    _validate_length,
    _validate_missing_thresholds,
    preprocess_chain,
)


_MANIFEST_NAME = "preprocessed_reference_dataset.json"
_FILE_INDEX_NAME = "file_index.json"


@dataclass
class _ChainRecord:
    """Summary of a successfully preprocessed chain."""

    source_path: str
    chain_id: str
    length: int
    sequence: str
    missing_residues: int
    missing_ratio: float
    longest_missing_block: int
    preprocessed: "PreprocessedChain" = field(repr=False)
    h5_path: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_path": self.source_path,
            "chain_id": self.chain_id,
            "length": self.length,
            "sequence": self.sequence,
            "missing_residues": self.missing_residues,
            "missing_ratio": self.missing_ratio,
            "longest_missing_block": self.longest_missing_block,
        }
        if self.h5_path is not None:
            payload["h5_path"] = self.h5_path
        return payload


def _structure_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).stem or path.stem


def _sanitise_chain_id(chain_id: str) -> str:
    safe = chain_id.strip()
    if not safe:
        return "unknown"
    return safe.replace("/", "_").replace(" ", "_")


def _write_h5(record: _ChainRecord, *, output_dir: Path, include_index: bool, index: int) -> Path:
    stem = _structure_stem(Path(record.source_path))
    chain_id = _sanitise_chain_id(record.chain_id)
    if include_index:
        file_name = f"{index:08d}_{stem}_chain_id_{chain_id}.h5"
    else:
        file_name = f"{stem}_chain_id_{chain_id}.h5"

    file_path = output_dir / file_name
    chain = record.preprocessed

    seq_bytes = np.array(chain.protein_seq, dtype=f"S{len(chain.protein_seq) or 1}")
    coords = np.asarray(chain.coords, dtype=np.float64)
    plddt = np.asarray(chain.plddt, dtype=np.float64)

    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("seq", data=seq_bytes)
        handle.create_dataset("N_CA_C_O_coord", data=coords)
        handle.create_dataset("plddt_scores", data=plddt)

    record.h5_path = file_name
    return file_path


def _discover_structure_files(input_root: Path, *, use_cif: bool) -> List[Path]:
    if input_root.is_file():
        return [input_root]

    if use_cif:
        patterns = ["*.cif", "*.cif.gz", "*.mmcif", "*.mmcif.gz"]
    else:
        patterns = [
            "*.cif",
            "*.cif.gz",
            "*.mmcif",
            "*.mmcif.gz",
            "*.pdb",
            "*.pdb.gz",
            "*.ent",
            "*.ent.gz",
        ]

    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_root.rglob(pattern)))
    return files


def _is_polymer_chain(chain: gemmi.Chain) -> bool:
    residues = [res for res in chain.get_polymer() if THREE_TO_ONE.get(res.name.upper())]
    return bool(residues)


def _process_structure_file(
    file_path: Path,
    *,
    min_len: Optional[int],
    max_len: Optional[int],
    gap_threshold: Optional[float],
) -> Tuple[List[_ChainRecord], Counter]:
    stats = Counter()
    stats["files_total"] += 1

    try:
        structure = _load_structure(file_path)
    except Exception:
        stats["parsing_errors"] += 1
        return [], stats

    polymer_chains: List[gemmi.Chain] = []
    if structure:
        model = structure[0]
        for chain in model:
            if _is_polymer_chain(chain):
                polymer_chains.append(chain)

    if not polymer_chains:
        stats["missing_coordinates"] += 1
        return [], stats

    if len({chain.name for chain in polymer_chains}) > 1:
        stats["complexes"] += 1
        return [], stats

    records: List[_ChainRecord] = []
    for chain in polymer_chains:
        stats["chains_total"] += 1
        processed = preprocess_chain(chain, gap_threshold=gap_threshold)
        if processed.missing_residues:
            stats["missing_coordinates"] += 1

        length = int(processed.coords.shape[0])
        length_ok, length_reason = _validate_length(
            length, min_len=min_len, max_len=max_len
        )
        if not length_ok:
            stats[length_reason] += 1  # type: ignore[index]
            continue

        missing_ok, missing_reason, ratio, longest = _validate_missing_thresholds(processed)
        if not missing_ok:
            stats[missing_reason] += 1  # type: ignore[index]
            continue

        stats["chains_written"] += 1
        record = _ChainRecord(
            source_path=str(file_path),
            chain_id=chain.name,
            length=length,
            sequence=processed.protein_seq,
            missing_residues=processed.missing_residues,
            missing_ratio=ratio,
            longest_missing_block=longest,
            preprocessed=processed,
        )
        records.append(record)

    return records, stats


def _write_manifest(
    output_dir: Path,
    *,
    entries: Sequence[_ChainRecord],
    stats: Counter,
    input_root: Path,
) -> Path:
    manifest = {
        "input_root": str(input_root.resolve()),
        "num_chains": len(entries),
        "stats": dict(stats),
        "chains": [entry.to_dict() for entry in entries],
    }
    manifest_path = output_dir / _MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def _write_file_index(output_dir: Path, files: Sequence[Path]) -> None:
    index_path = output_dir / _FILE_INDEX_NAME
    payload = {"files": [str(path) for path in files]}
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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
    """Preprocess AlphaFold-style structures into backbone summaries."""

    files = _discover_structure_files(input_root, use_cif=use_cif)
    if not files:
        raise ValueError(f"No structure files found under {input_root}")

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: List[_ChainRecord] = []
    stats = Counter()

    worker_count: Optional[int]
    if max_workers is None:
        worker_count = None
    elif max_workers <= 1:
        worker_count = 1
    else:
        worker_count = max_workers

    if worker_count in (None, 1):
        for file_path in files:
            file_entries, file_stats = _process_structure_file(
                file_path,
                min_len=min_len,
                max_len=max_len,
                gap_threshold=gap_threshold,
            )
            entries.extend(file_entries)
            stats.update(file_stats)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_structure_file,
                    file_path,
                    min_len=min_len,
                    max_len=max_len,
                    gap_threshold=gap_threshold,
                )
                for file_path in files
            ]
            for future in as_completed(futures):
                file_entries, file_stats = future.result()
                entries.extend(file_entries)
                stats.update(file_stats)

    entries.sort(key=lambda record: (record.source_path, record.chain_id))

    include_index = bool(file_index)
    for idx, record in enumerate(entries):
        _write_h5(record, output_dir=output_dir, include_index=include_index, index=idx)
        stats["h5_processed"] += 1

    manifest_path = _write_manifest(
        output_dir,
        entries=entries,
        stats=stats,
        input_root=input_root,
    )

    if file_index:
        _write_file_index(output_dir, files)

    return manifest_path, stats


__all__ = ["preprocess_reference_dataset"]

