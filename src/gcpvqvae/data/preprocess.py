"""Preprocessing utilities for turning structure files into cached datasets."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gemmi
import h5py
import numpy as np
import torch
from tqdm import tqdm

from gcpvqvae.data.dataset import (
    PREPROCESSED_MANIFEST,
    PREPROCESSED_SAMPLES_DIR,
    PREPROCESSED_VERSION,
    BackboneDataset,
)
from gcpvqvae.data.mmcif import THREE_TO_ONE

_BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")
_MANIFEST_NAME = "preprocessed_reference_dataset.json"
_FILE_INDEX_NAME = "file_index.json"

# Missing-data thresholds applied during preprocessing.  AlphaFold structures
# occasionally contain short unresolved segments; we accept those while
# filtering chains with large gaps or pervasive missing coordinates.
_MAX_MISSING_RATIO = 0.20
_MAX_MISSING_BLOCK = 15


@dataclass
class PreprocessedChain:
    """Container for the backbone representation of a protein chain."""

    protein_seq: str
    coords: np.ndarray
    plddt: np.ndarray
    missing_residues: int

    def __post_init__(self) -> None:
        if self.coords.dtype != np.float64:
            raise TypeError("coords must have dtype float64")
        if self.plddt.dtype != np.float64:
            raise TypeError("plddt must have dtype float64")
        if self.coords.ndim != 3 or self.coords.shape[1:] != (len(_BACKBONE_ATOMS), 3):
            raise ValueError("coords must have shape (L, 4, 3)")
        if self.coords.shape[0] != self.plddt.shape[0]:
            raise ValueError("coords and plddt must have matching length")
        if len(self.protein_seq) != self.coords.shape[0]:
            raise ValueError("protein_seq length must match coordinate length")


def _missing_mask(chain: PreprocessedChain) -> np.ndarray:
    """Return a boolean mask marking residues without valid Cα positions."""

    # Missing residues manifest either through explicit NaNs introduced when the
    # raw residue lacked a backbone atom, or through gap padding when a jump in
    # the residue numbering indicated an unresolved stretch.
    ca_coords = chain.coords[:, 1, :]
    return np.isnan(ca_coords).any(axis=1)


def _longest_missing_block(mask: np.ndarray) -> int:
    """Compute the maximum length of a contiguous missing segment."""

    if mask.size == 0:
        return 0

    longest = 0
    current = 0
    for missing in mask:
        if bool(missing):
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _validate_length(
    length: int,
    *,
    min_len: Optional[int],
    max_len: Optional[int],
) -> Tuple[bool, Optional[str]]:
    """Check whether a chain length satisfies the configured bounds."""

    if min_len is not None and length < min_len:
        return False, "chains_too_short"
    if max_len is not None and length > max_len:
        return False, "chains_too_long"
    return True, None


def _validate_missing_thresholds(
    chain: PreprocessedChain,
    *,
    max_missing_ratio: float = _MAX_MISSING_RATIO,
    max_missing_block: int = _MAX_MISSING_BLOCK,
) -> Tuple[bool, Optional[str], float, int]:
    """Validate missing-coordinate statistics after gap padding."""

    mask = _missing_mask(chain)
    if mask.size == 0:
        return True, None, 0.0, 0

    missing_ratio = float(mask.mean())
    longest_block = _longest_missing_block(mask)

    if missing_ratio > max_missing_ratio:
        return False, "missing_ratio_exceeded", missing_ratio, longest_block
    if longest_block > max_missing_block:
        return False, "missing_block_exceeded", missing_ratio, longest_block
    return True, None, missing_ratio, longest_block


def _load_structure(path: Path) -> gemmi.Structure:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed = suffixes and suffixes[-1] == ".gz"
    if compressed:
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        structure = gemmi.read_structure(str(path))
    else:
        doc = gemmi.cif.read(str(path))
        block = doc.sole_block()
        structure = gemmi.make_structure_from_block(block)
    structure.setup_entities()
    return structure


def _iter_polymer_residues(chain: gemmi.Chain) -> Iterable[gemmi.Residue]:
    yield from chain.get_polymer()


def _seq_id(residue: gemmi.Residue) -> Optional[int]:
    try:
        return int(residue.seqid.num)
    except Exception:
        return None


def _select_atom(residue: gemmi.Residue, atom_name: str) -> Optional[gemmi.Atom]:
    best_atom: Optional[gemmi.Atom] = None
    best_occupancy = -math.inf
    for atom in residue:
        if atom.name.strip() != atom_name:
            continue
        occupancy = float(atom.occ)
        if occupancy > best_occupancy:
            best_atom = atom
            best_occupancy = occupancy
    return best_atom


def _extract_residue_data(
    residue: gemmi.Residue,
) -> Tuple[np.ndarray, Optional[np.ndarray], float, bool]:
    coords = np.full((len(_BACKBONE_ATOMS), 3), np.nan, dtype=np.float64)
    ca_position: Optional[np.ndarray] = None
    ca_plddt = math.nan
    complete = True

    for atom_index, atom_name in enumerate(_BACKBONE_ATOMS):
        atom = _select_atom(residue, atom_name)
        if atom is None:
            complete = False
            continue
        coords[atom_index] = (atom.pos.x, atom.pos.y, atom.pos.z)
        if atom_name == "CA":
            ca_position = coords[atom_index].copy()
            ca_plddt = float(atom.b_iso) if not math.isnan(atom.b_iso) else math.nan
    if ca_position is None:
        complete = False
    if math.isnan(ca_plddt):
        complete = False
    if not complete:
        coords[:] = np.nan
        ca_position = None
        ca_plddt = math.nan
    return coords, ca_position, ca_plddt, complete


def _gap_length(prev_seqid: Optional[int], current_seqid: Optional[int]) -> int:
    if prev_seqid is None or current_seqid is None:
        return 0
    gap = int(current_seqid) - int(prev_seqid)
    if gap <= 1:
        return 0
    return gap - 1


def _should_insert_gap(
    missing: int,
    prev_ca: Optional[np.ndarray],
    curr_ca: Optional[np.ndarray],
    gap_threshold: Optional[float],
) -> bool:
    if missing <= 0:
        return False
    if gap_threshold is None:
        return True
    if prev_ca is None or curr_ca is None:
        return True
    distance = float(np.linalg.norm(curr_ca - prev_ca))
    if math.isnan(distance):
        return True
    return distance > gap_threshold


def preprocess_chain(
    chain: gemmi.Chain,
    *,
    gap_threshold: Optional[float] = None,
) -> PreprocessedChain:
    residues = list(_iter_polymer_residues(chain))
    if not residues:
        raise ValueError(f"Chain {chain.name!r} does not contain polymer residues")

    coords_rows: List[np.ndarray] = []
    plddt_values: List[float] = []
    seq_chars: List[str] = []
    missing_residues = 0

    prev_seqid: Optional[int] = None
    prev_ca: Optional[np.ndarray] = None

    for residue in residues:
        seqid = _seq_id(residue)

        coords, ca_position, plddt, complete = _extract_residue_data(residue)

        gap = _gap_length(prev_seqid, seqid)
        if _should_insert_gap(gap, prev_ca, ca_position, gap_threshold):
            for _ in range(gap):
                coords_rows.append(
                    np.full((len(_BACKBONE_ATOMS), 3), np.nan, dtype=np.float64)
                )
                plddt_values.append(math.nan)
                seq_chars.append("X")
            missing_residues += gap

        three_letter = residue.name.upper()
        seq_chars.append(THREE_TO_ONE.get(three_letter, "X"))
        coords_rows.append(coords)
        plddt_values.append(plddt)
        if not complete:
            missing_residues += 1
            prev_ca = None
        else:
            prev_ca = ca_position
        prev_seqid = seqid

    protein_seq = "".join(seq_chars)
    coords_array = np.stack(coords_rows, axis=0).astype(np.float64, copy=False)
    plddt_array = np.asarray(plddt_values, dtype=np.float64)

    return PreprocessedChain(
        protein_seq=protein_seq,
        coords=coords_array,
        plddt=plddt_array,
        missing_residues=missing_residues,
    )


def preprocess_structure(
    path: str | Path,
    chain_id: str,
    *,
    gap_threshold: Optional[float] = None,
) -> PreprocessedChain:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    structure = _load_structure(path_obj)
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                return preprocess_chain(chain, gap_threshold=gap_threshold)
    raise ValueError(f"Chain {chain_id!r} not found in structure {path}")


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


def _write_h5(
    record: _ChainRecord, *, output_dir: Path, include_index: bool, index: int
) -> Path:
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


def _discover_structure_files(input_root: Path) -> List[Path]:
    if input_root.is_file():
        return [input_root]

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

    files: set[Path] = set()
    for pattern in patterns:
        files.update(input_root.rglob(pattern))

    return sorted(files, key=lambda path: path.as_posix())


def _is_polymer_chain(chain: gemmi.Chain) -> bool:
    residues = [
        res for res in chain.get_polymer() if THREE_TO_ONE.get(res.name.upper())
    ]
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

        missing_ok, missing_reason, ratio, longest = _validate_missing_thresholds(
            processed
        )
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


def preprocess_backbone_dataset(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    chain_ids: Optional[Sequence[str]] = None,
    length_cap: int = 2048,
    k: int = 16,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Materialise a :class:`BackboneDataset` to disk for reuse."""

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
    progress_bar = tqdm(
        total=len(dataset),
        desc="Saving preprocessed samples",
        disable=not progress,
    )

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

            progress_bar.update(1)
    finally:
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


def preprocess_dataset(
    input_root: Path,
    output_dir: Path,
    *,
    max_len: Optional[int] = None,
    min_len: Optional[int] = None,
    max_workers: Optional[int] = None,
    file_index: bool = True,
    gap_threshold: Optional[float] = None,
):
    """Preprocess AlphaFold-style structures into backbone summaries."""

    print(f"\nSearching for structure files in {input_root.resolve()}...", flush=True)
    files = _discover_structure_files(input_root)
    print(
        f"  found {len(files)} structure files.",
        flush=True,
    )
    if not files:
        raise ValueError(f"No structure files found in {input_root.resolve()}")

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir.resolve()} is not a directory")
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
        for file_path in tqdm(files, desc="Processing files", unit="file"):
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
            with tqdm(
                total=len(futures), desc="Processing files", unit="file"
            ) as progress:
                for future in as_completed(futures):
                    file_entries, file_stats = future.result()
                    entries.extend(file_entries)
                    stats.update(file_stats)
                    progress.update(1)

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


__all__ = [
    "PreprocessedChain",
    "preprocess_chain",
    "preprocess_structure",
    "preprocess_dataset",
    "preprocess_backbone_dataset",
]
