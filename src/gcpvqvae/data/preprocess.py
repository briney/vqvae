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
_MANIFEST_NAME = "preprocessed_dataset.json"
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
    """Return a boolean mask marking residues without valid Cα positions.

    Args:
        chain: Preprocessed chain with ``coords`` field of shape ``(L, 4, 3)``.

    Returns:
        Boolean numpy array of shape ``(L,)`` where ``True`` denotes positions
        lacking CA coordinates.
    """

    # Missing residues manifest either through explicit NaNs introduced when the
    # raw residue lacked a backbone atom, or through gap padding when a jump in
    # the residue numbering indicated an unresolved stretch.
    ca_coords = chain.coords[:, 1, :]
    return np.isnan(ca_coords).any(axis=1)


def _longest_missing_block(mask: np.ndarray) -> int:
    """Compute the maximum length of a contiguous missing segment.

    Args:
        mask: Boolean mask indicating missing residues.

    Returns:
        Length of the longest consecutive ``True`` region in ``mask``.
    """

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
    """Check whether a chain length satisfies the configured bounds.

    Args:
        length: Observed chain length.
        min_len: Minimum allowable length; ``None`` disables the lower bound.
        max_len: Maximum allowable length; ``None`` disables the upper bound.

    Returns:
        Tuple ``(ok, reason)`` where ``ok`` indicates whether the chain satisfies
        the bounds and ``reason`` provides the rejection key used in statistics.
    """

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
    """Validate missing-coordinate statistics after gap padding.

    Args:
        chain: Processed chain containing NaN-padded coordinates.
        max_missing_ratio: Maximum allowed fraction of missing residues.
        max_missing_block: Maximum allowed consecutive run of missing residues.

    Returns:
        Tuple ``(ok, reason, ratio, longest)`` where ``ok`` flags whether the
        chain passes the thresholds, ``reason`` contains the stats key when it
        fails, ``ratio`` is the missing-residue proportion, and ``longest`` is
        the longest contiguous missing segment.
    """

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
    """Load a structure file into a Gemmi ``Structure`` object.

    Args:
        path: Path to an mmCIF or PDB file (optionally ``.gz`` compressed).

    Returns:
        Parsed Gemmi structure with entities instantiated.
    """
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
    """Yield polymer residues from a Gemmi chain.

    Args:
        chain: Chain from which polymer residues should be emitted.

    Returns:
        Iterator over polymer residues in sequence order.
    """
    yield from chain.get_polymer()


def _seq_id(residue: gemmi.Residue) -> Optional[int]:
    """Return the integer sequence identifier for a residue, if available.

    Args:
        residue: Residue whose ``seqid`` should be inspected.

    Returns:
        Integer sequence identifier or ``None`` if unavailable.
    """
    try:
        return int(residue.seqid.num)
    except Exception:
        return None


def _select_atom(residue: gemmi.Residue, atom_name: str) -> Optional[gemmi.Atom]:
    """Select the highest-occupancy atom with a given name from a residue.

    Args:
        residue: Residue to inspect.
        atom_name: Name of the atom (e.g., ``"CA"``).

    Returns:
        Gemmi atom with the best occupancy or ``None`` if missing.
    """
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
    """Extract backbone coordinates and reliability metadata from a residue.

    Args:
        residue: Gemmi residue from which to pull backbone atoms.

    Returns:
        Tuple ``(coords, ca_position, ca_plddt, complete)`` where ``coords`` is a
        ``(4, 3)`` array of backbone coordinates (NaN when missing),
        ``ca_position`` holds the CA coordinates when present, ``ca_plddt`` is
        the stored B-factor interpreted as pLDDT, and ``complete`` indicates
        whether all required atoms were observed.
    """
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
    """Compute the number of missing residues implied by sequence IDs.

    Args:
        prev_seqid: Sequence identifier of the previous residue.
        current_seqid: Sequence identifier of the current residue.

    Returns:
        Number of absent residues between the identifiers (0 when contiguous).
    """
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
    """Determine whether a padded gap should be inserted between residues.

    Args:
        missing: Count of missing residues inferred from sequence IDs.
        prev_ca: CA coordinates of the previous residue, if available.
        curr_ca: CA coordinates of the current residue, if available.
        gap_threshold: Maximum allowed CA-CA distance before treating the region
            as a gap. ``None`` always inserts when ``missing > 0``.

    Returns:
        ``True`` if a gap should be inserted, otherwise ``False``.
    """
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
    """Convert a Gemmi chain into a :class:`PreprocessedChain`.

    Args:
        chain: Polymer chain containing backbone coordinates.
        gap_threshold: Maximum allowed distance (Å) between neighbouring CA atoms
            before padding a gap; ``None`` pads for any missing residue index.

    Returns:
        Preprocessed chain with NaN-padded coordinates of shape ``(L, 4, 3)`` and
        pLDDT scores ``(L,)``.

    Raises:
        ValueError: If the chain contains no polymer residues.
    """
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
    """Preprocess a single chain from a structure file.

    Args:
        path: Path to an mmCIF or PDB file (optionally ``.gz`` compressed).
        chain_id: Identifier of the chain to preprocess.
        gap_threshold: Gap detection threshold passed to :func:`preprocess_chain`.

    Returns:
        :class:`PreprocessedChain` instance representing the requested chain.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the chain cannot be located.
    """
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
    """Return a filesystem-friendly stem for a structure path.

    Args:
        path: Path to a structure file.

    Returns:
        Safe filename stem without duplicate extensions.
    """
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).stem or path.stem


def _sanitise_chain_id(chain_id: str) -> str:
    """Sanitise a chain identifier for use in filenames.

    Args:
        chain_id: Original chain identifier.

    Returns:
        Chain identifier safe for filesystem use.
    """
    safe = chain_id.strip()
    if not safe:
        return "unknown"
    return safe.replace("/", "_").replace(" ", "_")


def _write_h5(
    record: _ChainRecord, *, output_dir: Path, include_index: bool, index: int
) -> Path:
    """Write a preprocessed chain to an HDF5 file.

    Args:
        record: Processed chain metadata and arrays.
        output_dir: Destination directory where the file will be created.
        include_index: Whether to prefix filenames with an incremental index.
        index: Unique index used when ``include_index`` is ``True``.

    Returns:
        Path to the written ``.h5`` file.
    """
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
    """Return a sorted list of structure files beneath ``input_root``.

    Args:
        input_root: Directory tree or single file containing structures.

    Returns:
        Sorted list of matching file paths.
    """
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
    """Return ``True`` if the chain contains at least one canonical residue.

    Args:
        chain: Chain to inspect.

    Returns:
        ``True`` if the chain contains canonical polymer residues.
    """
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
    """Process a structure and return cropped chains plus statistics.

    Args:
        file_path: Path to the structure file.
        min_len: Minimum allowed chain length.
        max_len: Maximum allowed chain length.
        gap_threshold: Gap threshold forwarded to :func:`preprocess_chain`.

    Returns:
        Tuple ``(records, stats)`` where ``records`` is a list of successfully
        preprocessed chains and ``stats`` is a :class:`collections.Counter`
        tracking processing outcomes.
    """
    stats = Counter()
    stats["files_total"] += 1

    try:
        structure = _load_structure(file_path)
    except Exception:
        stats["parsing_errors"] += 1
        return [], stats

    if not structure or len(structure) == 0:
        stats["parsing_errors"] += 1
        return [], stats

    polymer_chains: List[gemmi.Chain] = []
    model = structure[0] if structure else None
    if model is not None:
        for chain in model:
            if _is_polymer_chain(chain):
                polymer_chains.append(chain)

    if not polymer_chains:
        if model is None or len(model) == 0:
            stats["parsing_errors"] += 1
        else:
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
    """Write a manifest summarising the cached dataset.

    Args:
        output_dir: Directory where the manifest should be written.
        entries: Sequence of processed chain records.
        stats: Counter tracking preprocessing statistics.
        input_root: Root path that was scanned for structures.

    Returns:
        Path to the written manifest JSON file.
    """
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
    """Persist an index listing the raw structure files used as input.

    Args:
        output_dir: Directory where ``file_index.json`` should be written.
        files: Sequence of file paths included in the preprocessing run.
    """
    index_path = output_dir / _FILE_INDEX_NAME
    payload = {"files": [str(path) for path in files]}
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _to_cpu(data):
    """Detach tensors to CPU recursively within nested containers.

    Args:
        data: Arbitrary nested structure of tensors and Python containers.

    Returns:
        CPU-only copy of ``data`` preserving structure.
    """
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
    """Materialise a :class:`BackboneDataset` to disk for reuse.

    Args:
        input_root: Path to raw structures compatible with :class:`BackboneDataset`.
        output_dir: Destination directory for the cached tensors and manifest.
        chain_ids: Optional chain identifiers to retain.
        length_cap: Maximum residue count per chain.
        k: Nearest-neighbour parameter for edge construction.
        overwrite: Whether to delete pre-existing output directories.
        progress: Display progress bars while iterating the dataset.

    Returns:
        Path to the generated manifest file inside ``output_dir``.

    Raises:
        FileExistsError: If ``output_dir`` exists and ``overwrite`` is ``False``.
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
    """Preprocess AlphaFold-style structures into backbone summaries.

    Args:
        input_root: Directory tree or file containing structures to convert.
        output_dir: Destination directory for HDF5 samples and manifest.
        max_len: Maximum allowed chain length; ``None`` preserves full chains.
        min_len: Minimum allowed chain length; ``None`` disables the lower bound.
        max_workers: Number of worker processes for parallel preprocessing. Set
            to ``1`` for serial execution or ``None`` to auto-detect.
        file_index: Whether to generate ``file_index.json`` enumerating inputs.
        gap_threshold: Maximum CA-CA distance before inserting synthetic gaps.

    Returns:
        Tuple ``(manifest_path, stats)`` where ``manifest_path`` points to
        ``preprocessed_dataset.json`` and ``stats`` is a :class:`Counter`
        summarising preprocessing outcomes.

    Raises:
        ValueError: If no structure files are discovered.
        NotADirectoryError: If ``output_dir`` exists but is not a directory.
    """

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
