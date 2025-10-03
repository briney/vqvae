"""Preprocess AlphaFold-style structures into backbone arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import gemmi
import numpy as np

from gcpvqvae.data.mmcif import THREE_TO_ONE


_BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")


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


def _extract_residue_data(residue: gemmi.Residue) -> Tuple[np.ndarray, Optional[np.ndarray], float, bool]:
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
                coords_rows.append(np.full((len(_BACKBONE_ATOMS), 3), np.nan, dtype=np.float64))
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


__all__ = ["PreprocessedChain", "preprocess_chain", "preprocess_structure"]
