"""Utilities for reading and writing mmCIF structures.

This module provides a small, Torch-friendly representation of protein
backbone chains together with helpers to read mmCIF files.  Only the atoms
required by the GCP-VQVAE model (backbone N, CA and C) are extracted; all
other data are ignored.  The reader prefers :mod:`gemmi` but falls back to
Biopython when the former is unavailable.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

Tensor = torch.Tensor

try:  # pragma: no cover - exercised in integration tests
    import gemmi
except Exception:  # pragma: no cover - gemmi is an optional dependency
    gemmi = None

try:  # pragma: no cover - exercised in integration tests
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore
except Exception:  # pragma: no cover - Biopython is optional
    MMCIFParser = None
    PDBParser = None


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}
AA_TO_INDEX["X"] = len(AA_TO_INDEX)
PAD_INDEX = len(AA_TO_INDEX)

THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

CANONICAL_RESIDUES = set(THREE_TO_ONE.keys())

ONE_TO_THREE: Dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()}
ONE_TO_THREE["X"] = "UNK"


_PDB_SUFFIXES = {".pdb", ".ent"}


def _split_suffixes(path: Path) -> Tuple[List[str], bool]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed = False
    if suffixes and suffixes[-1] == ".gz":
        compressed = True
        suffixes = suffixes[:-1]
    return suffixes, compressed


def _detect_format(path: Path) -> str:
    suffixes, _ = _split_suffixes(path)
    suffix = suffixes[-1] if suffixes else path.suffix.lower()
    if suffix in _PDB_SUFFIXES:
        return "pdb"
    return "mmcif"


@dataclass
class BackboneRecord:
    """Container holding the minimal backbone information for a chain."""

    path: str
    chain_id: str
    coords: Tensor  # (L, 3, 3)
    mask: Tensor  # (L,)
    atom_mask: Tensor  # (L, 3)
    seq: Tensor  # (L,)
    seq_string: str
    residue_names: List[str]
    residue_ids: List[Tuple[int, str]]
    rotation: Tensor  # (3, 3)
    translation: Tensor  # (3,)
    nan_mask: Tensor  # (L,)

    @property
    def length(self) -> int:
        return int(self.coords.shape[0])


def _normalise_coords(coords: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Centralise the coordinates and return the applied rigid transform."""

    ca_mask = mask.to(torch.bool)
    ca_positions = coords[:, 1, :]
    if ca_mask.any():
        centroid = ca_positions[ca_mask].mean(dim=0)
    else:
        centroid = ca_positions.mean(dim=0)
    coords = coords - centroid.view(1, 1, 3)
    rotation = torch.eye(3, dtype=coords.dtype, device=coords.device)
    return coords, rotation, centroid


def _build_records_from_gemmi_structure(
    structure: "gemmi.Structure",
    path: Path,
    *,
    length_cap: int,
    chain_id: Optional[str],
) -> List[BackboneRecord]:
    records: List[BackboneRecord] = []

    for model in structure:
        for chain in model:
            chain_name = chain.name
            if chain_id is not None and chain_name != chain_id:
                continue

            residues = [res for res in chain.get_polymer() if res.name.upper() in CANONICAL_RESIDUES]
            if not residues:
                continue

            if length_cap and len(residues) > length_cap:
                residues = residues[:length_cap]

            coords_list: List[List[float]] = []
            mask_list: List[bool] = []
            atom_mask_list: List[List[bool]] = []
            seq_indices: List[int] = []
            seq_chars: List[str] = []
            residue_names: List[str] = []
            residue_ids: List[Tuple[int, str]] = []

            for residue in residues:
                three_letter = residue.name.upper()
                if three_letter not in CANONICAL_RESIDUES:
                    continue
                one_letter = THREE_TO_ONE.get(three_letter, "X")
                seq_chars.append(one_letter)
                seq_indices.append(AA_TO_INDEX.get(one_letter, AA_TO_INDEX["X"]))
                residue_names.append(three_letter)
                residue_ids.append((residue.seqid.num, residue.seqid.icode))

                atom_coords = {
                    "N": (None, -float("inf")),
                    "CA": (None, -float("inf")),
                    "C": (None, -float("inf")),
                }

                for atom in residue:
                    name = atom.name.strip()
                    if name not in atom_coords:
                        continue
                    occupancy = atom.occ
                    if occupancy < atom_coords[name][1]:
                        continue
                    atom_coords[name] = ((atom.pos.x, atom.pos.y, atom.pos.z), occupancy)

                coords_row: List[List[float]] = []
                atom_mask = []
                for atom_name in ("N", "CA", "C"):
                    pos, _ = atom_coords[atom_name]
                    if pos is None:
                        coords_row.append([0.0, 0.0, 0.0])
                        atom_mask.append(False)
                    else:
                        coords_row.append(list(pos))
                        atom_mask.append(True)

                coords_list.append(coords_row)
                atom_mask_list.append(atom_mask)
                mask_list.append(all(atom_mask))

            if not coords_list:
                continue

            coords = torch.tensor(coords_list, dtype=torch.float32)
            mask = torch.tensor(mask_list, dtype=torch.bool)
            atom_mask = torch.tensor(atom_mask_list, dtype=torch.bool)
            seq = torch.tensor(seq_indices, dtype=torch.long)
            nan_mask = torch.zeros((coords.shape[0],), dtype=torch.bool)

            coords, rotation, translation = _normalise_coords(coords, atom_mask[:, 1])

            record = BackboneRecord(
                path=str(path),
                chain_id=chain_name,
                coords=coords,
                mask=mask,
                atom_mask=atom_mask,
                seq=seq,
                seq_string="".join(seq_chars),
                residue_names=residue_names,
                residue_ids=residue_ids,
                rotation=rotation,
                translation=translation,
                nan_mask=nan_mask,
            )
            records.append(record)

        if records:
            break

    if chain_id is not None:
        records = [record for record in records if record.chain_id == chain_id]
    return records


def _load_mmcif_with_gemmi(
    path: Path, *, length_cap: int, chain_id: Optional[str]
) -> List[BackboneRecord]:
    if gemmi is None:
        return []

    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    structure = gemmi.make_structure_from_block(block)
    structure.setup_entities()
    return _build_records_from_gemmi_structure(
        structure, path, length_cap=length_cap, chain_id=chain_id
    )


def _load_pdb_with_gemmi(
    path: Path, *, length_cap: int, chain_id: Optional[str]
) -> List[BackboneRecord]:
    if gemmi is None:
        return []

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    return _build_records_from_gemmi_structure(
        structure, path, length_cap=length_cap, chain_id=chain_id
    )


def _build_records_from_biopython_structure(
    structure, path: Path, *, length_cap: int, chain_id: Optional[str]
) -> List[BackboneRecord]:
    records: List[BackboneRecord] = []

    for model in structure:
        for chain in model:
            chain_name = chain.id
            if chain_id is not None and chain_name != chain_id:
                continue

            residues = [res for res in chain if res.id[0] == " " and res.resname.upper() in CANONICAL_RESIDUES]
            if not residues:
                continue

            if length_cap and len(residues) > length_cap:
                residues = residues[:length_cap]

            coords_list: List[List[float]] = []
            mask_list: List[bool] = []
            atom_mask_list: List[List[bool]] = []
            seq_indices: List[int] = []
            seq_chars: List[str] = []
            residue_names: List[str] = []
            residue_ids: List[Tuple[int, str]] = []

            for residue in residues:
                three_letter = residue.resname.upper()
                if three_letter not in CANONICAL_RESIDUES:
                    continue
                one_letter = THREE_TO_ONE.get(three_letter, "X")
                seq_chars.append(one_letter)
                seq_indices.append(AA_TO_INDEX.get(one_letter, AA_TO_INDEX["X"]))
                residue_names.append(three_letter)
                residue_ids.append((residue.id[1], residue.id[2].strip()))

                coords_row: List[List[float]] = []
                atom_mask = []
                for atom_name in ("N", "CA", "C"):
                    atom = residue.child_dict.get(atom_name)
                    if atom is None:
                        coords_row.append([0.0, 0.0, 0.0])
                        atom_mask.append(False)
                    else:
                        coords_row.append(atom.coord.tolist())
                        atom_mask.append(True)

                coords_list.append(coords_row)
                atom_mask_list.append(atom_mask)
                mask_list.append(all(atom_mask))

            if not coords_list:
                continue

            coords = torch.tensor(coords_list, dtype=torch.float32)
            mask = torch.tensor(mask_list, dtype=torch.bool)
            atom_mask = torch.tensor(atom_mask_list, dtype=torch.bool)
            seq = torch.tensor(seq_indices, dtype=torch.long)
            nan_mask = torch.zeros((coords.shape[0],), dtype=torch.bool)

            coords, rotation, translation = _normalise_coords(coords, atom_mask[:, 1])

            record = BackboneRecord(
                path=str(path),
                chain_id=chain_name,
                coords=coords,
                mask=mask,
                atom_mask=atom_mask,
                seq=seq,
                seq_string="".join(seq_chars),
                residue_names=residue_names,
                residue_ids=residue_ids,
                rotation=rotation,
                translation=translation,
                nan_mask=nan_mask,
            )
            records.append(record)

        if records:
            break

    if chain_id is not None:
        records = [record for record in records if record.chain_id == chain_id]
    return records


def _load_mmcif_with_biopython(
    path: Path, *, length_cap: int, chain_id: Optional[str]
) -> List[BackboneRecord]:
    if MMCIFParser is None:
        return []

    parser = MMCIFParser.MMCIFParser(QUIET=True) if hasattr(MMCIFParser, "MMCIFParser") else MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", str(path))
    return _build_records_from_biopython_structure(
        structure, path, length_cap=length_cap, chain_id=chain_id
    )


def _load_pdb_with_biopython(
    path: Path, *, length_cap: int, chain_id: Optional[str]
) -> List[BackboneRecord]:
    if PDBParser is None:
        return []

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure("structure", str(path))
    return _build_records_from_biopython_structure(
        structure, path, length_cap=length_cap, chain_id=chain_id
    )


def load_mmcif(
    path: str,
    *,
    chain_id: Optional[str] = None,
    length_cap: int = 2048,
) -> List[BackboneRecord]:
    """Load backbone coordinates for all qualifying chains in ``path``.

    Parameters
    ----------
    path:
        File system path to an mmCIF or PDB file.
    chain_id:
        Optional chain identifier.  When provided the returned list only
        contains the corresponding chain.
    length_cap:
        Maximum number of residues to keep.  Chains longer than ``length_cap``
        are truncated to that length.
    """

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)

    format_hint = _detect_format(path_obj)
    loaders: List[Callable[[], List[BackboneRecord]]] = []
    if format_hint == "pdb":
        loaders.extend(
            [
                lambda: _load_pdb_with_gemmi(path_obj, length_cap=length_cap, chain_id=chain_id),
                lambda: _load_pdb_with_biopython(
                    path_obj, length_cap=length_cap, chain_id=chain_id
                ),
            ]
        )
        loaders.extend(
            [
                lambda: _load_mmcif_with_gemmi(path_obj, length_cap=length_cap, chain_id=chain_id),
                lambda: _load_mmcif_with_biopython(
                    path_obj, length_cap=length_cap, chain_id=chain_id
                ),
            ]
        )
    else:
        loaders.extend(
            [
                lambda: _load_mmcif_with_gemmi(path_obj, length_cap=length_cap, chain_id=chain_id),
                lambda: _load_mmcif_with_biopython(
                    path_obj, length_cap=length_cap, chain_id=chain_id
                ),
                lambda: _load_pdb_with_gemmi(path_obj, length_cap=length_cap, chain_id=chain_id),
                lambda: _load_pdb_with_biopython(path_obj, length_cap=length_cap, chain_id=chain_id),
            ]
        )

    records: List[BackboneRecord] = []
    any_loader_completed = False
    errors: List[Exception] = []
    for loader in loaders:
        try:
            candidate = loader()
            any_loader_completed = True
        except Exception as exc:  # pragma: no cover - parser errors fall back to other loaders
            errors.append(exc)
            continue
        if candidate:
            records = candidate
            break

    if not records and not any_loader_completed and errors:
        raise errors[-1]

    if chain_id is not None and not records:
        raise KeyError(f"Chain {chain_id!r} not found in {path}")

    return records


def write_mmcif(record: BackboneRecord, path: str) -> None:
    """Serialise a :class:`BackboneRecord` back to an mmCIF or PDB file."""

    if gemmi is None:
        raise RuntimeError("Writing mmCIF files requires the gemmi package")

    path_obj = Path(path)
    _, compressed = _split_suffixes(path_obj)
    format_hint = _detect_format(path_obj)

    coords = record.coords
    rotation = record.rotation
    translation = record.translation

    transformed = coords @ rotation.T + translation.view(1, 1, 3)

    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    model = gemmi.Model("0")
    chain = gemmi.Chain(record.chain_id or "A")

    for idx in range(record.length):
        if not record.mask[idx] or (record.nan_mask[idx] if record.nan_mask.numel() else False):
            continue
        residue = gemmi.Residue()
        residue.name = record.residue_names[idx] if idx < len(record.residue_names) else "UNK"
        seq_num, seq_icode = (
            record.residue_ids[idx] if idx < len(record.residue_ids) else (idx + 1, "")
        )
        residue.seqid = gemmi.SeqId(int(seq_num), str(seq_icode))
        residue.het_flag = " "

        for atom_name, atom_idx in zip(("N", "CA", "C"), range(3)):
            if idx < record.atom_mask.shape[0] and not record.atom_mask[idx, atom_idx]:
                continue
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.pos = gemmi.Position(*transformed[idx, atom_idx].tolist())
            atom.occ = 1.0
            atom.b_iso = 0.0
            residue.add_atom(atom)

        if residue:
            chain.add_residue(residue)

    if not chain:
        raise ValueError("Cannot write empty chain")

    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()

    if format_hint == "pdb":
        pdb_string = structure.make_pdb_string()
        if compressed:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(pdb_string)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(pdb_string)
    else:
        doc = structure.make_mmcif_document()
        doc.write_file(str(path))


__all__ = [
    "AA_ALPHABET",
    "AA_TO_INDEX",
    "PAD_INDEX",
    "BackboneRecord",
    "load_mmcif",
    "write_mmcif",
]
