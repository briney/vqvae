"""Utilities for reading and writing mmCIF structures."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import gemmi
import numpy as np
import torch

# A mapping from 3-letter residue codes to 1-letter codes.
# Source: https://www.ddbj.nig.ac.jp/ddbj/code-e.html
RESTYPE_MAP_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X",
}
RESTYPE_MAP_1_TO_3 = {v: k for k, v in RESTYPE_MAP_3_TO_1.items()}
STANDARD_AMINO_ACIDS_3 = list(RESTYPE_MAP_3_TO_1.keys())
# Also create a mapping from 1-letter code to integer index
AA_TO_INDEX = {aa: i for i, aa in enumerate(RESTYPE_MAP_3_TO_1.values())}
INDEX_TO_AA = {i: aa for aa, i in AA_TO_INDEX.items()}


@dataclasses.dataclass(frozen=True)
class ParsedMmcif:
    coords: torch.Tensor
    mask: torch.Tensor
    sequence: torch.Tensor
    pose_header: tuple[torch.Tensor, torch.Tensor]
    chain_id: str
    aatype: str


def load_mmcif(path: str, chain_id: str, max_length: int = 2048) -> Optional[ParsedMmcif]:
    try:
        structure = gemmi.read_structure(path)
    except (RuntimeError, ValueError) as e:
        print(f"Failed to read {path}: {e}")
        return None

    chain = None
    for model in structure:
        for ch in model:
            if ch.name == chain_id:
                chain = ch
                break

    if not chain:
        return None

    coords_list, mask_list, aatype_list = [], [], []
    for res_idx, residue in enumerate(chain):
        if residue.name not in STANDARD_AMINO_ACIDS_3:
            continue

        n_atom = residue.find_atom('N', '\0')
        ca_atom = residue.find_atom('CA', '\0')
        c_atom = residue.find_atom('C', '\0')

        is_valid = all(atom is not None for atom in [n_atom, ca_atom, c_atom])
        mask_list.append(is_valid)

        if is_valid:
            coords_list.append([n_atom.pos, ca_atom.pos, c_atom.pos])
        else:
            coords_list.append([(0,0,0), (0,0,0), (0,0,0)])

        aatype_list.append(RESTYPE_MAP_3_TO_1.get(residue.name, "X"))

    if not coords_list or len(coords_list) > max_length:
        return None

    coords = np.array([[(p.x, p.y, p.z) for p in res] for res in coords_list], dtype=np.float32)
    mask = np.array(mask_list, dtype=bool)
    aatype = "".join(aatype_list)
    sequence = np.array([AA_TO_INDEX.get(c, AA_TO_INDEX["X"]) for c in aatype], dtype=np.int64)

    ca_coords = coords[:, 1, :]
    valid_ca_coords = ca_coords[mask]

    if valid_ca_coords.shape[0] == 0:
        return None

    centroid = np.mean(valid_ca_coords, axis=0)
    coords -= centroid

    rotation = np.eye(3, dtype=np.float32)
    translation = centroid

    return ParsedMmcif(
        coords=torch.from_numpy(coords),
        mask=torch.from_numpy(mask),
        sequence=torch.from_numpy(sequence),
        pose_header=(torch.from_numpy(rotation), torch.from_numpy(translation)),
        chain_id=chain_id,
        aatype=aatype,
    )


def write_mmcif(
    coords: np.ndarray,
    mask: np.ndarray,
    aatype: str,
    chain_id: str,
    path: str,
) -> None:
    """Writes backbone coordinates to a minimal mmCIF file."""
    struct = gemmi.Structure()
    model = gemmi.Model('1')
    chain = gemmi.Chain(chain_id)

    for i in range(len(aatype)):
        if not mask[i]:
            continue

        res_name = RESTYPE_MAP_1_TO_3.get(aatype[i], "UNK")
        residue = gemmi.Residue()
        residue.name = res_name
        residue.seqid = gemmi.SeqId(i + 1)

        n_pos, ca_pos, c_pos = coords[i]

        n_atom = gemmi.Atom()
        n_atom.name = 'N'
        n_atom.element = gemmi.Element('N')
        n_atom.pos = gemmi.Position(*n_pos)
        residue.add_atom(n_atom)

        ca_atom = gemmi.Atom()
        ca_atom.name = 'CA'
        ca_atom.element = gemmi.Element('C')
        ca_atom.pos = gemmi.Position(*ca_pos)
        residue.add_atom(ca_atom)

        c_atom = gemmi.Atom()
        c_atom.name = 'C'
        c_atom.element = gemmi.Element('C')
        c_atom.pos = gemmi.Position(*c_pos)
        residue.add_atom(c_atom)

        chain.add_residue(residue)

    model.add_chain(chain)
    struct.add_model(model)
    struct.write_cif(path)
