"""Helpers for loading and filtering PDB/mmCIF structures for HDF5 export."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from Bio import pairwise2
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


# Shared residue mapping used across workers.  The dictionary covers the 20
# canonical amino acids together with the common ambiguity codes and rare
# residues observed in the PDB archive.  Any residue not listed here defaults
# to ``"X"`` which signals an unknown residue in downstream processing.
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
    "ASX": "B",
    "GLX": "Z",
    "PYL": "O",
    "SEC": "U",
}


@dataclass
class ChainMetadata:
    """Description of a single protein chain extracted from a structure."""

    chain_id: str
    chain: Chain
    residues: List[Residue]
    sequence: str
    ca_count: int


@dataclass
class ChainExtractionStats:
    """Counters describing how a structure was filtered."""

    total_chains: int = 0
    dropped_short: int = 0
    deduplicated: int = 0
    complexes: int = 0
    missing_chain_a: int = 0


def _global_identity(seq_a: str, seq_b: str) -> float:
    """Return the global sequence identity between two chains.

    Args:
        seq_a: First amino-acid sequence.
        seq_b: Second amino-acid sequence.

    Returns:
        Fraction of identical residues in a global alignment in ``[0.0, 1.0]``.
    """

    if not seq_a or not seq_b:
        return 0.0
    score = pairwise2.align.globalxx(seq_a, seq_b, one_alignment_only=True, score_only=True)
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 0.0
    return float(score) / float(max_len)


def _parse_structure(path: Path):
    """Parse ``path`` with the appropriate Biopython parser.

    Args:
        path: Path to an mmCIF or PDB file. ``.gz`` suffix triggers compressed
            reading.

    Returns:
        Biopython structure object with the structure contents parsed.
    """

    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed = suffixes and suffixes[-1] == ".gz"
    if compressed:
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True, auth_chains=False)
    else:
        parser = PDBParser(QUIET=True)
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return parser.get_structure(path.stem, handle)
    return parser.get_structure(path.stem, str(path))


def _first_model(structure):
    """Return the first model from a Biopython structure.

    Args:
        structure: Structure object returned by Biopython parsers.

    Returns:
        First model contained in the structure.

    Raises:
        ValueError: If the structure contains no models.
    """
    models = list(structure.get_models())
    if not models:
        raise ValueError("Structure does not contain any models")
    return models[0]


def _build_chain_metadata(chains: Iterable[Chain], *, min_len: int) -> Tuple[List[ChainMetadata], ChainExtractionStats]:
    """Analyse chains and collect metadata for downstream filtering.

    Args:
        chains: Iterable over Biopython chain objects.
        min_len: Minimum allowed sequence length; shorter chains are dropped.

    Returns:
        Tuple ``(metadata, stats)`` where ``metadata`` is a list of
        :class:`ChainMetadata` records sorted by sequence length and ``stats`` is
        a :class:`ChainExtractionStats` summary.
    """
    builder = PPBuilder()
    stats = ChainExtractionStats()
    initial_chain_ids: List[str] = []
    metadata: List[ChainMetadata] = []

    for chain in chains:
        chain_id = chain.id
        initial_chain_ids.append(chain_id)
        peptides = builder.build_peptides(chain, aa_only=False)
        residues: List[Residue] = []
        for peptide in peptides:
            residues.extend(list(peptide))
        if not residues:
            residues = [res for res in chain.get_residues() if res.id[0] == " "]
        if not residues:
            continue

        sequence_chars: List[str] = []
        for residue in residues:
            name = residue.get_resname().upper()
            sequence_chars.append(THREE_TO_ONE.get(name, "X"))

        sequence = "".join(sequence_chars)
        stats.total_chains += 1
        if len(sequence) < min_len:
            stats.dropped_short += 1
            continue

        ca_count = sum(1 for residue in residues if residue.has_id("CA"))
        metadata.append(ChainMetadata(chain_id=chain_id, chain=chain, residues=residues, sequence=sequence, ca_count=ca_count))

    if len(initial_chain_ids) > 1:
        stats.complexes = 1

    # Deduplicate highly similar chains by global identity.
    filtered: List[ChainMetadata] = []
    for candidate in metadata:
        duplicate_idx = None
        for idx, existing in enumerate(filtered):
            identity = _global_identity(candidate.sequence, existing.sequence)
            if identity >= 0.90:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            filtered.append(candidate)
            continue

        stats.deduplicated += 1
        existing = filtered[duplicate_idx]
        if candidate.ca_count > existing.ca_count:
            filtered[duplicate_idx] = candidate

    kept_ids = {item.chain_id for item in filtered}
    if "A" in initial_chain_ids and "A" not in kept_ids:
        stats.missing_chain_a = 1

    return filtered, stats


def extract_chains(path: str | Path, *, min_len: int = 0) -> Tuple[List[ChainMetadata], ChainExtractionStats]:
    """Load ``path`` and return filtered chain metadata with processing stats.

    Args:
        path: Path to an mmCIF or PDB file (plain text or gzip-compressed).
        min_len: Minimum sequence length to retain.

    Returns:
        Tuple ``(metadata, stats)`` where ``metadata`` is a list of
        :class:`ChainMetadata` instances and ``stats`` details filtering events.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the structure lacks models.
    """

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)

    structure = _parse_structure(path_obj)
    model = _first_model(structure)
    chains = list(model.get_chains())
    return _build_chain_metadata(chains, min_len=min_len)


__all__ = ["THREE_TO_ONE", "ChainMetadata", "ChainExtractionStats", "extract_chains"]
