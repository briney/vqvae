from pathlib import Path

import gemmi

from gcpvqvae.data.pdb_hdf5 import (
    THREE_TO_ONE,
    ChainMetadata,
    ChainExtractionStats,
    extract_chains,
)


def _add_atom(residue: gemmi.Residue, name: str, position) -> None:
    atom = gemmi.Atom()
    atom.name = name
    atom.pos = gemmi.Position(*position)
    atom.occ = 1.0
    atom.b_iso = 10.0
    residue.add_atom(atom)


def _write_structure(path: Path, *, models: int = 1, duplicate: bool = False) -> None:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)

    for model_index in range(models):
        model = gemmi.Model(str(model_index))

        chain_a = gemmi.Chain("A")
        for idx, name in enumerate(["ALA", "GLY", "SER"], start=1):
            residue = gemmi.Residue()
            residue.name = name
            residue.het_flag = " "
            residue.seqid = gemmi.SeqId(str(idx))
            offset = float(idx - 1)
            _add_atom(residue, "N", (offset, 0.0, 0.0))
            _add_atom(residue, "CA", (offset + 0.5, 0.5, 0.0))
            _add_atom(residue, "C", (offset + 1.0, 0.2, 0.3))
            chain_a.add_residue(residue)

        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        for idx, name in enumerate(["VAL", "TYR", "LEU"], start=1):
            residue = gemmi.Residue()
            residue.name = name
            residue.het_flag = " "
            residue.seqid = gemmi.SeqId(str(idx))
            offset = float(idx - 1)
            _add_atom(residue, "N", (offset, 2.0, 0.0))
            _add_atom(residue, "CA", (offset + 0.6, 2.5, 0.1))
            _add_atom(residue, "C", (offset + 1.1, 2.1, 0.4))
            chain_b.add_residue(residue)

        if duplicate:
            chain_c = gemmi.Chain("C")
            for idx, name in enumerate(["ALA", "GLY", "SER"], start=1):
                residue = gemmi.Residue()
                residue.name = name
                residue.het_flag = " "
                residue.seqid = gemmi.SeqId(str(idx))
                offset = float(idx - 1)
                _add_atom(residue, "N", (offset, 3.0, 0.0))
                if idx != 1:
                    _add_atom(residue, "CA", (offset + 0.5, 3.5, 0.0))
                _add_atom(residue, "C", (offset + 1.0, 3.2, 0.3))
                chain_c.add_residue(residue)
            model.add_chain(chain_c)
        else:
            # Add a short chain to test filtering.
            chain_short = gemmi.Chain("D")
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.het_flag = " "
            residue.seqid = gemmi.SeqId("1")
            _add_atom(residue, "N", (0.0, 4.0, 0.0))
            _add_atom(residue, "CA", (0.5, 4.5, 0.0))
            _add_atom(residue, "C", (1.0, 4.2, 0.3))
            chain_short.add_residue(residue)
            model.add_chain(chain_short)

        model.add_chain(chain_b)
        structure.add_model(model)

    structure.setup_entities()
    if path.suffix.lower() in {".pdb", ".ent"}:
        structure.write_minimal_pdb(str(path))
    else:
        doc = structure.make_mmcif_document()
        doc.write_file(str(path))


def test_three_to_one_contains_expected_mappings():
    assert THREE_TO_ONE["ALA"] == "A"
    assert THREE_TO_ONE["GLX"] == "Z"
    assert THREE_TO_ONE.get("UNK", "X") == "X"


def test_extract_chains_filters_and_tracks_stats(tmp_path):
    path = tmp_path / "structure.cif"
    _write_structure(path)

    chains, stats = extract_chains(path, min_len=2)

    assert isinstance(stats, ChainExtractionStats)
    assert stats.total_chains == len(chains)
    assert stats.dropped_short == 0
    assert stats.complexes == 1
    assert stats.missing_chain_a == 0

    assert len(chains) == 2
    identifiers = {chain.chain_id[0] for chain in chains}
    assert identifiers == {"A", "B"}
    for chain in chains:
        assert isinstance(chain, ChainMetadata)
        if chain.chain_id.startswith("A"):
            assert chain.sequence == "AGS"
        else:
            assert chain.sequence == "VYL"
        assert chain.ca_count == 3


def test_extract_chains_prefers_chain_with_more_ca_atoms(tmp_path):
    path = tmp_path / "structure_duplicate.cif"
    _write_structure(path, duplicate=True)

    chains, stats = extract_chains(path, min_len=1)

    assert stats.deduplicated == 1
    assert len(chains) == 2
    identifiers = {chain.chain_id[0] for chain in chains}
    assert identifiers == {"A", "B"}
    chain_a = next(chain for chain in chains if chain.chain_id.startswith("A"))
    chain_b = next(chain for chain in chains if chain.chain_id.startswith("B"))
    assert chain_a.ca_count == 3
    assert chain_b.ca_count == 3


def test_only_first_model_is_used(tmp_path):
    path = tmp_path / "multi_model.cif"
    _write_structure(path, models=2)

    chains, _ = extract_chains(path, min_len=1)

    ids = {chain.chain_id[0] for chain in chains}
    assert ids == {"A", "B"}
