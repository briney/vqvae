from pathlib import Path

import gemmi
import json
import torch

from gcpvqvae.data.dataset import (
    BackboneDataset,
    PREPROCESSED_MANIFEST,
)
from gcpvqvae.data.preprocess import preprocess_backbone_dataset as preprocess_dataset


def _make_dataset(path: Path) -> None:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    for chain_id, residue_names in {"A": ["ALA", "GLY", "SER"], "B": ["VAL", "TYR"]}.items():
        chain = gemmi.Chain(chain_id)
        for idx, name in enumerate(residue_names, start=1):
            residue = gemmi.Residue()
            residue.name = name
            residue.het_flag = " "
            residue.seqid = gemmi.SeqId(str(idx))
            base = 3.6 * (idx - 1)

            atom_n = gemmi.Atom()
            atom_n.name = "N"
            atom_n.pos = gemmi.Position(base, idx * 0.1, 0.0)
            atom_n.occ = 1.0
            residue.add_atom(atom_n)

            atom_ca = gemmi.Atom()
            atom_ca.name = "CA"
            atom_ca.pos = gemmi.Position(base + 1.2, idx * 0.1, 0.3)
            atom_ca.occ = 1.0
            residue.add_atom(atom_ca)

            atom_c = gemmi.Atom()
            atom_c.name = "C"
            atom_c.pos = gemmi.Position(base + 2.3, idx * 0.1, 0.5)
            atom_c.occ = 1.0
            residue.add_atom(atom_c)

            chain.add_residue(residue)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    doc = structure.make_mmcif_document()
    doc.write_file(str(path))


def test_preprocess_dataset_roundtrip(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "toy.cif"
    _make_dataset(raw_path)

    output_dir = tmp_path / "processed"
    manifest_path = preprocess_dataset(raw_dir, output_dir, k=4, progress=False)

    assert manifest_path == output_dir / PREPROCESSED_MANIFEST
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["num_samples"] == 2
    assert manifest["k"] == 4

    raw_dataset = BackboneDataset(raw_dir, k=4, progress=False)
    processed_dataset = BackboneDataset(output_dir, k=4, progress=False)

    assert len(raw_dataset) == len(processed_dataset) == 2

    raw_map = {}
    for sample in raw_dataset:
        metadata = sample["metadata"]
        raw_map[(metadata["path"], metadata["chain_id"])] = sample

    processed_map = {}
    for sample in processed_dataset:
        metadata = sample["metadata"]
        processed_map[(metadata["path"], metadata["chain_id"])] = sample

    assert raw_map.keys() == processed_map.keys()

    for key in raw_map:
        raw_sample = raw_map[key]
        processed_sample = processed_map[key]
        assert torch.allclose(raw_sample["coords"], processed_sample["coords"])
        assert torch.equal(raw_sample["mask"], processed_sample["mask"])
        assert raw_sample["metadata"]["sequence"] == processed_sample["metadata"]["sequence"]

    limited_dataset = BackboneDataset(output_dir, k=4, length_cap=2, progress=False)
    limited_sequences = {}
    for sample in limited_dataset:
        metadata = sample["metadata"]
        limited_sequences[metadata["chain_id"]] = metadata["sequence"]
        assert sample["coords"].shape[0] <= 2
        assert len(sample["seq_str"]) <= 2
    assert set(limited_sequences) == {"A", "B"}
    assert limited_sequences["A"] == "AG"
    assert limited_sequences["B"] == "VY"

    filtered_dataset = BackboneDataset(output_dir, k=4, chain_ids=["B"], progress=False)
    assert len(filtered_dataset) == 1
    filtered_sample = filtered_dataset[0]
    assert filtered_sample["metadata"]["chain_id"] == "B"
    assert filtered_sample["metadata"]["sequence"] == "VY"
