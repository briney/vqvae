"""Tests for the data ingestion and featurisation pipeline."""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
import torch

from gcpvqvae.data.batch import protein_batch_from_graph_dict
from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.data.featurize import featurize_backbone
from gcpvqvae.data.mmcif import PAD_INDEX, load_mmcif
from gcpvqvae.data.preprocess import preprocess_backbone_dataset as preprocess_dataset
from gcpvqvae.models.gcpnet import (
    GCPEmbeddingConfig,
    GCPFeedForwardConfig,
    GCPMessagePassingConfig,
    GCPNetConfig,
    GCPNetEncoder,
    GCPWidthConfig,
)


def _add_atom(residue: gemmi.Residue, name: str, position, *, occ: float = 1.0, altloc: str = "") -> None:
    atom = gemmi.Atom()
    atom.name = name
    atom.pos = gemmi.Position(*position)
    atom.occ = occ
    if altloc:
        atom.altloc = altloc
    atom.b_iso = 10.0
    residue.add_atom(atom)


def _build_test_structure(path: Path) -> dict[str, torch.Tensor]:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    info: dict[str, torch.Tensor] = {}

    model = gemmi.Model("0")
    chain_a = gemmi.Chain("A")
    ca_positions_a = []
    residue_names_a = ["ALA", "GLY", "SER"]
    for idx, name in enumerate(residue_names_a, start=1):
        residue = gemmi.Residue()
        residue.name = name
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))
        base = 3.8 * (idx - 1)
        _add_atom(residue, "N", (base, 0.5, 0.0))
        _add_atom(residue, "CA", (base + 1.2, 1.0, (-1) ** idx * 0.3), occ=0.2, altloc="A")
        _add_atom(residue, "CA", (base + 1.0, 1.2, (-1) ** idx * 0.4), occ=0.8, altloc="B")
        _add_atom(residue, "C", (base + 2.4, 0.7, 0.5))
        chain_a.add_residue(residue)
        ca_positions_a.append([base + 1.0, 1.2, (-1) ** idx * 0.4])

    model.add_chain(chain_a)
    info["A"] = torch.tensor(ca_positions_a, dtype=torch.float32)

    chain_b = gemmi.Chain("B")
    ca_positions_b = []
    residue_names_b = ["VAL", "TYR"]
    for idx, name in enumerate(residue_names_b, start=1):
        residue = gemmi.Residue()
        residue.name = name
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))
        base = 4.0 * (idx - 1)
        _add_atom(residue, "N", (0.2, base, 1.0))
        _add_atom(residue, "CA", (0.4, base + 1.1, 1.2))
        if idx == 1:
            _add_atom(residue, "C", (0.6, base + 2.0, 1.1))
        # omit C for the second residue to test masking behaviour
        chain_b.add_residue(residue)
        ca_positions_b.append([0.4, base + 1.1, 1.2])

    model.add_chain(chain_b)
    info["B"] = torch.tensor(ca_positions_b, dtype=torch.float32)

    structure.add_model(model)
    structure.setup_entities()
    if path.suffix.lower() in {".pdb", ".ent"}:
        structure.write_minimal_pdb(str(path))
    else:
        doc = structure.make_mmcif_document()
        doc.write_file(str(path))

    return info


@pytest.mark.parametrize("suffix", [".cif", ".pdb"])
def test_load_mmcif_parses_backbone(tmp_path, suffix):
    path = tmp_path / f"test{suffix}"
    info = _build_test_structure(path)

    records = load_mmcif(str(path))
    assert {record.chain_id for record in records} == {"A", "B"}

    record_a = next(record for record in records if record.chain_id == "A")
    assert record_a.coords.shape == (3, 3, 3)
    assert torch.all(record_a.mask)
    assert torch.all(record_a.atom_mask)

    expected_centroid = info["A"].mean(dim=0)
    assert torch.allclose(record_a.translation, expected_centroid, atol=1e-5)
    assert torch.allclose(record_a.coords[:, 1, :].mean(dim=0), torch.zeros(3), atol=1e-5)
    assert record_a.seq_string == "AGS"

    record_b = next(record for record in records if record.chain_id == "B")
    assert record_b.coords.shape == (2, 3, 3)
    assert record_b.mask.tolist() == [True, False]
    assert record_b.atom_mask[1].tolist() == [True, True, False]


@pytest.mark.parametrize("suffix", [".cif", ".pdb"])
def test_load_mmcif_filters_noncanonical_residues(tmp_path, suffix):
    path = tmp_path / f"noncanonical{suffix}"

    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    chain = gemmi.Chain("A")
    residue_names = ["ALA", "PTR", "GLY"]
    for idx, name in enumerate(residue_names, start=1):
        residue = gemmi.Residue()
        residue.name = name
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))
        base = 3.5 * (idx - 1)
        _add_atom(residue, "N", (base, 0.0, 0.0))
        _add_atom(residue, "CA", (base + 1.2, 0.2, 0.0))
        _add_atom(residue, "C", (base + 2.4, 0.0, 0.0))
        chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()

    if path.suffix.lower() in {".pdb", ".ent"}:
        structure.write_minimal_pdb(str(path))
    else:
        doc = structure.make_mmcif_document()
        doc.write_file(str(path))

    records = load_mmcif(str(path))
    assert len(records) == 1
    record = records[0]
    assert record.chain_id == "A"
    assert record.coords.shape[0] == 2
    assert record.seq_string == "AG"
    assert all(name in {"ALA", "GLY"} for name in record.residue_names)


@pytest.mark.parametrize("suffix", [".cif", ".pdb"])
def test_featurize_backbone_produces_expected_shapes(tmp_path, suffix):
    path = tmp_path / f"test{suffix}"
    _build_test_structure(path)

    record = load_mmcif(str(path), chain_id="A")[0]
    features = featurize_backbone(record, k=2)

    assert features["node_scalars"].shape == (3, 6)
    assert features["node_vectors"].shape == (3, 3, 3)
    assert features["backbone_vectors"].shape == (3, 6, 3)
    assert features["torsion_angles"].shape == (3, 3)

    norms = torch.linalg.norm(features["node_vectors"], dim=-1)
    valid = record.mask.unsqueeze(-1).expand_as(norms)
    assert torch.allclose(norms[valid], torch.ones_like(norms[valid]), atol=1e-4)

    edge_index = features["edge_index"]
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    frames = features["edge_frames"]
    if frames.numel():
        identity = torch.eye(3, dtype=frames.dtype)
        orthogonality = frames.transpose(-1, -2) @ frames
        assert torch.allclose(orthogonality, identity.expand_as(orthogonality), atol=1e-4)


def _make_encoder_config() -> GCPNetConfig:
    return GCPNetConfig(
        embedding=GCPEmbeddingConfig(
            node_scalar_dim=6,
            node_vector_dim=3,
            edge_scalar_dim=8,
            edge_vector_dim=1,
            output=GCPWidthConfig(scalar=32, vector=4),
        ),
        message_passing=GCPMessagePassingConfig(width=GCPWidthConfig(scalar=32, vector=4)),
        feed_forward=GCPFeedForwardConfig(width=GCPWidthConfig(scalar=64, vector=4)),
        latent_dim=16,
        num_layers=1,
    )


def test_backbone_dataset_batch_matches_gcpnet_expectations(tmp_path):
    structure_path = tmp_path / "toy.cif"
    _build_test_structure(structure_path)

    dataset = BackboneDataset(structure_path, k=2, progress=False)
    sample = dataset[0]
    batch = collate_backbones([sample])
    protein_batch = protein_batch_from_graph_dict(batch)

    config = _make_encoder_config()
    encoder = GCPNetEncoder(config)

    expected_nodes = int(batch["lengths"].sum().item())
    assert protein_batch.h.shape == (expected_nodes, config.node_scalar_dim)
    assert protein_batch.chi.shape == (expected_nodes, config.node_vector_dim, 3)

    edge_storage = next(iter(protein_batch.e.values()))
    assert edge_storage.scalars.shape[1] == config.edge_scalar_dim
    assert edge_storage.vectors.shape[-1] == 3

    outputs = encoder(protein_batch)

    node_embedding = outputs["node_embedding"]
    graph_embedding = outputs["graph_embedding"]
    assert node_embedding.shape[0] == expected_nodes
    assert graph_embedding.shape == (protein_batch.num_graphs(), config.latent_dim)
    assert torch.all(torch.isfinite(node_embedding))
    assert torch.all(torch.isfinite(graph_embedding))


def test_preprocessed_dataset_batch_matches_gcpnet_expectations(tmp_path):
    structure_path = tmp_path / "toy.cif"
    _build_test_structure(structure_path)

    raw_dataset = BackboneDataset(structure_path, k=2, progress=False)
    raw_sample = raw_dataset[0]

    processed_root = tmp_path / "processed"
    preprocess_dataset(structure_path, processed_root, k=2, progress=False)

    processed_dataset = BackboneDataset(processed_root, k=2, progress=False)
    processed_sample = processed_dataset[0]

    for key in (
        "coords",
        "mask",
        "node_scalars",
        "node_vectors",
        "edge_index",
        "edge_scalars",
        "edge_vectors",
    ):
        assert torch.allclose(processed_sample[key], raw_sample[key])

    batch = collate_backbones([processed_sample])
    protein_batch = protein_batch_from_graph_dict(batch)

    config = _make_encoder_config()
    encoder = GCPNetEncoder(config)

    expected_nodes = int(batch["lengths"].sum().item())
    assert protein_batch.h.shape == (expected_nodes, config.node_scalar_dim)
    assert protein_batch.batch_size == batch["coords"].shape[0]
    assert torch.equal(protein_batch.lengths, batch["lengths"])

    outputs = encoder(protein_batch)

    node_embedding = outputs["node_embedding"]
    graph_embedding = outputs["graph_embedding"]
    assert node_embedding.shape[0] == expected_nodes
    assert graph_embedding.shape == (protein_batch.num_graphs(), config.latent_dim)
    assert torch.all(torch.isfinite(node_embedding))
    assert torch.all(torch.isfinite(graph_embedding))


def test_protein_batch_filters_edges_from_masked_nodes():
    data_root = Path("tests/test_data/cif_50")
    dataset = BackboneDataset(
        data_root,
        k=4,
        length_cap=128,
        cache=True,
        progress=False,
    )

    # Pick a sample with at least one masked-out residue inside the sequence.
    sample = next(item for item in dataset if item["mask"].sum() < item["mask"].shape[0])
    batch = collate_backbones([sample])

    protein_batch = protein_batch_from_graph_dict(batch)
    storage = next(iter(protein_batch.e.values()))

    assert storage.edge_index.numel() > 0
    assert storage.edge_index.max().item() < protein_batch.h.shape[0]
    assert storage.edge_index.min().item() >= 0

    src, dst = storage.edge_index
    num_nodes = protein_batch.h.shape[0]
    assert torch.all(src < num_nodes)
    assert torch.all(dst < num_nodes)

    if storage.scalars.numel():
        assert storage.scalars.shape[0] == src.shape[0]
    if storage.vectors.numel():
        assert storage.vectors.shape[0] == src.shape[0]


def test_dataset_and_collate(tmp_path):
    cif_path = tmp_path / "test.cif"
    pdb_path = tmp_path / "test.pdb"
    _build_test_structure(cif_path)
    _build_test_structure(pdb_path)

    dataset = BackboneDataset(tmp_path, k=2, progress=False)
    assert len(dataset) == 4

    paths = [Path(p) for p, _ in dataset._keys]  # type: ignore[attr-defined]
    assert any(path.suffix.lower() == ".cif" for path in paths)
    assert any(path.suffix.lower() == ".pdb" for path in paths)

    sample = dataset[0]
    other = dataset[1]
    assert sample["coords"].ndim == 3
    assert sample["node_scalars"].shape[-1] == 6
    assert "metadata" in sample

    batch = collate_backbones([sample, other])
    assert batch["coords"].shape[0] == 2
    assert batch["coords"].shape[1] >= sample["coords"].shape[0]
    assert torch.all(batch["seq"][0, batch["lengths"][0] : ] == PAD_INDEX)

    total_nodes = int(batch["lengths"].sum())
    assert batch["node_batch"].shape[0] == total_nodes


def _assert_tensors_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    if tensor_a.dtype.is_floating_point:
        assert torch.allclose(tensor_a, tensor_b, atol=1e-6)
    else:
        assert torch.equal(tensor_a, tensor_b)


def test_dataset_repeated_parsing_matches():
    data_root = Path("tests/test_data/cif_50")

    first_dataset = BackboneDataset(
        data_root,
        k=2,
        cache=True,
        progress=False,
    )
    second_dataset = BackboneDataset(
        data_root,
        k=2,
        cache=True,
        progress=False,
    )

    assert len(first_dataset) == len(second_dataset)
    assert first_dataset._keys == second_dataset._keys  # type: ignore[attr-defined]

    for idx in range(len(first_dataset)):
        first_sample = first_dataset[idx]
        second_sample = second_dataset[idx]

        tensor_fields = [
            "coords",
            "mask",
            "atom_mask",
            "seq",
            "nan_mask",
            "node_scalars",
            "node_vectors",
            "backbone_vectors",
            "torsion_angles",
            "edge_index",
            "edge_scalars",
            "edge_vectors",
            "edge_frames",
        ]
        for field in tensor_fields:
            _assert_tensors_equal(first_sample[field], second_sample[field])  # type: ignore[index]

        pose_fields = ["rotation", "translation"]
        for field in pose_fields:
            _assert_tensors_equal(first_sample["pose"][field], second_sample["pose"][field])  # type: ignore[index]

        assert first_sample["metadata"] == second_sample["metadata"]


def test_dataset_skips_chains_without_valid_backbone():
    data_root = Path(__file__).resolve().parent / "test_data" / "cif_50"
    dataset = BackboneDataset(data_root, cache=True, progress=False)

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert sample["mask"].any(), "dataset yielded a chain with no valid residues"
