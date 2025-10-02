from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
import torch

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.data.mmcif import BackboneRecord, write_mmcif
from gcpvqvae.models.gcpnet import GCPNetConfig
from gcpvqvae.models.gcpvqvae import (
    DataPipelineConfig,
    GCPVQVAE,
    GCPVQVAEConfig,
    LatentAdapterConfig,
    RotationHeadConfig,
    VectorQuantizerConfig,
)
from gcpvqvae.models.transformer import TransformerConfig
from gcpvqvae.utils.checkpoint import save_checkpoint


def _add_atom(residue: gemmi.Residue, name: str, position, *, occ: float = 1.0, altloc: str = "") -> None:
    atom = gemmi.Atom()
    atom.name = name
    atom.pos = gemmi.Position(*position)
    atom.occ = occ
    if altloc:
        atom.altloc = altloc
    atom.b_iso = 10.0
    residue.add_atom(atom)


def _build_test_structure(path: Path) -> None:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    chain_a = gemmi.Chain("A")
    for idx, name in enumerate(["ALA", "GLY", "SER"], start=1):
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

    chain_b = gemmi.Chain("B")
    for idx, name in enumerate(["VAL", "TYR"], start=1):
        residue = gemmi.Residue()
        residue.name = name
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))
        base = 4.0 * (idx - 1)
        _add_atom(residue, "N", (0.2, base, 1.0))
        _add_atom(residue, "CA", (0.4, base + 1.1, 1.2))
        if idx == 1:
            _add_atom(residue, "C", (0.6, base + 2.0, 1.1))
        chain_b.add_residue(residue)

    model.add_chain(chain_a)
    model.add_chain(chain_b)
    structure.add_model(model)
    structure.setup_entities()

    if path.suffix.lower() in {".pdb", ".ent"}:
        structure.write_minimal_pdb(str(path))
    else:
        doc = structure.make_mmcif_document()
        doc.write_file(str(path))


def _make_config() -> GCPVQVAEConfig:
    gcp_cfg = GCPNetConfig(
        hidden_scalar_dim=64,
        hidden_vector_dim=8,
        latent_dim=32,
        layers=2,
    )
    vq_cfg = VectorQuantizerConfig(num_codes=32, dim=24, beta=0.25, decay=0.9, kmeans_iters=1)
    enc_cfg = TransformerConfig(
        input_dim=gcp_cfg.latent_dim,
        model_dim=48,
        output_dim=vq_cfg.dim,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    )
    dec_cfg = TransformerConfig(
        input_dim=vq_cfg.dim,
        model_dim=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=1,
        dropout=0.0,
    )
    rot_cfg = RotationHeadConfig(input_dim=48, translation_scale=1.0)
    data_cfg = DataPipelineConfig(length_cap=512, knn=4)
    return GCPVQVAEConfig(
        gcp=gcp_cfg,
        encoder=enc_cfg,
        decoder=dec_cfg,
        vq=vq_cfg,
        rotation=rot_cfg,
        data=data_cfg,
    )


@pytest.mark.parametrize("suffix", [".cif", ".pdb"])
def test_model_forward_runs(tmp_path, suffix) -> None:
    structure_path = Path(tmp_path) / f"toy{suffix}"
    _build_test_structure(structure_path)

    dataset = BackboneDataset(structure_path, k=2, progress=False)
    batch = collate_backbones([dataset[0]])

    model = GCPVQVAE(_make_config())
    output = model(batch)

    assert "total_loss" in output
    loss = output["total_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "expected gradients to flow"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("suffix", [".cif", ".pdb"])
def test_encode_decode_roundtrip(tmp_path, suffix) -> None:
    structure_path = Path(tmp_path) / f"toy{suffix}"
    _build_test_structure(structure_path)

    model = GCPVQVAE(_make_config())

    encoded = model.encode(str(structure_path), chain_id="A", k=2)
    tokens = encoded["tokens"]
    mask = encoded["mask"]

    assert tokens.shape[0] == encoded["length"]
    assert mask.shape[0] == encoded["length"]

    decoded = model.decode(
        tokens,
        pose_header=encoded["pose_header"],
        mask=mask,
        metadata=encoded["metadata"],
    )

    coords = decoded["coords"]
    assert coords.shape[0] == encoded["length"]
    assert decoded["mask"].shape == mask.shape

    record = decoded["records"]
    assert isinstance(record, BackboneRecord)
    assert record.coords.shape[0] == int(mask.sum().item())


def test_latent_adapter_projects_embeddings(tmp_path) -> None:
    structure_path = Path(tmp_path) / "adapter.cif"
    _build_test_structure(structure_path)

    dataset = BackboneDataset(structure_path, k=2, progress=False)
    batch = collate_backbones([dataset[0]])

    gcp_cfg = GCPNetConfig(
        hidden_scalar_dim=128,
        hidden_vector_dim=16,
        edge_scalar_dim=32,
        edge_scalar_input_dim=8,
        edge_vector_dim=1,
        latent_dim=256,
        layers=3,
    )
    adapter_cfg = LatentAdapterConfig(enabled=True, output_dim=32, bias=True)
    vq_cfg = VectorQuantizerConfig(num_codes=16, dim=24, beta=0.25, decay=0.9, kmeans_iters=1)
    enc_cfg = TransformerConfig(
        input_dim=gcp_cfg.latent_dim,
        model_dim=48,
        output_dim=vq_cfg.dim,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    )
    dec_cfg = TransformerConfig(
        input_dim=vq_cfg.dim,
        model_dim=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=1,
        dropout=0.0,
    )
    rot_cfg = RotationHeadConfig(input_dim=48, translation_scale=1.0)
    data_cfg = DataPipelineConfig(length_cap=512, knn=2)
    config = GCPVQVAEConfig(
        gcp=gcp_cfg,
        encoder=enc_cfg,
        decoder=dec_cfg,
        vq=vq_cfg,
        rotation=rot_cfg,
        data=data_cfg,
        adapter=adapter_cfg,
    )

    model = GCPVQVAE(config)
    assert model.latent_adapter is not None
    assert model.latent_adapter.weight.shape[1] == gcp_cfg.latent_dim
    assert model.latent_adapter.weight.shape[0] == adapter_cfg.output_dim
    assert model.encoder_transformer.config.input_dim == adapter_cfg.output_dim

    output = model(batch)
    assert output["gcp_embeddings"].shape[-1] == gcp_cfg.latent_dim
    assert output["encoder_hidden"].shape[-1] == vq_cfg.dim


def test_gcpnet_pretrained_initialisation(tmp_path) -> None:
    config = _make_config()
    model = GCPVQVAE(config)
    reference_state = {}
    with torch.no_grad():
        for idx, (name, param) in enumerate(model.encoder_gcp.state_dict().items()):
            filled = torch.full_like(param, float(idx + 1) / 100.0)
            param.copy_(filled)
            reference_state[name] = filled.clone()

    checkpoint_path = Path(tmp_path) / "gcpnet.pt"
    save_checkpoint({"gcp_state": model.encoder_gcp.state_dict()}, checkpoint_path)

    new_config = _make_config()
    new_config.gcp.init = "pretrained"
    new_config.gcp.init_checkpoint = str(checkpoint_path)
    new_config.gcp.strict_init = True

    restored = GCPVQVAE(new_config)
    loaded_state = restored.encoder_gcp.state_dict()
    for name, expected in reference_state.items():
        assert torch.allclose(loaded_state[name], expected)

