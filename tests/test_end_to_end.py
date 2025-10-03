"""End-to-end integration smoke tests."""

from __future__ import annotations

from pathlib import Path

import gemmi
import torch

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.geometry.frames import kabsch_align
from gcpvqvae.geometry.metrics import rmsd
from gcpvqvae.models.decoder import RotationDecoder
from gcpvqvae.models.gcpvqvae import (
    DataPipelineConfig,
    GCPNetConfig,
    GCPVQVAE,
    GCPVQVAEConfig,
    RotationHeadConfig,
    TransformerConfig,
    VectorQuantizerConfig,
)
from gcpvqvae.models.gcpnet import (
    GCPEmbeddingConfig,
    GCPFeedForwardConfig,
    GCPMessagePassingConfig,
)
from gcpvqvae.models.vq import VectorQuantizer


def _build_roundtrip_structure(path: Path) -> None:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    chain = gemmi.Chain("A")
    for idx in range(1, 5):
        residue = gemmi.Residue()
        residue.name = "GLY"
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))
        base = 3.6 * (idx - 1)
        alt = (-1) ** idx
        for name, pos in zip(
            ["N", "CA", "C"],
            [
                (base, 0.1 * alt, 0.0),
                (base + 1.2, 0.2 * alt, 0.2 * alt),
                (base + 2.3, 0.15 * alt, 0.3),
            ],
        ):
            atom = gemmi.Atom()
            atom.name = name
            atom.pos = gemmi.Position(*pos)
            atom.occ = 1.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    doc = structure.make_mmcif_document()
    doc.write_file(str(path))


def _make_small_config() -> GCPVQVAEConfig:
    gcp_cfg = GCPNetConfig(
        node_scalar_dim=6,
        node_vector_dim=3,
        edge_scalar_dim=8,
        edge_scalar_input_dim=8,
        edge_vector_dim=1,
        embedding=GCPEmbeddingConfig(scalar_dim=32, vector_dim=4),
        message_passing=GCPMessagePassingConfig(scalar_dim=32, vector_dim=4),
        feed_forward=GCPFeedForwardConfig(bottleneck_factor=2.0),
        latent_dim=32,
        num_layers=2,
    )
    vq_cfg = VectorQuantizerConfig(num_codes=16, dim=32, beta=0.25, decay=0.9, kmeans_iters=1)
    enc_cfg = TransformerConfig(
        input_dim=gcp_cfg.latent_dim,
        model_dim=64,
        output_dim=vq_cfg.dim,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    )
    dec_cfg = TransformerConfig(
        input_dim=vq_cfg.dim,
        model_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=1,
        dropout=0.0,
    )
    rot_cfg = RotationHeadConfig(input_dim=64, translation_scale=1.0)
    data_cfg = DataPipelineConfig(length_cap=256, knn=4)
    return GCPVQVAEConfig(gcp=gcp_cfg, encoder=enc_cfg, decoder=dec_cfg, vq=vq_cfg, rotation=rot_cfg, data=data_cfg)


def test_vq_decoder_pipeline_runs() -> None:
    batch, length, dim = 2, 4, 3
    vq = VectorQuantizer(num_codes=4, dim=dim, beta=0.1, decay=0.9, rotation_trick=True)
    decoder = RotationDecoder(dim, translation_scale=0.5)

    latents = torch.randn(batch, length, dim, requires_grad=True)
    quantized, indices, losses = vq(latents)
    coords, _ = decoder(quantized)

    assert coords.shape == (batch, length, 3, 3)
    assert indices.shape == (batch, length)
    total_loss = losses["commitment"] + losses["codebook"]
    total_loss.backward()
    assert torch.isfinite(latents.grad).all()


def test_roundtrip_rmsd_after_brief_training(tmp_path) -> None:
    structure_path = Path(tmp_path) / "toy.cif"
    _build_roundtrip_structure(structure_path)

    dataset = BackboneDataset(structure_path, k=2, progress=False)
    batch = collate_backbones([dataset[0]])

    model = GCPVQVAE(_make_small_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(300):
        optimizer.zero_grad()
        output = model(batch)
        output["total_loss"].backward()
        optimizer.step()

    encoded = model.encode(str(structure_path), chain_id="A", k=2)
    decoded = model.decode(
        encoded["tokens"],
        pose_header=encoded["pose_header"],
        mask=encoded["mask"],
        metadata=encoded["metadata"],
    )

    coords = decoded["coords"]
    mask = decoded["mask"]
    reference = dataset[0]["coords"]

    mask_flat = mask.repeat_interleave(3)
    _, _, aligned = kabsch_align(
        reference.view(-1, 3),
        coords.view(-1, 3),
        mask=mask_flat,
        allow_reflections=False,
        return_aligned=True,
    )
    aligned_coords = aligned.view_as(coords)

    value = rmsd(coords.unsqueeze(0), aligned_coords.unsqueeze(0), mask=mask.unsqueeze(0)).item()
    assert value < 2.0
