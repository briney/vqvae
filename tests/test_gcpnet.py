from pathlib import Path

import pytest
import torch

from gcpvqvae.data.batch import EdgeStorage, ProteinBatch
from gcpvqvae.models.gcpnet import (
    DEFAULT_EDGE_SCALAR_INPUT_DIM,
    GCPConv,
    GCPFeedForwardConfig,
    GCPMessagePassingConfig,
    GCPNetConfig,
    GCPNetEncoder,
    GCPWidthConfig,
    ScalarVector,
)
from gcpvqvae.models.gcpvqvae import GCPVQVAE
from gcpvqvae.system.configuration import build_model_config, compose_overrides
from gcpvqvae.system.train_gcpnet import _prepare_model_config
from gcpvqvae.utils.checkpoint import load_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]
# _CHECKPOINT_PATH = (
#     _REPO_ROOT
#     / "models"
#     / "checkpoints"
#     / "gcpnet"
#     / "structure_denoising"
#     / "ca_bb"
#     / "last.ckpt"
# )
_CHECKPOINT_PATH = GCPVQVAE._default_gcp_checkpoint_path()
_CONFIG_DIR = _REPO_ROOT / "src" / "gcpvqvae" / "configs"


def test_gcpnet_encoder_projects_edge_scalars_with_default_input_dim() -> None:
    config = GCPNetConfig()
    encoder = GCPNetEncoder(config)

    weight = encoder.embedding.edge_scalar_proj.weight
    assert weight.shape[1] == DEFAULT_EDGE_SCALAR_INPUT_DIM + encoder.embedding.num_rbf

    num_nodes = 4
    node_scalars = torch.randn(num_nodes, config.node_scalar_dim)
    node_vectors = torch.randn(num_nodes, config.node_vector_dim, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_scalars = torch.randn(edge_index.shape[1], DEFAULT_EDGE_SCALAR_INPUT_DIM)
    edge_vectors = torch.randn(edge_index.shape[1], config.edge_vector_input_dim, 3)
    edge_frames = torch.eye(3).expand(edge_index.shape[1], 3, 3).clone()
    positions = torch.randn(num_nodes, 3)

    proto = ProteinBatch(
        h=node_scalars,
        chi=node_vectors,
        e={
            "knn_k": EdgeStorage(
                edge_index=edge_index,
                scalars=edge_scalars,
                vectors=edge_vectors,
                frames=edge_frames,
                batch=torch.zeros(edge_index.shape[1], dtype=torch.long),
            )
        },
        xi=positions,
        batch=torch.zeros(num_nodes, dtype=torch.long),
        ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        mask=torch.ones(num_nodes, dtype=torch.bool),
    )

    output = encoder(proto)

    node_dim = output["node_embedding"].shape[-1]
    assert output["node_embedding"].shape == (num_nodes, node_dim)
    assert output["graph_embedding"].shape == (proto.num_graphs(), node_dim)


def test_protein_batch_to_handles_missing_optional_attrs() -> None:
    num_nodes = 3
    node_scalars = torch.randn(num_nodes, 4)
    node_vectors = torch.randn(num_nodes, 2, 3)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_scalars = torch.randn(edge_index.shape[1], 4)
    edge_vectors = torch.randn(edge_index.shape[1], 2, 3)
    edge_frames = torch.eye(3).expand(edge_index.shape[1], 3, 3).clone()

    batch = ProteinBatch(
        h=node_scalars,
        chi=node_vectors,
        e={
            "knn_k": EdgeStorage(
                edge_index=edge_index,
                scalars=edge_scalars,
                vectors=edge_vectors,
                frames=edge_frames,
                batch=torch.zeros(edge_index.shape[1], dtype=torch.long),
            )
        },
        xi=torch.randn(num_nodes, 3),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        mask=torch.ones(num_nodes, dtype=torch.bool),
    )

    moved = batch.to(device=torch.device("cpu"))
    assert isinstance(moved, ProteinBatch)


def test_prepare_model_config_defaults_edge_scalar_input_dim() -> None:
    config = _prepare_model_config({"gcp": {"edge_scalar_dim": 32}})

    assert config.gcp.edge_scalar_input_dim == DEFAULT_EDGE_SCALAR_INPUT_DIM


def test_gcpconv_supports_bfloat16_inputs() -> None:
    config = GCPNetConfig()
    conv = GCPConv(
        config.hidden_scalar_dim,
        config.hidden_vector_dim,
        edge_scalar_dim=config.edge_scalar_dim,
        edge_vector_channels=config.edge_vector_dim,
        hidden_scalar_dim=config.feed_forward_scalar_dim,
        hidden_vector_channels=config.feed_forward_vector_dim,
        dropout=config.dropout,
    )

    node_scalars = torch.randn(6, conv.scalar_dim, dtype=torch.bfloat16)
    node_vectors = torch.randn(6, conv.vector_dim, 3, dtype=torch.bfloat16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_scalars = torch.randn(
        edge_index.shape[1], conv.edge_scalar_dim, dtype=torch.bfloat16
    )
    edge_vectors = torch.randn(
        edge_index.shape[1], conv.edge_vector_channels, 3, dtype=torch.bfloat16
    )
    edge_frames = (
        torch.eye(3, dtype=torch.bfloat16).expand(edge_index.shape[1], 3, 3).clone()
    )

    node_features = ScalarVector(node_scalars, node_vectors)
    edge_features = ScalarVector(edge_scalars, edge_vectors)

    output = conv(node_features, edge_features, edge_index, edge_frames)

    assert output.scalars.dtype is torch.bfloat16
    assert output.vectors.dtype is torch.bfloat16


def test_gcpnet_encoder_supports_bfloat16_inputs() -> None:
    config = GCPNetConfig(
        message_passing=GCPMessagePassingConfig(
            width=GCPWidthConfig(scalar=16, vector=8)
        ),
        feed_forward=GCPFeedForwardConfig(width=GCPWidthConfig(scalar=32, vector=8)),
        latent_dim=32,
        num_layers=2,
    )
    encoder = GCPNetEncoder(config)

    num_nodes = 5
    num_edges = 4

    node_scalars = torch.randn(num_nodes, config.node_scalar_dim, dtype=torch.bfloat16)
    node_vectors = torch.randn(
        num_nodes, config.node_vector_dim, 3, dtype=torch.bfloat16
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_scalars = torch.randn(
        num_edges, DEFAULT_EDGE_SCALAR_INPUT_DIM, dtype=torch.bfloat16
    )
    edge_vectors = torch.randn(
        num_edges, config.edge_vector_input_dim, 3, dtype=torch.bfloat16
    )
    edge_frames = torch.eye(3, dtype=torch.bfloat16).expand(num_edges, 3, 3).clone()
    positions = torch.randn(num_nodes, 3, dtype=torch.bfloat16)

    proto = ProteinBatch(
        h=node_scalars,
        chi=node_vectors,
        e={
            "knn_k": EdgeStorage(
                edge_index=edge_index,
                scalars=edge_scalars,
                vectors=edge_vectors,
                frames=edge_frames,
                batch=torch.zeros(num_edges, dtype=torch.long),
            )
        },
        xi=positions,
        batch=torch.zeros(num_nodes, dtype=torch.long),
        ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        mask=torch.ones(num_nodes, dtype=torch.bool),
    )

    outputs = encoder(proto)

    assert outputs["node_embedding"].dtype is torch.bfloat16
    assert outputs["node_scalars"].dtype is torch.bfloat16
    assert outputs["node_vectors"].dtype is torch.bfloat16


def test_gcpnet_reference_checkpoint_loads() -> None:
    state = load_checkpoint(_CHECKPOINT_PATH, map_location="cpu")
    state_dict = GCPVQVAE._extract_gcp_state_dict(state)

    assert state_dict is not None
    assert state_dict, "checkpoint should contain reference parameters"

    encoder = GCPNetEncoder(GCPNetConfig())
    model_state = encoder.state_dict()

    assert isinstance(model_state, dict)


@pytest.mark.parametrize("config_name", ["base", "small", "xsmall"])
def test_packaged_configs_initialize_gcpnet_from_pretrained_weights(
    config_name: str,
) -> None:
    raw = compose_overrides(_CONFIG_DIR / f"{config_name}.yaml", ())
    model_config = build_model_config(raw.get("model"))

    model = GCPVQVAE(model_config)

    raw_state = load_checkpoint(_CHECKPOINT_PATH, map_location="cpu")
    extracted = GCPVQVAE._extract_gcp_state_dict(raw_state)
    assert extracted is not None

    model_state = model.encoder_gcp.state_dict()
    mapped = GCPVQVAE._coerce_gcp_state_dict(extracted, model_state)

    assert mapped, "expected pretrained weights to map onto the encoder"
    assert "embedding.node_scalar_proj.weight" in mapped

    for key, tensor in mapped.items():
        assert key in model_state
        assert torch.equal(model_state[key], tensor)
