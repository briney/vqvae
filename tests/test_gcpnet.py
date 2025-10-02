import torch

from gcpvqvae.models.gcpnet import (
    DEFAULT_EDGE_SCALAR_INPUT_DIM,
    GCPNetConfig,
    GCPNetEncoder,
)
from gcpvqvae.system.train_gcpnet import _prepare_model_config


def test_gcpnet_encoder_projects_edge_scalars_with_default_input_dim() -> None:
    config = GCPNetConfig(edge_scalar_dim=32)
    encoder = GCPNetEncoder(config)

    assert encoder.edge_scalar_proj is not None
    weight = encoder.edge_scalar_proj.weight
    assert weight.shape[1] == DEFAULT_EDGE_SCALAR_INPUT_DIM

    num_nodes = 4
    node_scalars = torch.randn(num_nodes, config.node_scalar_dim)
    node_vectors = torch.randn(num_nodes, config.node_vector_dim, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_scalars = torch.randn(edge_index.shape[1], DEFAULT_EDGE_SCALAR_INPUT_DIM)
    edge_vectors = torch.randn(edge_index.shape[1], config.edge_vector_dim, 3)
    edge_frames = torch.randn(edge_index.shape[1], 3, 3)

    output = encoder(
        node_scalars,
        node_vectors,
        edge_index,
        edge_scalars,
        edge_vectors,
        edge_frames,
    )

    assert output["embeddings"].shape == (num_nodes, config.latent_dim)


def test_prepare_model_config_defaults_edge_scalar_input_dim() -> None:
    config = _prepare_model_config({"gcp": {"edge_scalar_dim": 32}})

    assert config.gcp.edge_scalar_input_dim == DEFAULT_EDGE_SCALAR_INPUT_DIM


def test_gcpconv_supports_bfloat16_inputs() -> None:
    conv = GCPNetEncoder().layers[0]

    node_scalars = torch.randn(6, conv.scalar_dim, dtype=torch.bfloat16)
    node_vectors = torch.randn(6, conv.vector_dim, 3, dtype=torch.bfloat16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_scalars = torch.randn(edge_index.shape[1], conv.edge_scalar_dim, dtype=torch.bfloat16)
    edge_vectors = torch.randn(edge_index.shape[1], conv.edge_vector_channels, 3, dtype=torch.bfloat16)
    edge_frames = torch.randn(edge_index.shape[1], 3, 3, dtype=torch.bfloat16)

    scalars_out, vectors_out = conv(
        node_scalars,
        node_vectors,
        edge_index,
        edge_scalars,
        edge_vectors,
        edge_frames,
    )

    assert scalars_out.dtype is torch.bfloat16
    assert vectors_out.dtype is torch.bfloat16


def test_gcpnet_encoder_supports_bfloat16_inputs() -> None:
    config = GCPNetConfig(layers=2, hidden_scalar_dim=16, hidden_vector_dim=8, latent_dim=32)
    encoder = GCPNetEncoder(config)

    num_nodes = 5
    num_edges = 4

    node_scalars = torch.randn(num_nodes, config.node_scalar_dim, dtype=torch.bfloat16)
    node_vectors = torch.randn(num_nodes, config.node_vector_dim, 3, dtype=torch.bfloat16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_scalars = torch.randn(num_edges, config.edge_scalar_dim, dtype=torch.bfloat16)
    edge_vectors = torch.randn(num_edges, config.edge_vector_dim, 3, dtype=torch.bfloat16)
    edge_frames = torch.randn(num_edges, 3, 3, dtype=torch.bfloat16)

    outputs = encoder(
        node_scalars,
        node_vectors,
        edge_index,
        edge_scalars,
        edge_vectors,
        edge_frames,
    )

    assert outputs["embeddings"].dtype is torch.bfloat16
    assert outputs["node_scalars"].dtype is torch.bfloat16
    assert outputs["node_vectors"].dtype is torch.bfloat16
