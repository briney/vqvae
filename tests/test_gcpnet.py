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
