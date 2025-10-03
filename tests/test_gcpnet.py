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
from gcpvqvae.system.train_gcpnet import _prepare_model_config


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

    assert output["embeddings"].shape == (num_nodes, config.latent_dim)


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
    edge_scalars = torch.randn(edge_index.shape[1], conv.edge_scalar_dim, dtype=torch.bfloat16)
    edge_vectors = torch.randn(edge_index.shape[1], conv.edge_vector_channels, 3, dtype=torch.bfloat16)
    edge_frames = torch.eye(3, dtype=torch.bfloat16).expand(edge_index.shape[1], 3, 3).clone()

    node_features = ScalarVector(node_scalars, node_vectors)
    edge_features = ScalarVector(edge_scalars, edge_vectors)

    output = conv(node_features, edge_features, edge_index, edge_frames)

    assert output.scalars.dtype is torch.bfloat16
    assert output.vectors.dtype is torch.bfloat16


def test_gcpnet_encoder_supports_bfloat16_inputs() -> None:
    config = GCPNetConfig(
        message_passing=GCPMessagePassingConfig(width=GCPWidthConfig(scalar=16, vector=8)),
        feed_forward=GCPFeedForwardConfig(width=GCPWidthConfig(scalar=32, vector=8)),
        latent_dim=32,
        num_layers=2,
    )
    encoder = GCPNetEncoder(config)

    num_nodes = 5
    num_edges = 4

    node_scalars = torch.randn(num_nodes, config.node_scalar_dim, dtype=torch.bfloat16)
    node_vectors = torch.randn(num_nodes, config.node_vector_dim, 3, dtype=torch.bfloat16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_scalars = torch.randn(num_edges, DEFAULT_EDGE_SCALAR_INPUT_DIM, dtype=torch.bfloat16)
    edge_vectors = torch.randn(num_edges, config.edge_vector_input_dim, 3, dtype=torch.bfloat16)
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

    assert outputs["embeddings"].dtype is torch.bfloat16
    assert outputs["node_scalars"].dtype is torch.bfloat16
    assert outputs["node_vectors"].dtype is torch.bfloat16
