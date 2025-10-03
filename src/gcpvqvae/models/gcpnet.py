"""Graph Convolutional Point (GCP) encoder used by GCP-VQVAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .gcpcore import scalarize, safe_norm, vector_linear, vectorize
from gcpvqvae.data.batch import EdgeStorage, ProteinBatch
from gcpvqvae.geometry.ops import centralize, ensure_edge_frames, localize


# Default number of scalar edge features produced by the featurisation pipeline.
DEFAULT_EDGE_SCALAR_INPUT_DIM = 8


@dataclass
class ScalarVector:
    """Container bundling scalar and vector node/edge features."""

    scalars: Tensor
    vectors: Tensor

    def clone(self) -> "ScalarVector":
        return ScalarVector(self.scalars.clone(), self.vectors.clone())

    def to(self, *args, **kwargs) -> "ScalarVector":
        return ScalarVector(self.scalars.to(*args, **kwargs), self.vectors.to(*args, **kwargs))

    def apply_mask(self, mask: Tensor) -> "ScalarVector":
        if mask.ndim != 1:
            raise ValueError("Mask must be a 1D tensor")
        scalars = self.scalars * mask.unsqueeze(-1).to(self.scalars.dtype)
        vectors = self.vectors * mask.unsqueeze(-1).unsqueeze(-1).to(self.vectors.dtype)
        return ScalarVector(scalars, vectors)

    def detach(self) -> "ScalarVector":
        return ScalarVector(self.scalars.detach(), self.vectors.detach())

    def cat(self, other: "ScalarVector") -> "ScalarVector":
        scalars = torch.cat((self.scalars, other.scalars), dim=-1)
        vectors = torch.cat((self.vectors, other.vectors), dim=-2)
        return ScalarVector(scalars, vectors)


class VectorDropout(nn.Module):
    """Dropout variant that shares the mask across 3D vector components."""

    def __init__(self, p: float = 0.0, *, enabled: bool = True) -> None:
        super().__init__()
        self.p = float(p)
        self.enabled = enabled and self.p > 0.0

    def forward(self, vectors: Tensor) -> Tensor:
        if not self.enabled or self.p <= 0.0 or not self.training:
            return vectors
        if vectors.numel() == 0:
            return vectors
        if self.p >= 1.0:
            return torch.zeros_like(vectors)

        keep_prob = 1.0 - self.p
        mask = torch.rand(vectors.shape[:-1], device=vectors.device, dtype=torch.float32)
        mask = (mask < keep_prob).to(vectors.dtype) / keep_prob
        return vectors * mask.unsqueeze(-1)


class VectorLayerNorm(nn.Module):
    """LayerNorm-style normalisation for vector features."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, vectors: Tensor) -> Tensor:
        if vectors.size(-1) != 3:
            raise ValueError("VectorLayerNorm expects (..., C, 3) tensors")

        norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        mean = norms.mean(dim=-2, keepdim=True)
        scaled = vectors / torch.clamp(mean, min=self.eps)

        shape = [1] * (scaled.ndim - 2) + [-1, 1]
        return scaled * self.weight.view(*shape)


class GCPLayerNorm(nn.Module):
    """LayerNorm wrapper that handles coupled scalar/vector feature sets."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        normalize_scalars: bool = True,
        normalize_vectors: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.scalar_norm: Optional[nn.LayerNorm]
        self.vector_norm: Optional[VectorLayerNorm]

        self.scalar_norm = nn.LayerNorm(scalar_dim, eps=eps) if normalize_scalars and scalar_dim > 0 else None
        self.vector_norm = (
            VectorLayerNorm(vector_dim, eps=eps)
            if normalize_vectors and vector_dim > 0
            else None
        )

    def forward(self, features: ScalarVector) -> ScalarVector:
        scalars = features.scalars
        vectors = features.vectors
        scalar_dtype = scalars.dtype
        vector_dtype = vectors.dtype
        if self.scalar_norm is not None:
            scalars = self.scalar_norm(scalars.to(self.scalar_norm.weight.dtype)).to(scalar_dtype)
        if self.vector_norm is not None:
            vectors = self.vector_norm(vectors.to(self.vector_norm.weight.dtype)).to(vector_dtype)
        return ScalarVector(scalars, vectors)


class GCPDropout(nn.Module):
    """Coupled dropout module for scalar/vector feature tuples."""

    def __init__(
        self,
        p: float = 0.0,
        *,
        drop_scalars: bool = True,
        drop_vectors: bool = True,
    ) -> None:
        super().__init__()
        self.scalar_dropout: Optional[nn.Dropout]
        self.vector_dropout: Optional[VectorDropout]

        self.scalar_dropout = nn.Dropout(p) if drop_scalars and p > 0.0 else None
        self.vector_dropout = VectorDropout(p, enabled=drop_vectors and p > 0.0) if drop_vectors else None

    def forward(self, features: ScalarVector) -> ScalarVector:
        scalars = features.scalars
        vectors = features.vectors
        if self.scalar_dropout is not None:
            scalars = self.scalar_dropout(scalars)
        if self.vector_dropout is not None:
            vectors = self.vector_dropout(vectors)
        return ScalarVector(scalars, vectors)


class GCPEmbedding(nn.Module):
    """Project raw node and edge features to the working feature spaces."""

    def __init__(self, config: "GCPNetConfig") -> None:
        super().__init__()

        self.config = config
        self.edge_scalar_in_dim = (
            config.edge_scalar_input_dim if config.edge_scalar_input_dim is not None else config.edge_scalar_dim
        )
        self.edge_vector_in_dim = (
            config.edge_vector_input_dim if config.edge_vector_input_dim is not None else config.edge_vector_dim
        )

        self.node_scalar_proj = nn.Linear(
            config.node_scalar_dim,
            config.hidden_scalar_dim,
            bias=False,
        )
        self.node_vector_proj = nn.Parameter(
            torch.randn(config.hidden_vector_dim, config.node_vector_dim) * 0.02
        )

        self.num_rbf = 16
        centres = torch.linspace(0.0, 20.0, steps=self.num_rbf)
        self.register_buffer("rbf_centres", centres)
        self.rbf_sigma = 2.0

        self.edge_scalar_proj = nn.Linear(
            self.edge_scalar_in_dim + self.num_rbf,
            config.edge_scalar_dim,
            bias=False,
        )
        self.edge_vector_proj = nn.Parameter(
            torch.randn(config.edge_vector_dim, self.edge_vector_in_dim) * 0.02
        )

    def _gaussian_rbf(self, distances: Tensor) -> Tensor:
        diff = distances.unsqueeze(-1) - self.rbf_centres.to(distances.dtype)
        return torch.exp(-0.5 * (diff / self.rbf_sigma) ** 2)

    def forward(
        self,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_scalars: Tensor,
        edge_vectors: Tensor,
        edge_index: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[ScalarVector, ScalarVector]:
        node_scalar_dtype = self.node_scalar_proj.weight.dtype
        projected_scalars = self.node_scalar_proj(node_scalars.to(node_scalar_dtype)).to(node_scalars.dtype)
        projected_vectors = vector_linear(node_vectors, self.node_vector_proj).to(node_vectors.dtype)

        node_features = ScalarVector(projected_scalars, projected_vectors)
        if mask is not None:
            node_features = node_features.apply_mask(mask)

        if edge_scalars.numel() == 0:
            scalar_input = node_scalars.new_zeros((edge_index.shape[1], self.edge_scalar_in_dim))
        else:
            scalar_input = edge_scalars

        src, dst = edge_index
        distances = safe_norm(positions[dst] - positions[src], dim=-1)
        rbf = self._gaussian_rbf(distances)

        scalar_proj_dtype = self.edge_scalar_proj.weight.dtype
        scalar_concat = torch.cat((scalar_input.to(scalar_proj_dtype), rbf.to(scalar_proj_dtype)), dim=-1)
        projected_edge_scalars = self.edge_scalar_proj(scalar_concat)
        edge_scalar_dtype = scalar_input.dtype if scalar_input.numel() else node_scalars.dtype
        projected_edge_scalars = projected_edge_scalars.to(edge_scalar_dtype)

        if edge_vectors.numel() == 0:
            edge_vec = edge_vectors.new_zeros((edge_index.shape[1], self.edge_vector_in_dim, 3))
        else:
            edge_vec = edge_vectors

        projected_edge_vectors = vector_linear(edge_vec, self.edge_vector_proj).to(edge_vec.dtype)

        edge_features = ScalarVector(projected_edge_scalars, projected_edge_vectors)
        if mask is not None:
            src_mask = mask[src]
            dst_mask = mask[dst]
            edge_mask = (src_mask & dst_mask).to(projected_edge_scalars.dtype)
            if projected_edge_scalars.numel():
                projected_edge_scalars = projected_edge_scalars * edge_mask.unsqueeze(-1)
            projected_edge_vectors = projected_edge_vectors * edge_mask.unsqueeze(-1).unsqueeze(-1).to(projected_edge_vectors.dtype)
            edge_features = ScalarVector(projected_edge_scalars, projected_edge_vectors)

        return node_features, edge_features


class GCPConv(nn.Module):
    """Single GCP convolution layer operating on ``ScalarVector`` features."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        edge_scalar_dim: int,
        edge_vector_channels: int,
        hidden_scalar_dim: int,
        hidden_vector_channels: int,
        dropout: float = 0.0,
        vector_gate: bool = True,
        enable_e3_equivariance: bool = True,
        node_inputs: bool = True,
    ) -> None:
        super().__init__()

        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.edge_scalar_dim = edge_scalar_dim
        self.edge_vector_channels = edge_vector_channels
        self.hidden_vector_channels = hidden_vector_channels
        self.vector_gate = vector_gate
        self.enable_e3_equivariance = enable_e3_equivariance
        self.node_inputs = node_inputs

        combined_vector_dim = (vector_dim if node_inputs else 0) + edge_vector_channels
        if hidden_vector_channels > 0 and combined_vector_dim > 0:
            self.vector_down = nn.Parameter(
                torch.randn(hidden_vector_channels, combined_vector_dim) * 0.02
            )
            self.message_vector_channels = hidden_vector_channels
        else:
            self.vector_down = None
            self.message_vector_channels = 0

        if vector_dim > 0 and self.message_vector_channels > 0:
            self.vector_up = nn.Parameter(
                torch.randn(vector_dim, self.message_vector_channels) * 0.02
            )
        else:
            self.vector_up = None

        self.node_norm = GCPLayerNorm(scalar_dim, vector_dim)
        self.edge_norm = GCPLayerNorm(edge_scalar_dim, edge_vector_channels)

        scalar_in_dim = 0
        if node_inputs and scalar_dim > 0:
            scalar_in_dim += scalar_dim
        scalar_in_dim += self.message_vector_channels * 3
        scalar_in_dim += self.message_vector_channels
        scalar_in_dim += edge_scalar_dim

        if scalar_dim > 0 and scalar_in_dim == 0:
            raise ValueError("GCPConv requires a non-empty scalar input when scalar_dim > 0")

        if scalar_dim > 0 and scalar_in_dim > 0:
            self.scalar_mlp: Optional[nn.Module] = nn.Sequential(
                nn.Linear(scalar_in_dim, hidden_scalar_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_scalar_dim, scalar_dim, bias=False),
            )
        else:
            self.scalar_mlp = None

        if vector_gate and vector_dim > 0 and scalar_in_dim > 0:
            self.gate_mlp: Optional[nn.Module] = nn.Sequential(
                nn.Linear(scalar_in_dim, hidden_scalar_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_scalar_dim, vector_dim, bias=False),
            )
        else:
            self.gate_mlp = None

        self.update_dropout = GCPDropout(dropout)

    def forward(
        self,
        nodes: ScalarVector,
        edges: ScalarVector,
        edge_index: Tensor,
        edge_frames: Tensor,
        mask: Optional[Tensor] = None,
    ) -> ScalarVector:
        if nodes.scalars.shape[0] != nodes.vectors.shape[0]:
            raise ValueError("Scalar and vector node features must align")

        src, dst = edge_index
        num_nodes = nodes.scalars.shape[0]
        device = nodes.scalars.device
        scalar_dtype = nodes.scalars.dtype
        vector_dtype = nodes.vectors.dtype

        compute_scalar_dtype = torch.float32 if scalar_dtype in {torch.float16, torch.bfloat16} else scalar_dtype
        compute_vector_dtype = torch.float32 if vector_dtype in {torch.float16, torch.bfloat16} else vector_dtype

        norm_nodes = self.node_norm(nodes)
        norm_edges = self.edge_norm(edges)

        node_context = (
            norm_nodes.scalars.to(scalar_dtype)
            if self.node_inputs and norm_nodes.scalars.numel() > 0
            else nodes.scalars.new_zeros((num_nodes, 0))
        )

        counts = torch.zeros((num_nodes, 1), dtype=compute_scalar_dtype, device=device)
        if dst.numel() > 0:
            ones = torch.ones((dst.numel(), 1), dtype=compute_scalar_dtype, device=device)
            counts.index_add_(0, dst, ones)
        counts = torch.clamp(counts, min=1.0)

        combined_vectors = []
        if self.node_inputs and norm_nodes.vectors.size(-2) > 0:
            combined_vectors.append(norm_nodes.vectors.index_select(0, src))
        if norm_edges.vectors.size(-2) > 0:
            combined_vectors.append(norm_edges.vectors)

        if self.vector_down is not None and combined_vectors:
            edge_vectors = torch.cat(combined_vectors, dim=-2)
            down_vectors = vector_linear(edge_vectors, self.vector_down).to(vector_dtype)
            down_vectors_compute = down_vectors.to(compute_vector_dtype)
            agg_vectors = torch.zeros(
                (num_nodes, self.message_vector_channels, 3),
                dtype=compute_vector_dtype,
                device=device,
            )
            agg_vectors.index_add_(0, dst, down_vectors_compute)
            agg_vectors = (agg_vectors / counts.to(compute_vector_dtype).unsqueeze(-1)).to(vector_dtype)

            down_norm = safe_norm(down_vectors, dim=-1).to(compute_scalar_dtype)
            agg_norm = torch.zeros(
                (num_nodes, self.message_vector_channels),
                dtype=compute_scalar_dtype,
                device=device,
            )
            agg_norm.index_add_(0, dst, down_norm)
            agg_norm = (agg_norm / counts).to(scalar_dtype)

            projected = scalarize(down_vectors, edge_frames).to(compute_scalar_dtype)
            agg_projected = torch.zeros(
                (num_nodes, self.message_vector_channels * 3),
                dtype=compute_scalar_dtype,
                device=device,
            )
            agg_projected.index_add_(0, dst, projected)
            agg_projected = (agg_projected / counts).to(scalar_dtype)
        else:
            agg_vectors = nodes.vectors.new_zeros((num_nodes, self.message_vector_channels, 3))
            agg_norm = nodes.scalars.new_zeros((num_nodes, self.message_vector_channels))
            agg_projected = nodes.scalars.new_zeros((num_nodes, self.message_vector_channels * 3))

        if norm_edges.scalars.numel() > 0 and dst.numel() > 0:
            edge_scalars = norm_edges.scalars.to(compute_scalar_dtype)
            agg_edge = torch.zeros((num_nodes, self.edge_scalar_dim), dtype=compute_scalar_dtype, device=device)
            agg_edge.index_add_(0, dst, edge_scalars)
            agg_edge = (agg_edge / counts).to(scalar_dtype)
        else:
            agg_edge = nodes.scalars.new_zeros((num_nodes, self.edge_scalar_dim))

        scalar_inputs = []
        if node_context.numel() > 0:
            scalar_inputs.append(node_context)
        if agg_projected.numel() > 0:
            scalar_inputs.append(agg_projected)
        if agg_norm.numel() > 0:
            scalar_inputs.append(agg_norm)
        if agg_edge.numel() > 0:
            scalar_inputs.append(agg_edge)

        if scalar_inputs:
            scalar_input = torch.cat(scalar_inputs, dim=-1)
        else:
            scalar_input = nodes.scalars.new_zeros((num_nodes, 0))

        if self.scalar_mlp is not None and scalar_input.numel() > 0:
            mlp_dtype = self.scalar_mlp[0].weight.dtype  # type: ignore[index]
            scalar_update = self.scalar_mlp(scalar_input.to(mlp_dtype)).to(scalar_dtype)
        else:
            scalar_update = nodes.scalars.new_zeros_like(nodes.scalars)

        gate: Optional[Tensor]
        if self.gate_mlp is not None and scalar_input.numel() > 0:
            gate_dtype = self.gate_mlp[0].weight.dtype  # type: ignore[index]
            gate = torch.sigmoid(self.gate_mlp(scalar_input.to(gate_dtype))).to(vector_dtype)
        else:
            gate = None

        if self.vector_up is not None:
            vector_update = vectorize(
                agg_vectors,
                self.vector_up,
                gate=gate,
                vector_gate=self.vector_gate,
                enable_e3_equivariance=self.enable_e3_equivariance,
            ).to(vector_dtype)
        else:
            vector_update = nodes.vectors.new_zeros_like(nodes.vectors)

        updates = ScalarVector(scalar_update, vector_update)
        updates = self.update_dropout(updates)

        scalars_out = nodes.scalars + updates.scalars
        vectors_out = nodes.vectors + updates.vectors

        if mask is not None:
            scalars_out = scalars_out * mask.unsqueeze(-1).to(scalars_out.dtype)
            vectors_out = vectors_out * mask.unsqueeze(-1).unsqueeze(-1).to(vectors_out.dtype)

        return ScalarVector(scalars_out, vectors_out)


class GCPMessagePassing(nn.Module):
    """Stack of GCP convolutional modules with sparse aggregation."""

    def __init__(self, config: "GCPNetConfig") -> None:
        super().__init__()

        scalar_dim = config.hidden_scalar_dim
        vector_dim = config.hidden_vector_dim
        edge_scalar_dim = config.edge_scalar_dim
        edge_vector_dim = config.edge_vector_dim

        bottleneck_scalar = max(1, scalar_dim // 2)
        bottleneck_vector = max(1, vector_dim // 2)

        self.layers = nn.ModuleList(
            [
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=bottleneck_scalar,
                    hidden_vector_channels=bottleneck_vector,
                    dropout=config.dropout,
                    vector_gate=config.vector_gate,
                    enable_e3_equivariance=config.enable_e3_equivariance,
                    node_inputs=config.node_inputs,
                ),
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=scalar_dim,
                    hidden_vector_channels=vector_dim,
                    dropout=config.dropout,
                    vector_gate=config.vector_gate,
                    enable_e3_equivariance=config.enable_e3_equivariance,
                    node_inputs=config.node_inputs,
                ),
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=scalar_dim,
                    hidden_vector_channels=vector_dim,
                    dropout=config.dropout,
                    vector_gate=config.vector_gate,
                    enable_e3_equivariance=config.enable_e3_equivariance,
                    node_inputs=config.node_inputs,
                ),
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=bottleneck_scalar,
                    hidden_vector_channels=bottleneck_vector,
                    dropout=config.dropout,
                    vector_gate=config.vector_gate,
                    enable_e3_equivariance=config.enable_e3_equivariance,
                    node_inputs=config.node_inputs,
                ),
            ]
        )

    @staticmethod
    def _build_adjacency(
        edge_index: Tensor,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
        mask: Optional[Tensor],
    ) -> torch.Tensor:
        src, dst = edge_index
        values = torch.ones(src.numel(), dtype=dtype, device=device)
        if mask is not None:
            src_mask = mask[src].to(dtype)
            dst_mask = mask[dst].to(dtype)
            values = values * src_mask * dst_mask
        indices = torch.stack((dst, src))
        adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)
        return adjacency.coalesce()

    def _aggregate(self, features: ScalarVector, edge_index: Tensor, mask: Optional[Tensor]) -> ScalarVector:
        num_nodes = features.scalars.shape[0]
        device = features.scalars.device
        dtype = features.scalars.dtype
        compute_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        adjacency = self._build_adjacency(edge_index, num_nodes, compute_dtype, device, mask)

        ones = torch.ones((num_nodes, 1), device=device, dtype=compute_dtype)
        degree = torch.sparse.mm(adjacency, ones)
        degree = torch.clamp(degree, min=1.0)

        scalar_input = features.scalars.to(compute_dtype)
        agg_scalars = torch.sparse.mm(adjacency, scalar_input)
        agg_scalars = (agg_scalars / degree).to(dtype)

        vector_shape = features.vectors.shape
        vectors_flat = features.vectors.reshape(num_nodes, -1).to(compute_dtype)
        agg_vectors_flat = torch.sparse.mm(adjacency, vectors_flat)
        agg_vectors_flat = (agg_vectors_flat / degree).to(features.vectors.dtype)
        agg_vectors = agg_vectors_flat.view(vector_shape)

        if mask is not None:
            agg_scalars = agg_scalars * mask.unsqueeze(-1).to(agg_scalars.dtype)
            agg_vectors = agg_vectors * mask.unsqueeze(-1).unsqueeze(-1).to(agg_vectors.dtype)

        return ScalarVector(agg_scalars, agg_vectors)

    def forward(
        self,
        features: ScalarVector,
        edges: ScalarVector,
        edge_index: Tensor,
        edge_frames: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[ScalarVector, ScalarVector]:
        updated = features
        for layer in self.layers:
            updated = layer(updated, edges, edge_index, edge_frames, mask=mask)
            if mask is not None:
                updated = updated.apply_mask(mask)

        aggregated = self._aggregate(updated, edge_index, mask)
        return updated, aggregated


@dataclass
class GCPNetConfig:
    node_scalar_dim: int = 6
    node_vector_dim: int = 3
    edge_scalar_dim: int = 32
    edge_scalar_input_dim: Optional[int] = DEFAULT_EDGE_SCALAR_INPUT_DIM
    edge_vector_dim: int = 4
    edge_vector_input_dim: Optional[int] = 1
    hidden_scalar_dim: int = 128
    hidden_vector_dim: int = 16
    latent_dim: int = 256
    layers: int = 6
    dropout: float = 0.0
    vector_gate: bool = True
    enable_e3_equivariance: bool = True
    node_inputs: bool = True
    displacement_head: bool = False
    prenorm: bool = True
    init: str = "random"
    init_checkpoint: Optional[str] = None
    strict_init: bool = True


class GCPInteractions(nn.Module):
    """Residual interaction block wrapping message passing and feed-forward GCP stacks."""

    def __init__(self, config: GCPNetConfig) -> None:
        super().__init__()

        self.config = config
        self.prenorm = config.prenorm

        self.prenorm_layer = GCPLayerNorm(
            config.hidden_scalar_dim,
            config.hidden_vector_dim,
        ) if self.prenorm else None

        self.message_passing = GCPMessagePassing(config)
        self.residual_dropout = GCPDropout(config.dropout)

        self.skip_proj = nn.Linear(config.hidden_scalar_dim * 2, config.hidden_scalar_dim, bias=False)
        self.skip_vector_proj = nn.Parameter(
            torch.randn(config.hidden_vector_dim, config.hidden_vector_dim * 2) * 0.02
        )

        self.feed_forward = nn.ModuleList(
            [
                GCPConv(
                    config.hidden_scalar_dim,
                    config.hidden_vector_dim,
                    edge_scalar_dim=config.edge_scalar_dim,
                    edge_vector_channels=config.edge_vector_dim,
                    hidden_scalar_dim=config.hidden_scalar_dim * 2,
                    hidden_vector_channels=config.hidden_vector_dim,
                    dropout=config.dropout,
                    vector_gate=config.vector_gate,
                    enable_e3_equivariance=config.enable_e3_equivariance,
                    node_inputs=config.node_inputs,
                ),
                GCPConv(
                    config.hidden_scalar_dim,
                    config.hidden_vector_dim,
                    edge_scalar_dim=config.edge_scalar_dim,
                    edge_vector_channels=config.edge_vector_dim,
                    hidden_scalar_dim=config.hidden_scalar_dim * 2,
                    hidden_vector_channels=config.hidden_vector_dim,
                    dropout=config.dropout,
                    vector_gate=config.vector_gate,
                    enable_e3_equivariance=config.enable_e3_equivariance,
                    node_inputs=config.node_inputs,
                ),
            ]
        )

        self.feedforward_dropout = GCPDropout(config.dropout)

        if config.displacement_head:
            self.node_position_head = nn.Linear(config.hidden_scalar_dim, 3, bias=False)
        else:
            self.node_position_head = None

    def forward(
        self,
        features: ScalarVector,
        edges: ScalarVector,
        edge_index: Tensor,
        edge_frames: Tensor,
        mask: Optional[Tensor] = None,
        skip: Optional[ScalarVector] = None,
    ) -> Tuple[ScalarVector, ScalarVector, Optional[Tensor]]:
        x = features
        if self.prenorm_layer is not None:
            x = self.prenorm_layer(features)

        message_out, aggregated = self.message_passing(x, edges, edge_index, edge_frames, mask=mask)
        if mask is not None:
            message_out = message_out.apply_mask(mask)
            aggregated = aggregated.apply_mask(mask)

        message_update = self.residual_dropout(message_out)
        updated = ScalarVector(
            features.scalars + message_update.scalars,
            features.vectors + message_update.vectors,
        )

        skip_features = skip if skip is not None else aggregated
        combined_scalars = torch.cat((updated.scalars, skip_features.scalars), dim=-1)
        combined_vectors = torch.cat((updated.vectors, skip_features.vectors), dim=-2)

        skip_dtype = self.skip_proj.weight.dtype
        projected_scalars = self.skip_proj(combined_scalars.to(skip_dtype)).to(updated.scalars.dtype)
        projected_vectors = vector_linear(combined_vectors, self.skip_vector_proj).to(updated.vectors.dtype)
        feed_forward_input = ScalarVector(projected_scalars, projected_vectors)

        feed_forward_output = feed_forward_input
        for layer in self.feed_forward:
            feed_forward_output = layer(feed_forward_output, edges, edge_index, edge_frames, mask=mask)
            if mask is not None:
                feed_forward_output = feed_forward_output.apply_mask(mask)

        feed_forward_output = self.feedforward_dropout(feed_forward_output)

        result = ScalarVector(
            updated.scalars + feed_forward_output.scalars,
            updated.vectors + feed_forward_output.vectors,
        )

        displacement: Optional[Tensor]
        if self.node_position_head is not None:
            head_dtype = self.node_position_head.weight.dtype
            displacement = self.node_position_head(result.scalars.to(head_dtype)).to(result.scalars.dtype)
            if mask is not None:
                displacement = displacement * mask.unsqueeze(-1)
        else:
            displacement = None

        return result, aggregated, displacement


class GCPNetEncoder(nn.Module):
    """Stack of GCP interaction layers with scalar/vector read-out."""

    def __init__(self, config: Optional[GCPNetConfig] = None) -> None:
        super().__init__()

        self.config = config or GCPNetConfig()

        if self.config.edge_scalar_input_dim is None:
            self.config.edge_scalar_input_dim = self.config.edge_scalar_dim
        if self.config.edge_vector_input_dim is None:
            self.config.edge_vector_input_dim = self.config.edge_vector_dim

        self.embedding = GCPEmbedding(self.config)
        num_layers = self.config.layers if self.config.layers is not None else 6
        self.interactions = nn.ModuleList(
            [GCPInteractions(self.config) for _ in range(num_layers)]
        )

        readout_dim = self.config.hidden_scalar_dim + self.config.hidden_vector_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, self.config.latent_dim, bias=False),
        )

    def _prepare_edges(
        self,
        batch: ProteinBatch,
        mask: Optional[Tensor],
    ) -> Tuple[EdgeStorage, Tensor]:
        if isinstance(batch.e, dict):
            knn_keys = [name for name in batch.e if name.startswith("knn")]
            if len(knn_keys) != 1:
                raise ValueError("ProteinBatch must contain exactly one knn relation")
            edges = batch.e[knn_keys[0]]
        else:
            raise ValueError("ProteinBatch.e must be a mapping of edge relations")

        frames = edges.frames
        if frames is None or frames.shape[0] != edges.edge_index.shape[1]:
            frames = ensure_edge_frames(
                batch.xi,
                edges.edge_index,
                edge_batch=edges.batch,
                node_batch=batch.batch,
                mask=mask,
            )
            edges.frames = frames
        batch.edge_frames = frames
        return edges, frames

    def forward(self, batch: ProteinBatch) -> Dict[str, Union[Tensor, ProteinBatch]]:
        if not isinstance(batch, ProteinBatch):
            raise TypeError("GCPNetEncoder.forward expects a ProteinBatch")

        node_scalars = batch.h
        node_vectors = batch.chi
        if node_scalars.ndim != 2:
            raise ValueError("ProteinBatch.h must have shape (N, F)")
        if node_vectors.ndim != 3:
            raise ValueError("ProteinBatch.chi must have shape (N, C, 3)")

        mask = batch.mask.to(torch.bool) if batch.mask is not None else None

        batch.xi_raw = batch.xi.clone()
        centered_positions, centroids = centralize(batch.xi, batch.batch, mask=mask)
        batch.xi = centered_positions
        batch.centroids = centroids

        edges, frames = self._prepare_edges(batch, mask)

        edge_vectors = edges.vectors
        if edge_vectors.ndim == 2:
            edge_vectors = edge_vectors.unsqueeze(1)
        edge_vectors = localize(edge_vectors, frames).to(node_vectors.dtype)

        node_features, edge_features = self.embedding(
            node_scalars,
            node_vectors,
            edges.scalars,
            edge_vectors,
            edges.edge_index,
            batch.xi,
            mask,
        )

        displacements: Optional[Tensor] = None
        skip_features: Optional[ScalarVector] = None
        for layer in self.interactions:
            node_features, skip_features, displacement = layer(
                node_features,
                edge_features,
                edges.edge_index,
                frames,
                mask=mask,
                skip=skip_features,
            )
            if displacement is not None:
                displacements = displacement

        vec_norms = safe_norm(node_features.vectors, dim=-1)
        readout_input = torch.cat((node_features.scalars, vec_norms.to(node_features.scalars.dtype)), dim=-1)
        readout_dtype = self.readout[1].weight.dtype
        embeddings = self.readout(readout_input.to(readout_dtype)).to(node_features.scalars.dtype)

        output: Dict[str, Union[Tensor, ProteinBatch]] = {
            "embeddings": embeddings,
            "node_scalars": node_features.scalars,
            "node_vectors": node_features.vectors,
            "batch": batch,
        }

        if displacements is not None:
            output["displacement"] = displacements

        return output


__all__ = [
    "GCPNetEncoder",
    "GCPNetConfig",
    "GCPConv",
    "GCPLayerNorm",
    "GCPDropout",
    "VectorDropout",
    "VectorLayerNorm",
    "ScalarVector",
    "GCPEmbedding",
    "GCPMessagePassing",
    "GCPInteractions",
    "DEFAULT_EDGE_SCALAR_INPUT_DIM",
]
