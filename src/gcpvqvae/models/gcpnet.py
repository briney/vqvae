"""Graph Convolutional Point (GCP) encoder used by GCP-VQVAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .gcpcore import apply_gating, safe_norm, vector_linear
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


def _gather_edge_vectors(
    node_vectors: Tensor,
    edge_index: Tensor,
    edge_vectors: Tensor,
    *,
    edge_vector_channels: int,
) -> Tensor:
    src, _ = edge_index

    node_vec = node_vectors.index_select(0, src)
    if edge_vector_channels > 0:
        if edge_vectors.numel() == 0:
            edge_vec = torch.zeros(
                (src.numel(), edge_vector_channels, 3),
                dtype=node_vectors.dtype,
                device=node_vectors.device,
            )
        else:
            if edge_vectors.ndim == 2:
                edge_vec = edge_vectors.unsqueeze(1)
            else:
                edge_vec = edge_vectors
            if edge_vec.size(1) != edge_vector_channels:
                raise ValueError("edge_vectors has incompatible channel dimension")
    else:
        edge_vec = torch.zeros(
            (src.numel(), 0, 3),
            dtype=node_vectors.dtype,
            device=node_vectors.device,
        )

    combined_vectors = torch.cat((node_vec, edge_vec), dim=1)
    return combined_vectors


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
    """Single GCP convolution layer implementing Algorithm 1 from GCPNet."""

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
    ) -> None:
        super().__init__()

        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.edge_scalar_dim = edge_scalar_dim
        self.edge_vector_channels = edge_vector_channels
        self.hidden_vector_channels = hidden_vector_channels

        combined_vector_dim = vector_dim + edge_vector_channels

        self.scalar_norm = nn.LayerNorm(scalar_dim)
        self.vector_norm = VectorLayerNorm(vector_dim)

        self.vec_down = nn.Parameter(torch.randn(hidden_vector_channels, combined_vector_dim) * 0.02)
        self.vec_signature = nn.Parameter(torch.randn(3, combined_vector_dim) * 0.02)
        self.vec_up = nn.Parameter(torch.randn(vector_dim, hidden_vector_channels) * 0.02)

        scalar_in_dim = scalar_dim + 9 + hidden_vector_channels + edge_scalar_dim

        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_in_dim, hidden_scalar_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_scalar_dim, scalar_dim, bias=False),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(scalar_in_dim, hidden_scalar_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_scalar_dim, vector_dim, bias=False),
        )

        self.dropout = nn.Dropout(dropout)
        self.vector_dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_index: Tensor,
        edge_scalars: Tensor,
        edge_vectors: Tensor,
        edge_frames: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if node_scalars.shape[0] != node_vectors.shape[0]:
            raise ValueError("Scalar and vector node features must align")

        src, dst = edge_index
        num_nodes = node_scalars.shape[0]
        dtype = node_scalars.dtype
        device = node_scalars.device
        vector_dtype = node_vectors.dtype

        if src.numel() == 0:
            agg_vectors = torch.zeros(
                (num_nodes, self.hidden_vector_channels, 3), dtype=node_vectors.dtype, device=device
            )
            agg_q = torch.zeros((num_nodes, 9), dtype=dtype, device=device)
            agg_norm = torch.zeros((num_nodes, self.hidden_vector_channels), dtype=dtype, device=device)
            agg_edge = torch.zeros((num_nodes, self.edge_scalar_dim), dtype=dtype, device=device)
            counts = torch.ones((num_nodes, 1), dtype=dtype, device=device)
        else:
            norm_scalars = self.scalar_norm(node_scalars).to(dtype)
            norm_vectors = self.vector_norm(node_vectors).to(vector_dtype)

            combined_vectors = _gather_edge_vectors(
                norm_vectors, edge_index, edge_vectors, edge_vector_channels=self.edge_vector_channels
            )

            down = vector_linear(combined_vectors, self.vec_down).to(vector_dtype)
            down_norm = safe_norm(down, dim=-1).to(dtype)

            signature_vectors = vector_linear(combined_vectors, self.vec_signature).to(edge_frames.dtype)
            orient = torch.einsum("eac,ebc->eab", signature_vectors, edge_frames).reshape(-1, 9)
            orient = orient.to(dtype)

            counts = torch.zeros((num_nodes, 1), dtype=dtype, device=device)
            counts.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=dtype, device=device))
            counts = torch.clamp(counts, min=1.0)

            agg_vectors = torch.zeros(
                (num_nodes, self.hidden_vector_channels, 3), dtype=vector_dtype, device=device
            )
            agg_vectors.index_add_(0, dst, down)
            agg_vectors = agg_vectors / counts.unsqueeze(-1)

            agg_q = torch.zeros((num_nodes, 9), dtype=dtype, device=device)
            agg_q.index_add_(0, dst, orient)
            agg_q = agg_q / counts

            agg_norm = torch.zeros((num_nodes, self.hidden_vector_channels), dtype=dtype, device=device)
            agg_norm.index_add_(0, dst, down_norm)
            agg_norm = agg_norm / counts

            if self.edge_scalar_dim > 0:
                if edge_scalars.numel() == 0:
                    agg_edge = torch.zeros((num_nodes, self.edge_scalar_dim), dtype=dtype, device=device)
                else:
                    agg_edge = torch.zeros((num_nodes, self.edge_scalar_dim), dtype=dtype, device=device)
                    agg_edge.index_add_(0, dst, edge_scalars.to(dtype))
                    agg_edge = agg_edge / counts
            else:
                agg_edge = torch.zeros((num_nodes, 0), dtype=dtype, device=device)

            norm_scalars = norm_scalars  # retained for residual below

        if src.numel() == 0:
            norm_scalars = self.scalar_norm(node_scalars).to(dtype)

        scalar_input = torch.cat((norm_scalars, agg_q, agg_norm, agg_edge), dim=-1)

        mlp_input_dtype = self.scalar_mlp[0].weight.dtype
        gate_input_dtype = self.gate_mlp[0].weight.dtype

        scalar_update = self.scalar_mlp(scalar_input.to(mlp_input_dtype)).to(dtype)
        scalars_out = node_scalars + self.dropout(scalar_update)

        vector_update = vector_linear(agg_vectors, self.vec_up).to(vector_dtype)
        gate = torch.sigmoid(self.gate_mlp(scalar_input.to(gate_input_dtype))).to(vector_dtype)
        gated = apply_gating(vector_update, gate)
        vectors_out = node_vectors + self.vector_dropout(gated)

        return scalars_out, vectors_out


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
                ),
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=scalar_dim,
                    hidden_vector_channels=vector_dim,
                    dropout=config.dropout,
                ),
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=scalar_dim,
                    hidden_vector_channels=vector_dim,
                    dropout=config.dropout,
                ),
                GCPConv(
                    scalar_dim,
                    vector_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    edge_vector_channels=edge_vector_dim,
                    hidden_scalar_dim=bottleneck_scalar,
                    hidden_vector_channels=bottleneck_vector,
                    dropout=config.dropout,
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
        scalars = features.scalars
        vectors = features.vectors
        for layer in self.layers:
            scalars, vectors = layer(
                scalars,
                vectors,
                edge_index,
                edges.scalars,
                edges.vectors,
                edge_frames,
            )
            if mask is not None:
                scalars = scalars * mask.unsqueeze(-1)
                vectors = vectors * mask.unsqueeze(-1).unsqueeze(-1)

        updated = ScalarVector(scalars, vectors)
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

        if self.prenorm:
            self.scalar_norm = nn.LayerNorm(config.hidden_scalar_dim)
            self.vector_norm = VectorLayerNorm(config.hidden_vector_dim)
        else:
            self.scalar_norm = None
            self.vector_norm = None

        self.message_passing = GCPMessagePassing(config)
        self.residual_dropout = nn.Dropout(config.dropout)

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
                ),
                GCPConv(
                    config.hidden_scalar_dim,
                    config.hidden_vector_dim,
                    edge_scalar_dim=config.edge_scalar_dim,
                    edge_vector_channels=config.edge_vector_dim,
                    hidden_scalar_dim=config.hidden_scalar_dim * 2,
                    hidden_vector_channels=config.hidden_vector_dim,
                    dropout=config.dropout,
                ),
            ]
        )

        self.feedforward_dropout = nn.Dropout(config.dropout)

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
        if self.prenorm:
            scalars = self.scalar_norm(x.scalars)
            vectors = self.vector_norm(x.vectors).to(x.vectors.dtype)
            x = ScalarVector(scalars, vectors)

        message_out, aggregated = self.message_passing(x, edges, edge_index, edge_frames, mask=mask)
        if mask is not None:
            message_out = message_out.apply_mask(mask)
            aggregated = aggregated.apply_mask(mask)

        updated = ScalarVector(
            features.scalars + self.residual_dropout(message_out.scalars),
            features.vectors + self.residual_dropout(message_out.vectors),
        )

        skip_features = skip if skip is not None else aggregated
        combined_scalars = torch.cat((updated.scalars, skip_features.scalars), dim=-1)
        combined_vectors = torch.cat((updated.vectors, skip_features.vectors), dim=-2)

        skip_dtype = self.skip_proj.weight.dtype
        projected_scalars = self.skip_proj(combined_scalars.to(skip_dtype)).to(updated.scalars.dtype)
        projected_vectors = vector_linear(combined_vectors, self.skip_vector_proj).to(updated.vectors.dtype)
        feed_forward_input = ScalarVector(projected_scalars, projected_vectors)

        ff_scalars, ff_vectors = feed_forward_input.scalars, feed_forward_input.vectors
        for layer in self.feed_forward:
            ff_scalars, ff_vectors = layer(
                ff_scalars,
                ff_vectors,
                edge_index,
                edges.scalars,
                edges.vectors,
                edge_frames,
            )
            if mask is not None:
                ff_scalars = ff_scalars * mask.unsqueeze(-1)
                ff_vectors = ff_vectors * mask.unsqueeze(-1).unsqueeze(-1)

        feed_forward_output = ScalarVector(ff_scalars, ff_vectors)

        result = ScalarVector(
            updated.scalars + self.feedforward_dropout(feed_forward_output.scalars),
            updated.vectors + self.feedforward_dropout(feed_forward_output.vectors),
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
    "VectorLayerNorm",
    "ScalarVector",
    "GCPEmbedding",
    "GCPMessagePassing",
    "GCPInteractions",
    "DEFAULT_EDGE_SCALAR_INPUT_DIM",
]
