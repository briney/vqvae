"""Graph Convolutional Point (GCP) encoder used by GCP-VQVAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from .gcpcore import apply_gating, safe_norm, vector_linear


# Default number of scalar edge features produced by the featurisation pipeline.
DEFAULT_EDGE_SCALAR_INPUT_DIM = 8


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


@dataclass
class ScalarVector:
    """Container coupling scalar and vector channels."""

    scalars: Tensor
    vectors: Tensor

    def __post_init__(self) -> None:
        if self.scalars.ndim != 2:
            raise ValueError("Scalar features must have shape (N, C)")
        if self.vectors.ndim != 3 or self.vectors.size(-1) != 3:
            raise ValueError("Vector features must have shape (N, C, 3)")
        if self.scalars.size(0) != self.vectors.size(0):
            raise ValueError("Scalar and vector features must reference the same nodes")

    @property
    def num_nodes(self) -> int:
        return self.scalars.size(0)

    @property
    def scalar_channels(self) -> int:
        return self.scalars.size(-1)

    @property
    def vector_channels(self) -> int:
        return self.vectors.size(-2)

    def clone(self) -> ScalarVector:
        return ScalarVector(self.scalars.clone(), self.vectors.clone())

    def detach(self) -> ScalarVector:
        return ScalarVector(self.scalars.detach(), self.vectors.detach())

    def to(self, *args, **kwargs) -> ScalarVector:  # type: ignore[override]
        return ScalarVector(self.scalars.to(*args, **kwargs), self.vectors.to(*args, **kwargs))

    def apply_mask(self, mask: Optional[Tensor]) -> ScalarVector:
        if mask is None:
            return self
        if mask.ndim != 1 or mask.size(0) != self.num_nodes:
            raise ValueError("Mask must have shape (N,)")
        mask = mask.to(dtype=self.scalars.dtype, device=self.scalars.device)
        scalar_mask = mask.unsqueeze(-1)
        vector_mask = mask.unsqueeze(-1).unsqueeze(-1)
        return ScalarVector(self.scalars * scalar_mask, self.vectors * vector_mask)

    def add(self, other: ScalarVector) -> ScalarVector:
        if self.scalars.shape != other.scalars.shape or self.vectors.shape != other.vectors.shape:
            raise ValueError("ScalarVector addition requires matching shapes")
        return ScalarVector(self.scalars + other.scalars, self.vectors + other.vectors)

    def dropout(self, scalar_dropout: nn.Module, vector_dropout: nn.Module) -> ScalarVector:
        return ScalarVector(scalar_dropout(self.scalars), vector_dropout(self.vectors))

    @staticmethod
    def cat(features: Iterable[ScalarVector]) -> ScalarVector:
        feats = tuple(features)
        if not feats:
            raise ValueError("Expected at least one set of features to concatenate")
        num_nodes = feats[0].num_nodes
        if any(feat.num_nodes != num_nodes for feat in feats):
            raise ValueError("All ScalarVector inputs must have the same number of nodes")
        scalars = torch.cat([feat.scalars for feat in feats], dim=-1)
        vectors = torch.cat([feat.vectors for feat in feats], dim=-2)
        return ScalarVector(scalars, vectors)

    @staticmethod
    def zeros(
        num_nodes: int,
        scalar_channels: int,
        vector_channels: int,
        *,
        scalar_dtype: torch.dtype,
        vector_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> ScalarVector:
        vector_dtype = vector_dtype or scalar_dtype
        scalars = torch.zeros((num_nodes, scalar_channels), dtype=scalar_dtype, device=device)
        vectors = torch.zeros((num_nodes, vector_channels, 3), dtype=vector_dtype, device=device)
        return ScalarVector(scalars, vectors)


def _ensure_edge_vectors(edge_vectors: Tensor) -> Tensor:
    if edge_vectors.ndim == 2:
        return edge_vectors.unsqueeze(1)
    if edge_vectors.ndim == 3:
        if edge_vectors.size(-1) != 3:
            raise ValueError("edge_vectors must store 3D components in the last dimension")
        return edge_vectors
    raise ValueError("edge_vectors must have shape (E, 3) or (E, C, 3)")


def _gaussian_rbf(distances: Tensor, centres: Tensor, sigma: float) -> Tensor:
    diff = distances.unsqueeze(-1) - centres
    return torch.exp(-0.5 * (diff / sigma) ** 2)


class GCPEmbedding(nn.Module):
    """Project raw node/edge features into ScalarVector containers."""

    def __init__(
        self,
        node_scalar_in: int,
        node_vector_in: int,
        edge_scalar_in: int,
        edge_vector_in: int,
        *,
        node_scalar_out: int = 128,
        node_vector_out: int = 16,
        edge_scalar_out: int = 32,
        edge_vector_out: int = 4,
        rbf_sigma: float = 2.0,
        rbf_centres: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.node_scalar_proj = nn.Linear(node_scalar_in, node_scalar_out, bias=False)
        self.node_vector_proj = nn.Parameter(torch.randn(node_vector_out, node_vector_in) * 0.02)

        if rbf_centres is None:
            rbf_centres = torch.linspace(2.0, 16.0, steps=8)
        self.register_buffer("rbf_centres", rbf_centres)
        self.rbf_sigma = float(rbf_sigma)

        self.edge_scalar_in = edge_scalar_in
        self.edge_vector_in = edge_vector_in

        self.edge_scalar_proj = nn.Linear(edge_scalar_in + self.rbf_centres.numel(), edge_scalar_out, bias=False)
        if edge_vector_out > 0 and edge_vector_in > 0:
            self.edge_vector_proj = nn.Parameter(torch.randn(edge_vector_out, edge_vector_in) * 0.02)
        else:
            self.register_parameter("edge_vector_proj", None)

        self.node_scalar_out = node_scalar_out
        self.node_vector_out = node_vector_out
        self.edge_scalar_out = edge_scalar_out
        self.edge_vector_out = edge_vector_out

    def forward(
        self,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_scalars: Tensor,
        edge_vectors: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tuple[ScalarVector, ScalarVector]:
        if node_scalars.ndim != 2:
            raise ValueError("node_scalars must have shape (N, F)")
        if node_vectors.ndim != 3 or node_vectors.size(-1) != 3:
            raise ValueError("node_vectors must have shape (N, C, 3)")
        if edge_scalars.ndim != 2:
            raise ValueError("edge_scalars must have shape (E, F)")

        edge_vectors = _ensure_edge_vectors(edge_vectors)

        if edge_scalars.size(1) not in (0, self.edge_scalar_in):
            raise ValueError(
                f"Expected {self.edge_scalar_in} scalar edge features, received {edge_scalars.size(1)}"
            )
        if edge_vectors.size(1) != self.edge_vector_in:
            raise ValueError(
                f"Expected {self.edge_vector_in} vector edge channels, received {edge_vectors.size(1)}"
            )

        scalar_dtype = node_scalars.dtype
        vector_dtype = node_vectors.dtype

        node_scalar_proj = self.node_scalar_proj(node_scalars.to(self.node_scalar_proj.weight.dtype)).to(scalar_dtype)
        node_vector_proj = vector_linear(node_vectors, self.node_vector_proj).to(vector_dtype)
        node_features = ScalarVector(node_scalar_proj, node_vector_proj).apply_mask(mask)

        distances = safe_norm(edge_vectors, dim=-1).mean(dim=-1)
        rbf = _gaussian_rbf(
            distances,
            self.rbf_centres.to(device=distances.device, dtype=distances.dtype),
            self.rbf_sigma,
        )

        if edge_scalars.size(1) == 0 and self.edge_scalar_in > 0:
            edge_scalar_base = edge_scalars.new_zeros((edge_scalars.size(0), self.edge_scalar_in))
        else:
            edge_scalar_base = edge_scalars
        combined_edge_scalars = torch.cat(
            (edge_scalar_base, rbf.to(edge_scalar_base.dtype)),
            dim=-1,
        )
        edge_scalar_proj = self.edge_scalar_proj(
            combined_edge_scalars.to(self.edge_scalar_proj.weight.dtype)
        ).to(scalar_dtype)

        if self.edge_vector_proj is None:
            edge_vector_proj = torch.zeros(
                (edge_vectors.size(0), self.edge_vector_out, 3),
                dtype=vector_dtype,
                device=edge_vectors.device,
            )
        else:
            edge_vector_proj = vector_linear(edge_vectors, self.edge_vector_proj).to(vector_dtype)

        edge_features = ScalarVector(edge_scalar_proj, edge_vector_proj)

        return node_features, edge_features


def _sparse_mean(dst: Tensor, values: Tensor, num_nodes: int) -> Tensor:
    if values.shape[0] == 0:
        return values.new_zeros((num_nodes, values.shape[1]))

    edge_count = dst.numel()
    device = dst.device
    dtype = values.dtype
    mm_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    indices = torch.stack((dst, torch.arange(edge_count, device=device)))
    ones = torch.ones(edge_count, dtype=mm_dtype, device=device)
    matrix = torch.sparse_coo_tensor(indices, ones, size=(num_nodes, edge_count))

    values_mm = values.to(dtype=mm_dtype)
    aggregated = torch.sparse.mm(matrix, values_mm)
    counts = torch.sparse.mm(matrix, torch.ones((edge_count, 1), dtype=mm_dtype, device=device))
    aggregated = aggregated / torch.clamp(counts, min=1.0)

    return aggregated.to(dtype)


class GCPModule(nn.Module):
    """Lightweight GCP transformation used inside interaction layers."""

    def __init__(
        self,
        node_scalar_in: int,
        node_vector_in: int,
        node_scalar_out: int,
        node_vector_out: int,
        edge_scalar_dim: int,
        edge_vector_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.node_scalar_out = node_scalar_out
        self.node_vector_out = node_vector_out
        self.edge_scalar_dim = edge_scalar_dim
        self.edge_vector_dim = edge_vector_dim

        self.node_scalar_norm = nn.LayerNorm(node_scalar_in) if node_scalar_in > 0 else None
        self.node_vector_norm = VectorLayerNorm(node_vector_in) if node_vector_in > 0 else None
        self.edge_scalar_norm = nn.LayerNorm(edge_scalar_dim) if edge_scalar_dim > 0 else None
        self.edge_vector_norm = VectorLayerNorm(edge_vector_dim) if edge_vector_dim > 0 else None

        self.scalar_src = nn.Linear(node_scalar_in, node_scalar_out, bias=False)
        self.scalar_edge = nn.Linear(edge_scalar_dim, node_scalar_out, bias=False) if edge_scalar_dim > 0 else None
        self.scalar_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(node_scalar_out, node_scalar_out, bias=False),
        )

        if node_vector_out > 0 and node_vector_in > 0:
            self.vector_src = nn.Parameter(torch.randn(node_vector_out, node_vector_in) * 0.02)
        else:
            self.register_parameter("vector_src", None)

        if node_vector_out > 0 and edge_vector_dim > 0:
            self.vector_edge = nn.Parameter(torch.randn(node_vector_out, edge_vector_dim) * 0.02)
        else:
            self.register_parameter("vector_edge", None)

        self.vector_gate = nn.Linear(node_scalar_out, node_vector_out, bias=False) if node_vector_out > 0 else None

        self.scalar_dropout = nn.Dropout(dropout)
        self.vector_dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: ScalarVector,
        edges: ScalarVector,
        edge_index: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> ScalarVector:
        node_scalars = nodes.scalars
        node_vectors = nodes.vectors
        edge_scalars = edges.scalars
        edge_vectors = edges.vectors

        if node_scalars.size(0) != node_vectors.size(0):
            raise ValueError("Scalar and vector node features must align")

        src, dst = edge_index
        num_nodes = node_scalars.size(0)
        scalar_dtype = node_scalars.dtype
        vector_dtype = node_vectors.dtype
        device = node_scalars.device

        if self.node_scalar_norm is not None:
            norm_node_scalars = self.node_scalar_norm(
                node_scalars.to(self.node_scalar_norm.weight.dtype)
            ).to(scalar_dtype)
        else:
            norm_node_scalars = node_scalars

        if self.node_vector_norm is not None and node_vectors.numel():
            norm_node_vectors = self.node_vector_norm(
                node_vectors.to(self.node_vector_norm.weight.dtype)
            ).to(vector_dtype)
        else:
            norm_node_vectors = node_vectors

        if self.edge_scalar_norm is not None and edge_scalars.numel():
            norm_edge_scalars = self.edge_scalar_norm(
                edge_scalars.to(self.edge_scalar_norm.weight.dtype)
            ).to(edge_scalars.dtype)
        elif self.edge_scalar_norm is not None:
            norm_edge_scalars = edge_scalars.new_zeros((edge_scalars.size(0), self.edge_scalar_dim))
        else:
            norm_edge_scalars = edge_scalars

        if self.edge_vector_norm is not None and edge_vectors.numel():
            norm_edge_vectors = self.edge_vector_norm(
                edge_vectors.to(self.edge_vector_norm.weight.dtype)
            ).to(edge_vectors.dtype)
        else:
            norm_edge_vectors = edge_vectors

        if src.numel() == 0:
            aggregated_scalars = torch.zeros((num_nodes, self.node_scalar_out), dtype=scalar_dtype, device=device)
            aggregated_vectors = torch.zeros((num_nodes, self.node_vector_out, 3), dtype=vector_dtype, device=device)
        else:
            src_linear = self.scalar_src(norm_node_scalars.to(self.scalar_src.weight.dtype)).to(scalar_dtype)
            edge_linear = None
            if self.scalar_edge is not None and norm_edge_scalars.numel():
                edge_linear = self.scalar_edge(norm_edge_scalars.to(self.scalar_edge.weight.dtype)).to(scalar_dtype)

            edge_messages = src_linear.index_select(0, src)
            if edge_linear is not None:
                edge_messages = edge_messages + edge_linear

            aggregated_scalars = _sparse_mean(dst, edge_messages, num_nodes).to(scalar_dtype)

            if self.node_vector_out > 0:
                if self.vector_src is not None and norm_node_vectors.numel():
                    src_vectors = vector_linear(norm_node_vectors, self.vector_src).to(vector_dtype)
                else:
                    src_vectors = torch.zeros(
                        (num_nodes, self.node_vector_out, 3), dtype=vector_dtype, device=device
                    )

                if self.vector_edge is not None and norm_edge_vectors.numel():
                    edge_vectors_linear = vector_linear(norm_edge_vectors, self.vector_edge).to(vector_dtype)
                else:
                    edge_vectors_linear = torch.zeros(
                        (edge_vectors.size(0), self.node_vector_out, 3),
                        dtype=vector_dtype,
                        device=edge_vectors.device,
                    )

                edge_vector_messages = src_vectors.index_select(0, src)
                edge_vector_messages = edge_vector_messages + edge_vectors_linear
                aggregated_vector_flat = _sparse_mean(
                    dst, edge_vector_messages.reshape(edge_vector_messages.size(0), -1), num_nodes
                )
                aggregated_vectors = aggregated_vector_flat.view(num_nodes, self.node_vector_out, 3).to(vector_dtype)
            else:
                aggregated_vectors = torch.zeros((num_nodes, 0, 3), dtype=vector_dtype, device=device)

        scalar_update = self.scalar_mlp(
            aggregated_scalars.to(self.scalar_mlp[1].weight.dtype)
        ).to(scalar_dtype)
        scalar_update = self.scalar_dropout(scalar_update)

        if self.node_vector_out > 0:
            gate_input = aggregated_scalars
            if self.vector_gate is not None:
                gate = torch.sigmoid(self.vector_gate(gate_input.to(self.vector_gate.weight.dtype))).to(vector_dtype)
            else:
                gate = torch.ones((num_nodes, self.node_vector_out), dtype=vector_dtype, device=device)
            vector_update = apply_gating(aggregated_vectors, gate)
            vector_update = self.vector_dropout(vector_update)
        else:
            vector_update = aggregated_vectors

        updated = ScalarVector(scalar_update, vector_update).apply_mask(mask)
        return updated


class GCPMessagePassing(nn.Module):
    """Stack of four GCP modules with a bottleneck in the middle stages."""

    def __init__(
        self,
        node_scalar_dim: int,
        node_vector_dim: int,
        edge_scalar_dim: int,
        edge_vector_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        bottleneck_scalar = max(1, node_scalar_dim // 2)
        bottleneck_vector = max(1, node_vector_dim // 2) if node_vector_dim > 0 else 0

        self.blocks = nn.ModuleList(
            [
                GCPModule(
                    node_scalar_dim,
                    node_vector_dim,
                    bottleneck_scalar,
                    bottleneck_vector,
                    edge_scalar_dim,
                    edge_vector_dim,
                    dropout=dropout,
                ),
                GCPModule(
                    bottleneck_scalar,
                    bottleneck_vector,
                    bottleneck_scalar,
                    bottleneck_vector,
                    edge_scalar_dim,
                    edge_vector_dim,
                    dropout=dropout,
                ),
                GCPModule(
                    bottleneck_scalar,
                    bottleneck_vector,
                    bottleneck_scalar,
                    bottleneck_vector,
                    edge_scalar_dim,
                    edge_vector_dim,
                    dropout=dropout,
                ),
                GCPModule(
                    bottleneck_scalar,
                    bottleneck_vector,
                    node_scalar_dim,
                    node_vector_dim,
                    edge_scalar_dim,
                    edge_vector_dim,
                    dropout=dropout,
                ),
            ]
        )

    def forward(
        self,
        nodes: ScalarVector,
        edges: ScalarVector,
        edge_index: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> ScalarVector:
        features = nodes
        for block in self.blocks:
            features = block(features, edges, edge_index, mask=mask)
        return features


class GCPInteractions(nn.Module):
    """Full interaction block with message passing and feed-forward stack."""

    def __init__(
        self,
        node_scalar_dim: int,
        node_vector_dim: int,
        edge_scalar_dim: int,
        edge_vector_dim: int,
        *,
        skip_scalar_dim: int,
        skip_vector_dim: int,
        dropout: float = 0.0,
        prenorm: bool = True,
        position_head: bool = False,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm

        self.scalar_norm = nn.LayerNorm(node_scalar_dim) if prenorm else None
        self.vector_norm = VectorLayerNorm(node_vector_dim) if prenorm and node_vector_dim > 0 else None

        self.message_passing = GCPMessagePassing(
            node_scalar_dim,
            node_vector_dim,
            edge_scalar_dim,
            edge_vector_dim,
            dropout=dropout,
        )

        self.scalar_dropout = nn.Dropout(dropout)
        self.vector_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.ModuleList(
            [
                GCPModule(
                    node_scalar_dim + skip_scalar_dim,
                    node_vector_dim + skip_vector_dim,
                    node_scalar_dim,
                    node_vector_dim,
                    edge_scalar_dim,
                    edge_vector_dim,
                    dropout=dropout,
                ),
                GCPModule(
                    node_scalar_dim,
                    node_vector_dim,
                    node_scalar_dim,
                    node_vector_dim,
                    edge_scalar_dim,
                    edge_vector_dim,
                    dropout=dropout,
                ),
            ]
        )

        if position_head and node_vector_dim > 0:
            self.position_weight = nn.Parameter(torch.randn(1, node_vector_dim) * 0.02)
        else:
            self.register_parameter("position_weight", None)

    def forward(
        self,
        nodes: ScalarVector,
        skip: ScalarVector,
        edges: ScalarVector,
        edge_index: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tuple[ScalarVector, Optional[Tensor]]:
        features = nodes
        if self.prenorm:
            if self.scalar_norm is not None:
                scalar_norm = self.scalar_norm(features.scalars.to(self.scalar_norm.weight.dtype)).to(features.scalars.dtype)
            else:
                scalar_norm = features.scalars
            if self.vector_norm is not None and features.vector_channels > 0:
                vector_norm = self.vector_norm(features.vectors.to(self.vector_norm.weight.dtype)).to(features.vectors.dtype)
            else:
                vector_norm = features.vectors
            features = ScalarVector(scalar_norm, vector_norm)

        message = self.message_passing(features, edges, edge_index, mask=mask)
        message = message.dropout(self.scalar_dropout, self.vector_dropout)

        updated = nodes.add(message).apply_mask(mask)

        concat = ScalarVector.cat((updated, skip))
        ff = concat
        for block in self.feed_forward:
            ff = block(ff, edges, edge_index, mask=mask)
        ff = ff.dropout(self.scalar_dropout, self.vector_dropout)

        output = updated.add(ff).apply_mask(mask)

        positions: Optional[Tensor] = None
        if self.position_weight is not None and output.vector_channels > 0:
            positions = vector_linear(output.vectors, self.position_weight).squeeze(1)
            if mask is not None:
                positions = positions * mask.to(dtype=positions.dtype, device=positions.device).unsqueeze(-1)

        return output, positions


@dataclass
class GCPNetConfig:
    node_scalar_dim: int = 6
    node_vector_dim: int = 3
    edge_scalar_dim: int = 32
    edge_scalar_input_dim: Optional[int] = DEFAULT_EDGE_SCALAR_INPUT_DIM
    edge_vector_dim: int = 4
    edge_vector_input_dim: int = 1
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


class GCPNetEncoder(nn.Module):
    """Stack of GCP interaction layers with scalar/vector read-out."""

    def __init__(self, config: Optional[GCPNetConfig] = None) -> None:
        super().__init__()

        self.config = config or GCPNetConfig()

        if self.config.edge_scalar_input_dim is None:
            edge_scalar_in = self.config.edge_scalar_dim
        else:
            edge_scalar_in = self.config.edge_scalar_input_dim

        edge_vector_in = self.config.edge_vector_input_dim

        self.embedding = GCPEmbedding(
            self.config.node_scalar_dim,
            self.config.node_vector_dim,
            edge_scalar_in,
            edge_vector_in,
            node_scalar_out=self.config.hidden_scalar_dim,
            node_vector_out=self.config.hidden_vector_dim,
            edge_scalar_out=self.config.edge_scalar_dim,
            edge_vector_out=self.config.edge_vector_dim,
        )

        self.layers = nn.ModuleList()
        for idx in range(self.config.layers):
            position_head = bool(self.config.displacement_head and idx == self.config.layers - 1)
            self.layers.append(
                GCPInteractions(
                    self.config.hidden_scalar_dim,
                    self.config.hidden_vector_dim,
                    self.config.edge_scalar_dim,
                    self.config.edge_vector_dim,
                    skip_scalar_dim=self.config.hidden_scalar_dim,
                    skip_vector_dim=self.config.hidden_vector_dim,
                    dropout=self.config.dropout,
                    prenorm=self.config.prenorm,
                    position_head=position_head,
                )
            )

        readout_dim = self.config.hidden_scalar_dim + self.config.hidden_vector_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, self.config.latent_dim, bias=False),
        )

        if self.layers and isinstance(self.layers[-1], GCPInteractions):
            self.displacement_head = self.layers[-1].position_weight
        else:
            self.displacement_head = None

    def forward(
        self,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_index: Tensor,
        edge_scalars: Tensor,
        edge_vectors: Tensor,
        edge_frames: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        del edge_frames  # Edge frames are not used in the simplified encoder.

        node_features, edge_features = self.embedding(
            node_scalars,
            node_vectors,
            edge_scalars,
            edge_vectors,
            mask=mask,
        )

        features = node_features
        skip = node_features
        displacement: Optional[Tensor] = None

        for layer in self.layers:
            features, maybe_positions = layer(features, skip, edge_features, edge_index, mask=mask)
            if maybe_positions is not None:
                displacement = maybe_positions

        scalars = features.scalars
        vectors = features.vectors

        vec_norms = safe_norm(vectors, dim=-1)
        readout_input = torch.cat((scalars, vec_norms.to(scalars.dtype)), dim=-1)
        readout_dtype = self.readout[1].weight.dtype
        embeddings = self.readout(readout_input.to(readout_dtype)).to(scalars.dtype)

        output: Dict[str, Tensor] = {
            "embeddings": embeddings,
            "node_scalars": scalars,
            "node_vectors": vectors,
        }

        if displacement is not None:
            output["displacement"] = displacement

        return output


__all__ = [
    "GCPNetEncoder",
    "GCPNetConfig",
    "GCPInteractions",
    "GCPMessagePassing",
    "GCPEmbedding",
    "ScalarVector",
    "VectorLayerNorm",
    "DEFAULT_EDGE_SCALAR_INPUT_DIM",
]
