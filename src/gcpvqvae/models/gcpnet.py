"""Graph Convolutional Point (GCP) encoder used by GCP-VQVAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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


@dataclass
class GCPNetConfig:
    node_scalar_dim: int = 6
    node_vector_dim: int = 3
    edge_scalar_dim: int = 8
    edge_scalar_input_dim: Optional[int] = DEFAULT_EDGE_SCALAR_INPUT_DIM
    edge_vector_dim: int = 1
    hidden_scalar_dim: int = 128
    hidden_vector_dim: int = 16
    latent_dim: int = 256
    layers: int = 6
    dropout: float = 0.0
    displacement_head: bool = False
    init: str = "random"
    init_checkpoint: Optional[str] = None
    strict_init: bool = True


class GCPNetEncoder(nn.Module):
    """Stack of GCP convolutional layers with scalar/vector read-out."""

    def __init__(self, config: Optional[GCPNetConfig] = None) -> None:
        super().__init__()

        self.config = config or GCPNetConfig()

        if self.config.edge_scalar_input_dim is None:
            self.edge_scalar_in_dim = self.config.edge_scalar_dim
        else:
            self.edge_scalar_in_dim = self.config.edge_scalar_input_dim

        self.scalar_proj = nn.Linear(self.config.node_scalar_dim, self.config.hidden_scalar_dim, bias=False)
        self.vector_proj = nn.Parameter(
            torch.randn(self.config.hidden_vector_dim, self.config.node_vector_dim) * 0.02
        )

        if self.config.edge_scalar_dim > 0:
            self.edge_scalar_proj = nn.Linear(
                self.edge_scalar_in_dim,
                self.config.edge_scalar_dim,
                bias=False,
            )
            if self.edge_scalar_in_dim == self.config.edge_scalar_dim:
                with torch.no_grad():
                    self.edge_scalar_proj.weight.copy_(torch.eye(self.config.edge_scalar_dim))
        else:
            self.edge_scalar_proj = None

        self.layers = nn.ModuleList(
            [
                GCPConv(
                    self.config.hidden_scalar_dim,
                    self.config.hidden_vector_dim,
                    edge_scalar_dim=self.config.edge_scalar_dim,
                    edge_vector_channels=self.config.edge_vector_dim,
                    hidden_scalar_dim=self.config.hidden_scalar_dim * 2,
                    hidden_vector_channels=self.config.hidden_vector_dim,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.layers)
            ]
        )

        readout_dim = self.config.hidden_scalar_dim + self.config.hidden_vector_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, self.config.latent_dim, bias=False),
        )

        if self.config.displacement_head:
            self.displacement_head = nn.Sequential(
                nn.LayerNorm(self.config.hidden_scalar_dim),
                nn.Linear(self.config.hidden_scalar_dim, self.config.hidden_scalar_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.config.hidden_scalar_dim, 3, bias=False),
            )
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
        if node_scalars.ndim != 2:
            raise ValueError("node_scalars must have shape (N, F)")
        if node_vectors.ndim != 3:
            raise ValueError("node_vectors must have shape (N, C, 3)")

        scalar_proj_dtype = self.scalar_proj.weight.dtype
        scalars = self.scalar_proj(node_scalars.to(scalar_proj_dtype)).to(node_scalars.dtype)
        vectors = vector_linear(node_vectors, self.vector_proj).to(node_vectors.dtype)

        if mask is not None:
            mask = mask.to(torch.bool)
            scalars = scalars * mask.unsqueeze(-1)
            vectors = vectors * mask.unsqueeze(-1).unsqueeze(-1)

        if self.edge_scalar_proj is not None and edge_scalars.numel():
            proj_dtype = self.edge_scalar_proj.weight.dtype
            edge_scalars_projected = self.edge_scalar_proj(edge_scalars.to(proj_dtype)).to(edge_scalars.dtype)
        elif self.edge_scalar_proj is not None:
            edge_scalars_projected = edge_scalars.new_zeros(
                (edge_scalars.shape[0], self.config.edge_scalar_dim)
            )
        else:
            edge_scalars_projected = edge_scalars

        for layer in self.layers:
            scalars, vectors = layer(
                scalars,
                vectors,
                edge_index,
                edge_scalars_projected,
                edge_vectors,
                edge_frames,
            )
            if mask is not None:
                scalars = scalars * mask.unsqueeze(-1)
                vectors = vectors * mask.unsqueeze(-1).unsqueeze(-1)

        vec_norms = safe_norm(vectors, dim=-1)
        readout_input = torch.cat((scalars, vec_norms.to(scalars.dtype)), dim=-1)
        readout_dtype = self.readout[1].weight.dtype
        embeddings = self.readout(readout_input.to(readout_dtype)).to(scalars.dtype)

        output: Dict[str, Tensor] = {
            "embeddings": embeddings,
            "node_scalars": scalars,
            "node_vectors": vectors,
        }

        if self.displacement_head is not None:
            disp_input_dtype = self.displacement_head[1].weight.dtype
            displacement = self.displacement_head(scalars.to(disp_input_dtype)).to(scalars.dtype)
            if mask is not None:
                displacement = displacement * mask.unsqueeze(-1)
            output["displacement"] = displacement

        return output


__all__ = [
    "GCPNetEncoder",
    "GCPNetConfig",
    "GCPConv",
    "VectorLayerNorm",
    "DEFAULT_EDGE_SCALAR_INPUT_DIM",
]
