"""Feature construction for nodes, edges, and local frames."""

from __future__ import annotations

import dataclasses

import torch
from torch.nn import functional as F

from gcpvqvae.data.protein_io import ParsedProtein
from gcpvqvae.geometry.frames import build_edge_frames
from gcpvqvae.geometry.torsion import backbone_torsions


@dataclasses.dataclass(frozen=True)
class GraphFeatures:
    """A dataclass to hold all computed graph features."""
    # Node features
    node_scalars: torch.Tensor  # [L, h_in]
    node_vectors: torch.Tensor  # [L, chi_in, 3]

    # Edge features
    edge_index: torch.Tensor    # [2, E]
    edge_scalars: torch.Tensor  # [E, e_in]
    edge_vectors: torch.Tensor  # [E, xi_in, 3]

    # Edge frames
    edge_frames: torch.Tensor   # [E, 3, 3]


def _rbf(d, centers, sigma=2.0):
    """Radial basis function expansion of distances."""
    k = (-1 / (2 * sigma**2) * (d.unsqueeze(-1) - centers)**2).exp()
    return k


def _build_backbone_vectors(coords: torch.Tensor):
    """
    Computes the 6 backbone unit vectors (v1-v6) as defined in the workplan.
    v1 = N->CA, v2 = CA->C, v3 = C_i->N_{i+1}
    v4 = -(v1 x v2), v5 = v3 x v1, v6 = v2 x v3
    """
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]

    # Pad for C_i -> N_{i+1} vector
    c_pad = torch.cat([c[:-1], torch.zeros(1, 3, device=c.device)], dim=0)
    n_pad = torch.cat([n[1:], torch.zeros(1, 3, device=n.device)], dim=0)

    v1 = ca - n
    v2 = c - ca
    v3 = n_pad - c_pad

    v4 = -torch.cross(v1, v2, dim=-1)
    v5 = torch.cross(v3, v1, dim=-1)
    v6 = torch.cross(v2, v3, dim=-1)

    # Normalize all vectors
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    v3 = F.normalize(v3, dim=-1)
    v4 = F.normalize(v4, dim=-1)
    v5 = F.normalize(v5, dim=-1)
    v6 = F.normalize(v6, dim=-1)

    return torch.stack([v1, v2, v3, v4, v5, v6], dim=1)


def featurize_backbone(
    parsed_protein: ParsedProtein,
    k_neighbors: int = 16
) -> GraphFeatures:
    """
    Constructs all node and edge features for the GCPNet model.

    Args:
        parsed_protein: The parsed protein data.
        k_neighbors: The number of nearest neighbors for graph construction.

    Returns:
        A GraphFeatures object containing all necessary tensors.
    """
    coords = parsed_protein.coords
    mask = parsed_protein.mask
    L = coords.shape[0]

    # 1. Node scalar features (torsions)
    # h_in = 6
    node_scalars = backbone_torsions(coords, mask)

    # 2. Node vector features
    # chi_in = 3
    all_vectors = _build_backbone_vectors(coords)
    # Select v1, v2, v4 as per workplan
    node_vectors = torch.stack([all_vectors[:, 0], all_vectors[:, 1], all_vectors[:, 3]], dim=1)

    # 3. Graph construction (kNN + sequence)
    ca_coords = coords[:, 1]

    # Pairwise distances
    dist_matrix = torch.cdist(ca_coords, ca_coords)

    # k-Nearest Neighbors (excluding self)
    k = min(k_neighbors, L - 1)
    if k > 0:
        _, knn_indices = torch.topk(dist_matrix, k=k + 1, largest=False)
        knn_indices = knn_indices[:, 1:] # remove self

    # Sequence edges
    seq_edges = torch.stack([torch.arange(L-1), torch.arange(1, L)], dim=0)

    # Combine edges and make them symmetric
    edge_index = torch.cat([seq_edges, seq_edges.flip(0)], dim=1)
    if k > 0:
        knn_row = torch.arange(L, device=coords.device).unsqueeze(-1).repeat(1, k).flatten()
        knn_col = knn_indices.flatten()
        knn_edges = torch.stack([knn_row, knn_col], dim=0)
        edge_index = torch.cat([edge_index, knn_edges], dim=1)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    i_nodes, j_nodes = edge_index

    # 4. Edge scalar features
    # e_in = 8
    rbf_centers = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], device=coords.device, dtype=torch.float32)
    edge_distances = dist_matrix[i_nodes, j_nodes]
    edge_scalars = _rbf(edge_distances, rbf_centers)

    # 5. Edge vector features
    # xi_in = 1
    r_ij = ca_coords[j_nodes] - ca_coords[i_nodes]
    edge_vectors = r_ij.unsqueeze(1) # Add channel dimension

    # 6. Edge frames
    edge_frames = build_edge_frames(edge_index, coords)

    return GraphFeatures(
        node_scalars=node_scalars,
        node_vectors=node_vectors,
        edge_index=edge_index,
        edge_scalars=edge_scalars,
        edge_vectors=edge_vectors,
        edge_frames=edge_frames,
    )
