"""Feature construction for nodes, edges, and local frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from gcpvqvae.geometry.frames import build_local_frames
from gcpvqvae.geometry.torsion import backbone_torsions

from .mmcif import BackboneRecord

Tensor = torch.Tensor

RBF_CENTRES = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
RBF_SIGMA = 2.0


def _safe_normalise(vec: Tensor, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return vec / norm


def _rbf(distances: Tensor, centres: Tensor, sigma: float) -> Tensor:
    diff = distances.unsqueeze(-1) - centres
    return torch.exp(-0.5 * (diff / sigma) ** 2)


@dataclass
class NodeFeatures:
    scalars: Tensor  # (L, 6)
    vectors: Tensor  # (L, 3, 3)
    backbone_vectors: Tensor  # (L, 6, 3)
    torsions: Tensor  # (L, 3)


@dataclass
class EdgeFeatures:
    edge_index: Tensor  # (2, E)
    scalars: Tensor  # (E, 8)
    vectors: Tensor  # (E, 3)
    frames: Tensor  # (E, 3, 3)


def _backbone_vectors(coords: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    n = coords[:, 0, :]
    ca = coords[:, 1, :]
    c = coords[:, 2, :]

    L = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    valid = mask.to(torch.bool)

    v1 = torch.zeros((L, 3), device=device, dtype=dtype)
    v2 = torch.zeros_like(v1)
    v3 = torch.zeros_like(v1)

    valid_v1 = valid
    valid_v2 = valid
    valid_v3 = torch.zeros_like(valid)
    valid_v3[:-1] = valid[:-1] & valid[1:]

    if valid_v1.any():
        v1[valid_v1] = ca[valid_v1] - n[valid_v1]
        v1[valid_v1] = _safe_normalise(v1[valid_v1])
    if valid_v2.any():
        v2[valid_v2] = c[valid_v2] - ca[valid_v2]
        v2[valid_v2] = _safe_normalise(v2[valid_v2])
    if valid_v3.any():
        v3[:-1][valid_v3[:-1]] = n[1:][valid_v3[:-1]] - c[:-1][valid_v3[:-1]]
        v3[:-1][valid_v3[:-1]] = _safe_normalise(v3[:-1][valid_v3[:-1]])

    v4 = torch.zeros_like(v1)
    v5 = torch.zeros_like(v1)
    v6 = torch.zeros_like(v1)

    valid_cross = valid_v1 & valid_v2
    if valid_cross.any():
        v4[valid_cross] = -torch.cross(v1[valid_cross], v2[valid_cross], dim=-1)
        v4[valid_cross] = _safe_normalise(v4[valid_cross])

    valid_v5 = valid_v3 & valid_v1
    if valid_v5.any():
        v5[valid_v5] = torch.cross(v3[valid_v5], v1[valid_v5], dim=-1)
        v5[valid_v5] = _safe_normalise(v5[valid_v5])

    valid_v6 = valid_v2 & valid_v3
    if valid_v6.any():
        v6[valid_v6] = torch.cross(v2[valid_v6], v3[valid_v6], dim=-1)
        v6[valid_v6] = _safe_normalise(v6[valid_v6])

    stack = torch.stack((v1, v2, v3, v4, v5, v6), dim=1)
    node_vectors = torch.stack((v1, v2, v4), dim=1)
    return stack, node_vectors


def build_node_features(backbone: BackboneRecord) -> NodeFeatures:
    coords = backbone.coords
    mask = backbone.mask

    torsion_dict = backbone_torsions(coords)
    phi = torsion_dict["phi"]
    psi = torsion_dict["psi"]
    omega = torsion_dict["omega"]

    L = coords.shape[0]
    dtype = coords.dtype
    device = coords.device

    sin_phi = torch.zeros((L,), dtype=dtype, device=device)
    cos_phi = torch.zeros_like(sin_phi)
    sin_psi = torch.zeros_like(sin_phi)
    cos_psi = torch.zeros_like(sin_phi)
    sin_omega = torch.zeros_like(sin_phi)
    cos_omega = torch.zeros_like(sin_phi)

    valid_phi = torch.zeros((L,), dtype=torch.bool, device=device)
    valid_phi[1:] = mask[:-1] & mask[1:]

    valid_psi = torch.zeros_like(valid_phi)
    valid_psi[:-1] = mask[:-1] & mask[1:]

    valid_omega = valid_psi.clone()

    sin_phi[valid_phi] = torch.sin(phi[valid_phi])
    cos_phi[valid_phi] = torch.cos(phi[valid_phi])
    sin_psi[valid_psi] = torch.sin(psi[valid_psi])
    cos_psi[valid_psi] = torch.cos(psi[valid_psi])
    sin_omega[valid_omega] = torch.sin(omega[valid_omega])
    cos_omega[valid_omega] = torch.cos(omega[valid_omega])

    scalars = torch.stack(
        (sin_phi, cos_phi, sin_psi, cos_psi, sin_omega, cos_omega),
        dim=-1,
    )

    backbone_vectors, node_vectors = _backbone_vectors(coords, mask)
    torsions = torch.stack((phi, psi, omega), dim=-1)

    return NodeFeatures(
        scalars=scalars,
        vectors=node_vectors,
        backbone_vectors=backbone_vectors,
        torsions=torsions,
    )


def _sequence_edges(mask: Tensor) -> Tensor:
    if mask.numel() < 2:
        return torch.empty((2, 0), dtype=torch.long)

    valid = mask.to(torch.bool)
    src = torch.arange(mask.numel() - 1, dtype=torch.long)
    dst = src + 1
    valid_pairs = valid[:-1] & valid[1:]
    src = src[valid_pairs]
    dst = dst[valid_pairs]

    edges = torch.stack((src, dst), dim=0)
    rev = torch.stack((dst, src), dim=0)
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat((edges, rev), dim=1)


def _knn_edges(ca: Tensor, mask: Tensor, k: int) -> Tensor:
    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=ca.device)

    positions = ca[valid_idx]
    distances = torch.cdist(positions, positions, p=2)

    k = min(k, max(positions.shape[0] - 1, 0))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=ca.device)

    topk = torch.topk(-distances, k=k + 1, dim=-1).indices

    src_list = []
    dst_list = []
    for row, idx in enumerate(valid_idx):
        neighbours = topk[row]
        neighbours = neighbours[neighbours != row][:k]
        if neighbours.numel() == 0:
            continue
        src_nodes = idx.repeat(neighbours.numel())
        dst_nodes = valid_idx[neighbours]
        src_list.append(src_nodes)
        dst_list.append(dst_nodes)

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long, device=ca.device)

    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    edges = torch.stack((src, dst), dim=0)
    rev = torch.stack((dst, src), dim=0)
    return torch.cat((edges, rev), dim=1)


def build_edge_features(
    backbone: BackboneRecord,
    *,
    k: int = 16,
) -> EdgeFeatures:
    ca = backbone.coords[:, 1, :]
    mask = backbone.atom_mask[:, 1]

    knn = _knn_edges(ca, mask, k)
    seq_edges = _sequence_edges(mask)

    edge_index = torch.cat((knn, seq_edges), dim=1) if knn.numel() else seq_edges
    if edge_index.numel():
        edge_index = edge_index.unique(dim=1)

    if edge_index.numel() == 0:
        return EdgeFeatures(
            edge_index=edge_index,
            scalars=torch.empty((0, 8), dtype=ca.dtype),
            vectors=torch.empty((0, 3), dtype=ca.dtype),
            frames=torch.empty((0, 3, 3), dtype=ca.dtype),
        )

    src, dst = edge_index
    disp = ca[dst] - ca[src]
    distances = torch.linalg.norm(disp, dim=-1)
    centres = RBF_CENTRES.to(device=ca.device, dtype=ca.dtype)
    scalars = _rbf(distances, centres, RBF_SIGMA)
    vectors = disp
    frames = build_local_frames(ca, edge_index, mask=mask)

    return EdgeFeatures(
        edge_index=edge_index,
        scalars=scalars,
        vectors=vectors,
        frames=frames,
    )


def featurize_backbone(backbone: BackboneRecord, *, k: int = 16) -> Dict[str, Tensor]:
    node = build_node_features(backbone)
    edge = build_edge_features(backbone, k=k)

    return {
        "node_scalars": node.scalars,
        "node_vectors": node.vectors,
        "backbone_vectors": node.backbone_vectors,
        "torsion_angles": node.torsions,
        "edge_index": edge.edge_index,
        "edge_scalars": edge.scalars,
        "edge_vectors": edge.vectors,
        "edge_frames": edge.frames,
    }


__all__ = [
    "NodeFeatures",
    "EdgeFeatures",
    "build_node_features",
    "build_edge_features",
    "featurize_backbone",
]
