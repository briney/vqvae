"""Geometry utilities used for batched graph preprocessing."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch

try:  # pragma: no cover - torch_scatter is optional at runtime
    from torch_scatter import scatter_sum
except Exception:  # pragma: no cover - provide a functional fallback

    def scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("Fallback scatter_sum only supports dim=0")
        if dim_size is None:
            dim_size = int(index.max().item() + 1) if index.numel() else 0
        out_shape = (dim_size,) + src.shape[1:]
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        if index.numel():
            out.index_add_(0, index, src)
        return out

from gcpvqvae.geometry.frames import build_local_frames

Tensor = torch.Tensor

_IDENTITY_CACHE = {}


def _scatter_count(index: Tensor, *, dim_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Count occurrences for each index value.

    Args:
        index: Tensor of shape ``(N,)`` with graph identifiers.
        dim_size: Total number of graphs.
        device: Target device for the counts tensor.
        dtype: Floating-point dtype used for the counts.

    Returns:
        Tensor of shape ``(dim_size, 1)`` containing counts per graph, clamped
        to ``>= 1`` to avoid division by zero.
    """
    ones = torch.ones((index.shape[0], 1), dtype=dtype, device=device)
    counts = scatter_sum(ones, index, dim=0, dim_size=dim_size)
    return torch.clamp(counts, min=1.0)


def centralize(
    positions: Tensor,
    batch: Tensor,
    *,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return centralised coordinates and per-graph centroids.

    Args:
        positions: Tensor of shape ``(N, 3)`` containing node coordinates.
        batch: Tensor of shape ``(N,)`` assigning each node to a graph.
        mask: Optional boolean tensor ``(N,)`` selecting nodes used for centroid
            computation.

    Returns:
        Tuple ``(centered, centroids)`` where ``centered`` has shape ``(N, 3)``
        and ``centroids`` has shape ``(G, 3)``.

    Raises:
        ValueError: If tensor shapes are incompatible.
    """

    if positions.ndim != 2 or positions.size(-1) != 3:
        raise ValueError("positions must have shape (N, 3)")

    if positions.shape[0] != batch.shape[0]:
        raise ValueError("positions and batch must align in the first dimension")

    device = positions.device
    dtype = positions.dtype
    num_graphs = int(batch.max().item() + 1) if batch.numel() else 0

    weights: Tensor
    weighted_pos: Tensor
    if mask is not None:
        mask = mask.to(device=device, dtype=dtype).unsqueeze(-1)
        weights = scatter_sum(mask, batch, dim=0, dim_size=num_graphs)
        weighted_pos = scatter_sum(positions * mask, batch, dim=0, dim_size=num_graphs)
    else:
        ones = torch.ones((positions.shape[0], 1), dtype=dtype, device=device)
        weights = scatter_sum(ones, batch, dim=0, dim_size=num_graphs)
        weighted_pos = scatter_sum(positions, batch, dim=0, dim_size=num_graphs)

    weights = torch.clamp(weights, min=1.0)
    centroids = weighted_pos / weights
    centered = positions - centroids.index_select(0, batch)
    return centered, centroids


def decentralize(positions: Tensor, centroids: Tensor, batch: Tensor) -> Tensor:
    """Undo :func:`centralize` for a subset of nodes.

    Args:
        positions: Tensor of shape ``(N, 3)`` containing centred coordinates.
        centroids: Tensor of shape ``(G, 3)`` storing per-graph centroids.
        batch: Tensor assigning each node to a centroid.

    Returns:
        Tensor of shape ``(N, 3)`` with coordinates translated back.

    Raises:
        ValueError: If tensor shapes are incompatible.
    """

    if positions.ndim != 2 or positions.size(-1) != 3:
        raise ValueError("positions must have shape (N, 3)")
    if centroids.ndim != 2 or centroids.size(-1) != 3:
        raise ValueError("centroids must have shape (G, 3)")

    return positions + centroids.index_select(0, batch)


def localize(vectors: Tensor, frames: Tensor) -> Tensor:
    """Express ``vectors`` in the coordinate system defined by ``frames``.

    Args:
        vectors: Tensor of shape ``(E, 3)`` or ``(E, C, 3)`` containing vectors
            to transform.
        frames: Tensor of shape ``(E, 3, 3)`` containing rotation matrices.

    Returns:
        Tensor matching the shape of ``vectors`` with coordinates expressed in
        the local frames.

    Raises:
        ValueError: If the input shapes are incompatible.
    """

    if frames.ndim != 3 or frames.shape[-2:] != (3, 3):
        raise ValueError("frames must have shape (E, 3, 3)")

    if vectors.size(0) != frames.size(0):
        raise ValueError("vectors and frames must agree on the first dimension")

    if vectors.ndim == 2:
        return torch.einsum("eji,ej->ei", frames, vectors)
    if vectors.ndim == 3:
        return torch.einsum("eji,eci->ecj", frames, vectors)
    raise ValueError("vectors must have shape (E, 3) or (E, C, 3)")


def _cached_identity(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return cached batches of identity matrices to avoid reallocations.

    Args:
        count: Number of frames requested.
        device: Device on which tensors should reside.
        dtype: Floating-point dtype for the identity matrices.

    Returns:
        Tensor of shape ``(count, 3, 3)`` containing batched identity matrices.
    """
    key = (device, dtype)
    cached = _IDENTITY_CACHE.get(key)
    if cached is None or cached.shape[0] < count:
        identity = torch.eye(3, device=device, dtype=dtype)
        cached = identity.unsqueeze(0).repeat(max(count, 1), 1, 1)
        _IDENTITY_CACHE[key] = cached
    return cached[:count]


def ensure_edge_frames(
    positions: Tensor,
    edge_index: Tensor,
    *,
    edge_batch: Optional[Tensor] = None,
    node_batch: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Return rotation frames for each edge, computing them lazily if needed.

    Args:
        positions: Tensor of shape ``(N, 3)`` containing node coordinates.
        edge_index: Tensor of shape ``(2, E)`` describing directed edges.
        edge_batch: Optional tensor ``(E,)`` mapping edges to graphs.
        node_batch: Optional tensor ``(N,)`` mapping nodes to graphs. Required
            when ``positions`` is empty.
        mask: Optional boolean tensor ``(N,)`` indicating valid nodes.

    Returns:
        Tensor of shape ``(E, 3, 3)`` containing rotation frames.

    Raises:
        ValueError: If inputs have incompatible shapes.
    """

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape (2, E)")

    num_edges = edge_index.size(1)
    if num_edges == 0:
        return _cached_identity(0, positions.device, positions.dtype)

    if node_batch is None:
        if positions.shape[0] == 0:
            raise ValueError("node_batch is required when positions are empty")
        node_batch = positions.new_zeros((positions.shape[0],), dtype=torch.long)

    if edge_batch is None:
        src, _ = edge_index
        edge_batch = node_batch.index_select(0, src)

    num_graphs = int(node_batch.max().item() + 1) if node_batch.numel() else 0
    frames = positions.new_empty((num_edges, 3, 3))

    for graph in range(num_graphs):
        node_mask = node_batch == graph
        edge_mask = edge_batch == graph
        if edge_mask.sum().item() == 0:
            continue

        offset = int(node_mask.nonzero(as_tuple=False)[0].item()) if node_mask.any() else 0
        local_nodes = positions[node_mask]
        local_edges = edge_index[:, edge_mask] - offset
        local_mask = mask[node_mask] if mask is not None else None

        if local_nodes.size(0) == 0:
            frames[edge_mask] = _cached_identity(int(edge_mask.sum()), positions.device, positions.dtype)
            continue

        local_frames = build_local_frames(local_nodes, local_edges, mask=local_mask)
        frames[edge_mask] = local_frames.to(device=positions.device, dtype=positions.dtype)

    return frames


__all__ = ["centralize", "decentralize", "localize", "ensure_edge_frames"]
