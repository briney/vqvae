"""Lightweight batch container mirroring the :mod:`torch_geometric` API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import torch

try:  # pragma: no cover - torch_geometric is optional in the test env
    from torch_geometric.data import Batch as _PyGBatch  # type: ignore
except Exception:  # pragma: no cover - fallback to a simple object base class
    _PyGBatch = object  # type: ignore[misc, assignment]

Tensor = torch.Tensor


@dataclass
class EdgeStorage:
    """Container holding per-relation edge features."""

    edge_index: Tensor
    scalars: Tensor
    vectors: Tensor
    frames: Optional[Tensor] = None
    batch: Optional[Tensor] = None
    name: str = "knn_k"

    def clone(self) -> "EdgeStorage":
        """Return a detached copy of the edge storage."""
        frames = None if self.frames is None else self.frames.clone()
        batch = None if self.batch is None else self.batch.clone()
        return EdgeStorage(
            edge_index=self.edge_index.clone(),
            scalars=self.scalars.clone(),
            vectors=self.vectors.clone(),
            frames=frames,
            batch=batch,
            name=self.name,
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "EdgeStorage":
        """Move the edge storage tensors to a target device/dtype.

        Args:
            device: Torch device to move tensors to.
            dtype: Optional floating-point dtype applied to scalar/vector fields.

        Returns:
            New :class:`EdgeStorage` residing on the requested device/dtype.
        """
        kwargs = {"device": device}
        if dtype is not None and self.scalars.dtype.is_floating_point:
            kwargs["dtype"] = dtype

        scalars = self.scalars.to(**kwargs)
        vectors = self.vectors.to(**kwargs)
        frames = None if self.frames is None else self.frames.to(**kwargs)

        batch_tensor = self.batch
        if batch_tensor is not None:
            batch_tensor = batch_tensor.to(device=device)

        return EdgeStorage(
            edge_index=self.edge_index.to(device=device),
            scalars=scalars,
            vectors=vectors,
            frames=frames,
            batch=batch_tensor,
            name=self.name,
        )


class ProteinBatch(_PyGBatch):
    """Minimal batch object understood by :class:`GCPNetEncoder`."""

    def __init__(
        self,
        *,
        h: Tensor,
        chi: Tensor,
        e: Mapping[str, EdgeStorage] | EdgeStorage,
        xi: Tensor,
        batch: Tensor,
        ptr: Tensor,
        mask: Optional[Tensor] = None,
        **extras: Tensor,
    ) -> None:
        """Initialise a batched graph compatible with GCPNet.

        Args:
            h: Node scalar features of shape ``(N, F_h)``.
            chi: Node vector features of shape ``(N, 3, 3)``.
            e: Mapping of edge storages or a single :class:`EdgeStorage`.
            xi: Node coordinate tensor of shape ``(N, 3)``.
            batch: Batch assignment per node matching PyG semantics.
            ptr: Pointer offsets delimiting individual graphs.
            mask: Optional boolean mask for valid nodes.
            **extras: Additional tensors copied onto the batch instance.
        """
        super().__init__()  # type: ignore[misc]
        self.h = h
        self.chi = chi
        if isinstance(e, EdgeStorage):
            self.e: MutableMapping[str, EdgeStorage] = {e.name: e}
        else:
            self.e = dict(e)
        self.xi = xi
        self.batch = batch
        self.ptr = ptr
        self.mask = mask
        self.centroids: Optional[Tensor] = None
        self.edge_frames: Optional[Tensor] = None
        for key, value in extras.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------ helpers
    def num_graphs(self) -> int:
        """Return the number of graphs represented by the batch."""
        return int(self.ptr.numel() - 1)

    def __len__(self) -> int:  # pragma: no cover - compatibility shim
        """Return the number of nodes in the batch."""
        return self.h.shape[0]

    def items(self) -> Iterable[tuple[str, Tensor]]:  # pragma: no cover - shim
        """Iterate over tensor-valued attributes stored on the batch."""
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                yield key, value

    def clone(self) -> "ProteinBatch":
        """Return a detached copy of the batch and its edge storages."""
        edges = {name: storage.clone() for name, storage in self.e.items()}
        mask = None if self.mask is None else self.mask.clone()
        extras: Dict[str, Tensor] = {}
        for key, value in self.__dict__.items():
            if key in {"h", "chi", "e", "xi", "batch", "ptr", "mask", "centroids", "edge_frames"}:
                continue
            if isinstance(value, Tensor):
                extras[key] = value.clone()
        clone = ProteinBatch(
            h=self.h.clone(),
            chi=self.chi.clone(),
            e=edges,
            xi=self.xi.clone(),
            batch=self.batch.clone(),
            ptr=self.ptr.clone(),
            mask=mask,
            **extras,
        )
        if self.centroids is not None:
            clone.centroids = self.centroids.clone()
        if self.edge_frames is not None:
            clone.edge_frames = self.edge_frames.clone()
        return clone

    def to(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ProteinBatch":
        """Move batch tensors to the requested device and dtype.

        Args:
            device: Destination device for tensor data.
            dtype: Optional floating-point dtype applied to relevant tensors.

        Returns:
            New :class:`ProteinBatch` instance residing on the target device.
        """
        kwargs = {"device": device}
        if dtype is not None and self.h.dtype.is_floating_point:
            kwargs["dtype"] = dtype

        h = self.h.to(**kwargs)
        chi = self.chi.to(**kwargs)
        xi = self.xi.to(**kwargs)
        mask = None if self.mask is None else self.mask.to(device=device)
        batch = self.batch.to(device=device)
        ptr = self.ptr.to(device=device)

        edges = {name: storage.to(device=device, dtype=dtype) for name, storage in self.e.items()}

        extras: Dict[str, Tensor] = {}
        for key, value in self.__dict__.items():
            if key in {"h", "chi", "e", "xi", "batch", "ptr", "mask", "centroids", "edge_frames"}:
                continue
            if isinstance(value, Tensor):
                if dtype is not None and value.dtype.is_floating_point:
                    extras[key] = value.to(device=device, dtype=dtype)
                else:
                    extras[key] = value.to(device=device)

        out = ProteinBatch(
            h=h,
            chi=chi,
            e=edges,
            xi=xi,
            batch=batch,
            ptr=ptr,
            mask=mask,
            **extras,
        )

        centroids = getattr(self, "centroids", None)
        if centroids is not None:
            out.centroids = centroids.to(**kwargs)
        edge_frames = getattr(self, "edge_frames", None)
        if edge_frames is not None:
            out.edge_frames = edge_frames.to(**kwargs)

        return out


def protein_batch_from_graph_dict(
    batch: Mapping[str, Tensor],
    *,
    relation_name: str = "knn_k",
) -> ProteinBatch:
    """Construct a :class:`ProteinBatch` from a collated mini-batch.

    Args:
        batch: Dictionary emitted by :func:`gcpvqvae.data.dataset.collate_backbones`.
            Required keys include ``node_scalars`` with shape ``(B, L, F)``,
            ``node_vectors`` with shape ``(B, L, 3, 3)``, ``coords`` with shape
            ``(B, L, 3, 3)``, and ``edge_index``.
        relation_name: Edge relation label used to store the resulting
            :class:`EdgeStorage`.

    Returns:
        :class:`ProteinBatch` where node tensors have been flattened to ``(N, …)``
        using the validity mask and PyG bookkeeping tensors are derived from the
        masked lengths.

    Raises:
        KeyError: If required tensors are missing from ``batch``.

    Examples:
        >>> collated = collate_backbones([dataset[0], dataset[1]])
        >>> protein_batch = protein_batch_from_graph_dict(collated)
        >>> protein_batch.h.shape
        torch.Size([int(protein_batch.lengths.sum()), collated["node_scalars"].shape[-1]])
    """

    if "node_scalars" not in batch or "node_vectors" not in batch:
        raise KeyError("Batch dictionary must contain node features")

    node_scalars = batch["node_scalars"]
    node_vectors = batch["node_vectors"]
    coords = batch["coords"]
    mask = batch["mask"].to(torch.bool)
    if "nan_mask" in batch and isinstance(batch["nan_mask"], Tensor):
        mask = mask & ~batch["nan_mask"].to(torch.bool)

    batch_size, max_len = node_scalars.shape[:2]
    flat_nodes = batch_size * max_len

    flat_h = node_scalars.reshape(flat_nodes, -1)
    flat_chi = node_vectors.reshape(flat_nodes, node_vectors.shape[-2], node_vectors.shape[-1])
    flat_xi = coords[:, :, 1, :].reshape(flat_nodes, 3)
    flat_mask = mask.reshape(-1)

    valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    h = flat_h.index_select(0, valid_indices)
    chi = flat_chi.index_select(0, valid_indices)
    xi = flat_xi.index_select(0, valid_indices)

    # ``lengths`` in the collated batch reflects the padded sequence length.  GCPNet
    # operates only on the valid residues (where ``mask`` is ``True``), so we need
    # to build the PyG bookkeeping tensors from the masked counts instead of the
    # padded lengths.  Otherwise ``batch`` would report more nodes than there are
    # coordinate entries which leads to shape mismatches when centralising the
    # node positions during message passing.
    valid_lengths = mask.to(dtype=torch.long).sum(dim=1)
    node_batch = torch.repeat_interleave(
        torch.arange(batch_size, device=valid_lengths.device, dtype=torch.long),
        valid_lengths,
    )
    ptr = torch.zeros((batch_size + 1,), dtype=torch.long, device=valid_lengths.device)
    ptr[1:] = torch.cumsum(valid_lengths, dim=0)

    edge_index = batch["edge_index"]
    edge_scalars = batch["edge_scalars"]
    edge_vectors = batch["edge_vectors"]
    edge_frames = batch.get("edge_frames")
    edge_batch = batch.get("edge_batch")

    if edge_index.numel():
        device = edge_index.device
        index_map = torch.full(
            (flat_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        index_map[valid_indices] = torch.arange(
            valid_indices.numel(), device=device, dtype=torch.long
        )

        mapped_src = index_map.index_select(0, edge_index[0])
        mapped_dst = index_map.index_select(0, edge_index[1])
        edge_mask = (mapped_src >= 0) & (mapped_dst >= 0)

        mapped_src = mapped_src[edge_mask]
        mapped_dst = mapped_dst[edge_mask]

        edge_index = torch.stack((mapped_src, mapped_dst), dim=0)

        if edge_scalars.numel():
            edge_scalars = edge_scalars[edge_mask]
        if edge_vectors.numel():
            edge_vectors = edge_vectors[edge_mask]
        if edge_frames is not None and edge_frames.numel():
            edge_frames = edge_frames[edge_mask]
        if edge_batch is not None and edge_batch.numel():
            edge_batch = edge_batch[edge_mask]
    else:
        # Ensure tensors remain empty with consistent shapes when there are no edges.
        edge_scalars = edge_scalars[:0]
        edge_vectors = edge_vectors[:0]
        if edge_frames is not None:
            edge_frames = edge_frames[:0]
        if edge_batch is not None:
            edge_batch = edge_batch[:0]

    storage = EdgeStorage(
        edge_index=edge_index,
        scalars=edge_scalars,
        vectors=edge_vectors,
        frames=edge_frames,
        batch=edge_batch,
        name=relation_name,
    )

    extras: Dict[str, Tensor] = {}
    for key in ("coords", "atom_mask", "backbone_vectors", "torsion_angles", "rotations", "translations"):
        if key in batch and isinstance(batch[key], Tensor):
            extras[key] = batch[key]

    protein_batch = ProteinBatch(
        h=h,
        chi=chi,
        e={relation_name: storage},
        xi=xi,
        batch=node_batch,
        ptr=ptr,
        mask=torch.ones_like(node_batch, dtype=torch.bool),
        **extras,
    )
    protein_batch.lengths = valid_lengths
    protein_batch.original_lengths = batch["lengths"]
    protein_batch.valid_indices = valid_indices
    protein_batch.full_mask = mask
    protein_batch.batch_size = batch_size
    protein_batch.max_length = max_len
    if "metadata" in batch:
        protein_batch.metadata = batch["metadata"]  # type: ignore[attr-defined]
    if "sequences" in batch:
        protein_batch.sequences = batch["sequences"]  # type: ignore[attr-defined]
    return protein_batch


__all__ = ["EdgeStorage", "ProteinBatch", "protein_batch_from_graph_dict"]
