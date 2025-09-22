"""Torch dataset wrappers for protein backbones and tokens."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from gcpvqvae.data.featurize import GraphFeatures, featurize_backbone
from gcpvqvae.data.protein_io import ParsedProtein, load_protein_file


class BackboneDataset(Dataset):
    """
    A PyTorch Dataset for loading and featurizing protein backbones from mmCIF files.

    The dataset scans a directory for `.cif` files, expecting a format like
    `{name}_{chain}.cif` (e.g., `1abc_A.cif`). It loads, parses, and featurizes
    each structure on the fly. Featurized results are cached in memory to
    speed up subsequent epochs.
    """
    def __init__(
        self,
        root: str | os.PathLike,
        k_neighbors: int = 16,
        length_cap: int = 2048,
        num_workers: int = 4, # unused, but in config
    ):
        super().__init__()
        self.root = Path(root)
        self.k_neighbors = k_neighbors
        self.max_length = length_cap
        self._cache: dict[int, Any] = {}
        self._samples = self._scan_directory()

    def _scan_directory(self) -> list[tuple[Path, str]]:
        """Scans the root directory for valid mmCIF files and chains."""
        samples = []
        for path in self.root.glob("*.cif"):
            stem = path.stem
            if "_" not in stem:
                continue
            # Use rsplit to handle names like "pdb_1abc_A"
            chain_id = stem.rsplit("_", 1)[-1]
            samples.append((path, chain_id))

        # Here you could add a pre-filtering step to check if files are valid
        # to avoid errors during training, but for simplicity, we'll handle
        # errors in __getitem__.
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[GraphFeatures, ParsedProtein] | None:
        if index in self._cache:
            return self._cache[index]

        path, chain_id = self._samples[index]

        # Load and parse the protein file
        parsed_protein = load_protein_file(
            str(path),
            chain_id=chain_id,
            max_length=self.max_length,
        )
        if parsed_protein is None:
            return None # Skip invalid samples

        # Featurize the backbone into a graph representation
        graph_features = featurize_backbone(
            parsed_protein,
            k_neighbors=self.k_neighbors,
        )

        result = (graph_features, parsed_protein)
        self._cache[index] = result
        return result


def collate_backbones(batch: list[tuple[GraphFeatures, ParsedProtein] | None]):
    """
    Collate function for a batch of backbone samples.

    This function takes a list of (GraphFeatures, ParsedMmcif) tuples and
    combines them into a single batched graph. It also pads the raw
    coordinate and mask tensors to the maximum length in the batch.

    Returns:
        A dictionary of batched tensors ready for the model's forward pass.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    graph_features_list, parsed_protein_list = zip(*batch)

    # Collate graph features
    node_scalars = torch.cat([g.node_scalars for g in graph_features_list], dim=0)
    node_vectors = torch.cat([g.node_vectors for g in graph_features_list], dim=0)
    edge_scalars = torch.cat([g.edge_scalars for g in graph_features_list], dim=0)
    edge_vectors = torch.cat([g.edge_vectors for g in graph_features_list], dim=0)
    edge_frames = torch.cat([g.edge_frames for g in graph_features_list], dim=0)

    # Create a batch index for nodes
    node_counts = [g.node_scalars.shape[0] for g in graph_features_list]
    batch_idx = torch.repeat_interleave(
        torch.arange(len(node_counts)), torch.tensor(node_counts)
    )

    # Offset edge indices
    edge_offsets = torch.cumsum(torch.tensor([0] + node_counts[:-1]), dim=0)
    edge_index = torch.cat(
        [g.edge_index + offset for g, offset in zip(graph_features_list, edge_offsets)],
        dim=1,
    )

    # Pad the sequence-based tensors (coords, mask)
    coords = torch.nn.utils.rnn.pad_sequence(
        [p.coords for p in parsed_protein_list], batch_first=True
    )
    mask = torch.nn.utils.rnn.pad_sequence(
        [p.mask for p in parsed_protein_list], batch_first=True
    )

    return {
        "node_scalars": node_scalars,
        "node_vectors": node_vectors,
        "edge_index": edge_index,
        "edge_scalars": edge_scalars,
        "edge_vectors": edge_vectors,
        "edge_frames": edge_frames,
        "batch_idx": batch_idx,
        "coords": coords,
        "mask": mask,
    }
