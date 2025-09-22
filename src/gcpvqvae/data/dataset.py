"""Torch dataset wrappers for protein backbones and tokens."""

from __future__ import annotations

from torch.utils.data import Dataset


class BackboneDataset(Dataset):
    """Dataset of protein backbones ready for GCP featurization."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


def collate_backbones(batch):
    """Collate function for backbone samples (stub)."""
    raise NotImplementedError("collate_backbones is not yet implemented")
