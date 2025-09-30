"""Torch dataset wrappers for protein backbones and tokens."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .featurize import featurize_backbone
from .mmcif import PAD_INDEX, BackboneRecord, load_mmcif

Tensor = torch.Tensor

try:  # pragma: no cover - tqdm is optional at runtime
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _load_records_for_dataset(args: Tuple[str, int, Optional[Sequence[str]]]) -> List[BackboneRecord]:
    path, length_cap, chain_filter = args
    chains = set(chain_filter) if chain_filter is not None else None

    records = load_mmcif(path, length_cap=length_cap)
    filtered: List[BackboneRecord] = []
    for record in records:
        if chains is not None and record.chain_id not in chains:
            continue

        if record.mask.sum().item() == 0:
            continue

        filtered.append(record)

    return filtered


def _discover_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]

    patterns = [
        "*.cif",
        "*.cif.gz",
        "*.mmcif",
        "*.mmcif.gz",
        "*.pdb",
        "*.pdb.gz",
        "*.ent",
        "*.ent.gz",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.rglob(pattern)))
    return files


class BackboneDataset(Dataset):
    """Dataset of protein backbones ready for GCP featurization."""

    def __init__(
        self,
        root: str | Path,
        *,
        chain_ids: Optional[Iterable[str]] = None,
        length_cap: int = 2048,
        k: int = 16,
        cache: bool = True,
        progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(root)

        self.length_cap = length_cap
        self.k = k
        self.chain_filter = set(chain_ids) if chain_ids is not None else None
        self._cache_enabled = cache
        self._records: Dict[Tuple[str, str], BackboneRecord] = {}
        self._keys: List[Tuple[str, str]] = []

        files = _discover_files(self.root)

        total = len(files)
        show_progress = progress and tqdm is not None and total > 0
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=total, desc="Parsing backbone files")

        chain_filter: Optional[Sequence[str]]
        if self.chain_filter is not None:
            chain_filter = sorted(self.chain_filter)
        else:
            chain_filter = None

        worker_count = (num_workers or 0)
        try:
            if worker_count > 1 and total > 1:
                ctx = get_context("spawn")
                with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
                    args = [(str(file), length_cap, chain_filter) for file in files]
                    for records in executor.map(_load_records_for_dataset, args):
                        self._store_records(records)
                        if progress_bar is not None:
                            progress_bar.update(1)
            else:
                for file in files:
                    records = _load_records_for_dataset((str(file), length_cap, chain_filter))
                    self._store_records(records)
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if not self._keys:
            raise ValueError("Dataset does not contain any valid backbone chains")

    def __len__(self) -> int:
        return len(self._keys)

    def _store_records(self, records: Iterable[BackboneRecord]) -> None:
        for record in records:
            key = (record.path, record.chain_id)
            self._keys.append(key)
            if self._cache_enabled:
                self._records[key] = record

    def _get_record(self, key: Tuple[str, str]) -> BackboneRecord:
        if self._cache_enabled and key in self._records:
            return self._records[key]

        path, chain_id = key
        records = load_mmcif(path, chain_id=chain_id, length_cap=self.length_cap)
        if not records:
            raise KeyError(f"Unable to load chain {chain_id!r} from {path}")
        record = records[0]
        if self._cache_enabled:
            self._records[key] = record
        return record

    def __getitem__(self, index: int) -> Dict[str, Tensor | Dict[str, object]]:
        key = self._keys[index]
        record = self._get_record(key)

        features = featurize_backbone(record, k=self.k)

        sample: Dict[str, Tensor | Dict[str, object]] = {
            "coords": record.coords.clone(),
            "mask": record.mask.clone(),
            "atom_mask": record.atom_mask.clone(),
            "seq": record.seq.clone(),
            "seq_str": record.seq_string,
            "nan_mask": record.nan_mask.clone(),
            "node_scalars": features["node_scalars"],
            "node_vectors": features["node_vectors"],
            "backbone_vectors": features["backbone_vectors"],
            "torsion_angles": features["torsion_angles"],
            "edge_index": features["edge_index"],
            "edge_scalars": features["edge_scalars"],
            "edge_vectors": features["edge_vectors"],
            "edge_frames": features["edge_frames"],
            "pose": {
                "rotation": record.rotation.clone(),
                "translation": record.translation.clone(),
            },
            "metadata": {
                "path": record.path,
                "chain_id": record.chain_id,
                "sequence": record.seq_string,
                "residue_ids": list(record.residue_ids),
                "residue_names": list(record.residue_names),
            },
        }

        return sample


def collate_backbones(batch: List[Dict[str, Tensor | Dict[str, object]]]) -> Dict[str, Tensor | List[Dict[str, object]]]:
    if not batch:
        raise ValueError("Batch must not be empty")

    max_len = max(item["coords"].shape[0] for item in batch)  # type: ignore[arg-type]
    batch_size = len(batch)

    dtype = batch[0]["coords"].dtype  # type: ignore[index]
    device = batch[0]["coords"].device  # type: ignore[index]

    coords = torch.zeros((batch_size, max_len, 3, 3), dtype=dtype, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    atom_mask = torch.zeros((batch_size, max_len, 3), dtype=torch.bool, device=device)
    seq = torch.full((batch_size, max_len), PAD_INDEX, dtype=torch.long, device=device)
    nan_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    node_scalars = torch.zeros((batch_size, max_len, 6), dtype=dtype, device=device)
    node_vectors = torch.zeros((batch_size, max_len, 3, 3), dtype=dtype, device=device)
    backbone_vectors = torch.zeros((batch_size, max_len, 6, 3), dtype=dtype, device=device)
    torsion_angles = torch.zeros((batch_size, max_len, 3), dtype=dtype, device=device)

    lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)

    edge_indices: List[Tensor] = []
    edge_scalars: List[Tensor] = []
    edge_vectors: List[Tensor] = []
    edge_frames: List[Tensor] = []
    edge_batch = []
    node_batch = []

    node_offset = 0
    metadata: List[Dict[str, object]] = []
    sequences: List[str] = []
    rotations = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    translations = torch.zeros((batch_size, 3), dtype=dtype, device=device)

    for i, item in enumerate(batch):
        length = item["coords"].shape[0]  # type: ignore[index]
        lengths[i] = length

        coords[i, :length] = item["coords"]  # type: ignore[index]
        mask[i, :length] = item["mask"]  # type: ignore[index]
        atom_mask[i, :length] = item["atom_mask"]  # type: ignore[index]
        seq[i, :length] = item["seq"]  # type: ignore[index]
        nan_mask[i, :length] = item["nan_mask"]  # type: ignore[index]
        node_scalars[i, :length] = item["node_scalars"]  # type: ignore[index]
        node_vectors[i, :length] = item["node_vectors"]  # type: ignore[index]
        backbone_vectors[i, :length] = item["backbone_vectors"]  # type: ignore[index]
        torsion_angles[i, :length] = item["torsion_angles"]  # type: ignore[index]

        edge_idx = item["edge_index"]  # type: ignore[index]
        edge_indices.append(edge_idx + node_offset)
        edge_scalars.append(item["edge_scalars"])  # type: ignore[index]
        edge_vectors.append(item["edge_vectors"])  # type: ignore[index]
        edge_frames.append(item["edge_frames"])  # type: ignore[index]
        edge_batch.append(torch.full((edge_idx.shape[1],), i, dtype=torch.long, device=device))

        node_batch.append(torch.full((length,), i, dtype=torch.long, device=device))

        pose = item["pose"]  # type: ignore[index]
        rotations[i] = pose["rotation"]  # type: ignore[index]
        translations[i] = pose["translation"]  # type: ignore[index]

        metadata.append(item["metadata"])  # type: ignore[index]
        sequences.append(item["seq_str"])  # type: ignore[index]

        node_offset += length

    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_scalars_tensor = torch.cat(edge_scalars, dim=0) if edge_scalars else torch.empty((0, 8), dtype=dtype, device=device)
    edge_vectors_tensor = torch.cat(edge_vectors, dim=0) if edge_vectors else torch.empty((0, 3), dtype=dtype, device=device)
    edge_frames_tensor = torch.cat(edge_frames, dim=0) if edge_frames else torch.empty((0, 3, 3), dtype=dtype, device=device)
    edge_batch_tensor = torch.cat(edge_batch, dim=0) if edge_batch else torch.empty((0,), dtype=torch.long, device=device)
    node_batch_tensor = torch.cat(node_batch, dim=0) if node_batch else torch.empty((0,), dtype=torch.long, device=device)

    return {
        "coords": coords,
        "mask": mask,
        "atom_mask": atom_mask,
        "seq": seq,
        "nan_mask": nan_mask,
        "node_scalars": node_scalars,
        "node_vectors": node_vectors,
        "backbone_vectors": backbone_vectors,
        "torsion_angles": torsion_angles,
        "edge_index": edge_index,
        "edge_scalars": edge_scalars_tensor,
        "edge_vectors": edge_vectors_tensor,
        "edge_frames": edge_frames_tensor,
        "edge_batch": edge_batch_tensor,
        "node_batch": node_batch_tensor,
        "lengths": lengths,
        "rotations": rotations,
        "translations": translations,
        "metadata": metadata,
        "sequences": sequences,
    }


__all__ = ["BackboneDataset", "collate_backbones"]
