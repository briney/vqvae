"""Torch dataset wrappers for protein backbones and tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .featurize import featurize_backbone
from .mmcif import (
    AA_TO_INDEX,
    ONE_TO_THREE,
    PAD_INDEX,
    BackboneRecord,
    load_mmcif,
)

Tensor = torch.Tensor

PREPROCESSED_MANIFEST = "preprocessed_dataset.json"
PREPROCESSED_REFERENCE_MANIFEST = "preprocessed_reference_dataset.json"
PREPROCESSED_SAMPLES_DIR = "samples"
PREPROCESSED_VERSION = 1

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


def _clone_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value


def _trim_sample(sample: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    if max_length <= 0:
        return sample

    seq_length = int(sample["coords"].shape[0])  # type: ignore[index]
    if seq_length <= max_length:
        return sample

    slice_obj = slice(0, max_length)
    tensor_keys = [
        "coords",
        "mask",
        "atom_mask",
        "seq",
        "nan_mask",
        "node_scalars",
        "node_vectors",
        "backbone_vectors",
        "torsion_angles",
    ]
    for key in tensor_keys:
        if key in sample and isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key][slice_obj]

    if "edge_index" in sample and isinstance(sample["edge_index"], torch.Tensor):
        edge_index = sample["edge_index"]
        valid = (edge_index < max_length).all(dim=0)
        sample["edge_index"] = edge_index[:, valid]
        for edge_key in ("edge_scalars", "edge_vectors", "edge_frames"):
            if edge_key in sample and isinstance(sample[edge_key], torch.Tensor):
                sample[edge_key] = sample[edge_key][valid]

    if "seq_str" in sample and isinstance(sample["seq_str"], str):
        sample["seq_str"] = sample["seq_str"][:max_length]

    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        trimmed = dict(metadata)
        if "sequence" in trimmed and isinstance(trimmed["sequence"], str):
            trimmed["sequence"] = trimmed["sequence"][:max_length]
        if "residue_ids" in trimmed and isinstance(trimmed["residue_ids"], list):
            trimmed["residue_ids"] = trimmed["residue_ids"][:max_length]
        if "residue_names" in trimmed and isinstance(trimmed["residue_names"], list):
            trimmed["residue_names"] = trimmed["residue_names"][:max_length]
        sample["metadata"] = trimmed

    return sample


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
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(root)

        self.length_cap = length_cap
        self.k = k
        self.chain_filter = set(chain_ids) if chain_ids is not None else None
        self._cache_enabled = cache
        self._records: Dict[Tuple[str, str], BackboneRecord | Dict[str, Any]] = {}
        self._keys: List[Tuple[str, str]] = []
        self._preprocessed = False
        self._preprocessed_files: List[Path] = []
        self._preprocessed_entries: List[Dict[str, Any]] = []
        self._preprocessed_loader: str = "pt"
        self._source_length_cap: Optional[int] = None

        manifest_candidates: List[Path] = []
        if self.root.is_file() and self.root.name in (
            PREPROCESSED_MANIFEST,
            PREPROCESSED_REFERENCE_MANIFEST,
        ):
            manifest_candidates.append(self.root)
        elif self.root.is_dir():
            for name in (PREPROCESSED_MANIFEST, PREPROCESSED_REFERENCE_MANIFEST):
                candidate = self.root / name
                if candidate.is_file():
                    manifest_candidates.append(candidate)

        for manifest_path in manifest_candidates:
            self._init_from_preprocessed(manifest_path, chain_ids, length_cap, k)
            if not self._keys:
                continue
            return

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

        try:
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
            cached = self._records[key]
            if isinstance(cached, BackboneRecord):
                return cached

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
        if self._preprocessed:
            return self._get_preprocessed_sample(index, key)
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

    def _init_from_preprocessed(
        self,
        manifest_path: Path,
        chain_ids: Optional[Iterable[str]],
        length_cap: int,
        k: int,
    ) -> None:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        chain_filter = set(chain_ids) if chain_ids is not None else None

        entries: List[Dict[str, Any]]
        manifest_k: Optional[int]
        if "entries" in manifest:
            version = manifest.get("version")
            if version != PREPROCESSED_VERSION:
                raise ValueError(
                    f"Unsupported preprocessed dataset version {version!r}; expected {PREPROCESSED_VERSION}"
                )

            manifest_k = manifest.get("k")
            if manifest_k is not None and manifest_k != k:
                raise ValueError(
                    f"Preprocessed dataset was generated with k={manifest_k} "
                    f"but k={k} was requested"
                )

            entries = manifest.get("entries")
            if not isinstance(entries, list):
                raise ValueError("Invalid preprocessed dataset manifest: missing 'entries'")

            filtered: List[Dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                chain_id = entry.get("chain_id")
                file_rel = entry.get("file")
                if not isinstance(chain_id, str) or not isinstance(file_rel, str):
                    continue
                if chain_filter is not None and chain_id not in chain_filter:
                    continue
                filtered.append(entry)

            if not filtered:
                raise ValueError("Dataset does not contain any valid backbone chains")

            source_length_cap = manifest.get("length_cap")
            if isinstance(source_length_cap, int) and source_length_cap > 0:
                self._source_length_cap = source_length_cap
            else:
                self._source_length_cap = None

            self.k = manifest_k if isinstance(manifest_k, int) and manifest_k > 0 else k
            entries = filtered
            loader = "pt"
        elif "chains" in manifest:
            chains = manifest.get("chains")
            if not isinstance(chains, list):
                raise ValueError("Invalid preprocessed dataset manifest: missing 'chains'")

            filtered = []
            for entry in chains:
                if not isinstance(entry, dict):
                    continue
                chain_id = entry.get("chain_id")
                h5_path = entry.get("h5_path")
                if not isinstance(chain_id, str) or not isinstance(h5_path, str):
                    continue
                if chain_filter is not None and chain_id not in chain_filter:
                    continue
                filtered.append(entry)

            if not filtered:
                raise ValueError("Dataset does not contain any valid backbone chains")

            self._source_length_cap = None
            self.k = k
            entries = filtered
            loader = "h5"
        else:
            raise ValueError("Invalid preprocessed dataset manifest: missing entries list")

        self.length_cap = length_cap
        if self._source_length_cap is not None:
            self.length_cap = min(self.length_cap, self._source_length_cap)
        self.chain_filter = chain_filter
        self._preprocessed = True
        self._preprocessed_files = []
        self._preprocessed_entries = []
        self._keys = []

        for entry in entries:
            if loader == "pt":
                file_rel = entry["file"]
            else:
                file_rel = entry["h5_path"]
            sample_path = manifest_path.parent / file_rel
            if not sample_path.exists():
                raise FileNotFoundError(sample_path)
            source_path = entry.get("source_path")
            if not isinstance(source_path, str):
                source_path = str(sample_path)
            chain_id = entry.get("chain_id")
            if not isinstance(chain_id, str):
                raise ValueError("Manifest entry missing chain identifier")
            key = (source_path, chain_id)
            self._keys.append(key)
            self._preprocessed_files.append(sample_path)
            self._preprocessed_entries.append(entry)

        self._preprocessed_loader = loader

    def _get_preprocessed_sample(
        self, index: int, key: Tuple[str, str]
    ) -> Dict[str, Tensor | Dict[str, object]]:
        if self._cache_enabled and key in self._records:
            cached = self._records[key]
        else:
            sample_path = self._preprocessed_files[index]
            entry = self._preprocessed_entries[index]
            if sample_path.suffix == ".h5" or self._preprocessed_loader == "h5":
                cached = _load_h5_sample(sample_path, entry, k=self.k)
            else:
                cached = torch.load(sample_path, map_location="cpu")
                if not isinstance(cached, dict):
                    raise TypeError(f"Preprocessed sample at {sample_path} is not a mapping")
            if self._cache_enabled:
                self._records[key] = cached

        sample = _clone_nested(cached)
        if self.length_cap and self.length_cap > 0:
            sample = _trim_sample(sample, self.length_cap)
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


__all__ = [
    "BackboneDataset",
    "collate_backbones",
    "PREPROCESSED_MANIFEST",
    "PREPROCESSED_SAMPLES_DIR",
    "PREPROCESSED_VERSION",
]
def _decode_h5_sequence(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "V"}:
            return value.tobytes().decode("utf-8").rstrip("\x00")
        if value.dtype.kind == "U":
            return "".join(value.tolist())
        if value.shape == ():
            return str(value.item())
    if isinstance(value, str):
        return value
    raise TypeError(f"Unable to decode sequence from HDF5 payload of type {type(value)!r}")


def _normalise_backbone_coords(coords: torch.Tensor, ca_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = coords.clone()
    dtype = coords.dtype
    device = coords.device
    if coords.numel() == 0:
        rotation = torch.eye(3, dtype=dtype, device=device)
        translation = torch.zeros(3, dtype=dtype, device=device)
        return coords, rotation, translation

    if ca_mask.any():
        centroid = coords[ca_mask, 1, :].mean(dim=0)
    else:
        centroid = coords[:, 1, :].mean(dim=0)

    coords = coords - centroid.view(1, 1, 3)
    rotation = torch.eye(3, dtype=dtype, device=device)
    translation = centroid
    return coords, rotation, translation


def _load_h5_sample(
    sample_path: Path,
    entry: Dict[str, Any],
    *,
    k: int,
) -> Dict[str, Any]:
    with h5py.File(sample_path, "r") as handle:
        coords_raw = np.asarray(handle["N_CA_C_O_coord"])  # (L, 4, 3)
        seq_value = handle["seq"][()]
        plddt = np.asarray(handle.get("plddt_scores")) if "plddt_scores" in handle else None

    seq_str = _decode_h5_sequence(seq_value)
    nca_coords = coords_raw[:, :3, :]
    atom_present = ~np.isnan(nca_coords)
    coords_clean = np.where(atom_present, nca_coords, 0.0)
    atom_mask = atom_present.all(axis=2)
    residue_mask = atom_mask.all(axis=1)
    nan_mask = np.isnan(nca_coords[:, 1, :]).any(axis=1)

    coords_tensor = torch.from_numpy(coords_clean.astype(np.float32, copy=False))
    atom_mask_tensor = torch.from_numpy(atom_mask)
    mask_tensor = torch.from_numpy(residue_mask)
    nan_mask_tensor = torch.from_numpy(nan_mask)

    coords_tensor, rotation, translation = _normalise_backbone_coords(
        coords_tensor, atom_mask_tensor[:, 1]
    )

    seq_indices = [AA_TO_INDEX.get(residue, AA_TO_INDEX["X"]) for residue in seq_str]
    seq_tensor = torch.tensor(seq_indices, dtype=torch.long)

    residue_names = [ONE_TO_THREE.get(residue, "UNK") for residue in seq_str]
    residue_ids = [(idx + 1, " ") for idx in range(len(seq_str))]

    source_path = entry.get("source_path") or str(sample_path)
    chain_id = entry.get("chain_id") or "?"

    record = BackboneRecord(
        path=str(source_path),
        chain_id=str(chain_id),
        coords=coords_tensor,
        mask=mask_tensor,
        atom_mask=atom_mask_tensor,
        seq=seq_tensor,
        seq_string=seq_str,
        residue_names=residue_names,
        residue_ids=residue_ids,
        rotation=rotation,
        translation=translation,
        nan_mask=nan_mask_tensor,
    )

    features = featurize_backbone(record, k=k)

    metadata: Dict[str, Any] = {
        "path": record.path,
        "chain_id": record.chain_id,
        "sequence": seq_str,
        "residue_ids": residue_ids,
        "residue_names": residue_names,
    }
    if plddt is not None:
        metadata["plddt_scores"] = plddt.tolist()

    sample: Dict[str, Any] = {
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
        "metadata": metadata,
    }
    return sample
