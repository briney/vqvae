"""Torch dataset wrappers for protein backbones and tokens."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .featurize import featurize_backbone
from .mmcif import AA_TO_INDEX, ONE_TO_THREE, PAD_INDEX, BackboneRecord, load_mmcif

Tensor = torch.Tensor

PREPROCESSED_MANIFEST = "preprocessed_dataset.json"
PREPROCESSED_SAMPLES_DIR = "samples"
PREPROCESSED_VERSION = 1

try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
except ImportError:  # pragma: no cover - fallback when futures missing
    ProcessPoolExecutor = None  # type: ignore[assignment]
    as_completed = None  # type: ignore[assignment]

try:  # pragma: no cover - tqdm is optional at runtime
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _load_records_for_dataset(args: Tuple[str, int, Optional[Sequence[str]]]) -> List[BackboneRecord]:
    """Load backbone records for a single structure path.

    Args:
        args: Tuple ``(path, length_cap, chain_ids)`` where ``path`` points to a
            backbone file, ``length_cap`` limits residues per chain (0 disables
            trimming), and ``chain_ids`` optionally filters to specific chain
            identifiers.

    Returns:
        List of ``BackboneRecord`` instances satisfying the filters. Chains with
        zero valid residues are skipped.
    """
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
    """Enumerate candidate backbone files under ``root``.

    Args:
        root: Directory or file path pointing to protein structure data.

    Returns:
        List of file paths matching known backbone extensions (mmCIF/PDB). If
        ``root`` is a file, the list contains only ``root``.
    """
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
    """Recursively clone tensors, dicts, tuples, and lists.

    Args:
        value: Arbitrary nested structure containing tensors and Python
            containers.

    Returns:
        Deep copy of ``value`` where tensors are cloned to avoid in-place
        modification of cached samples.
    """
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
    """Truncate a preprocessed sample to ``max_length`` residues in-place.

    Args:
        sample: Mapping with tensor entries such as ``coords`` and ``edge_index``.
        max_length: Length cap to enforce. Values ``<= 0`` leave the sample
            unchanged.

    Returns:
        The trimmed sample mapping. All sequence-dependent fields are slice
        truncated, and edges exceeding the cap are removed.
    """
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


def _decode_h5_sequence(raw: Any) -> str:
    """Convert HDF5-backed sequence payloads into ASCII strings.

    Args:
        raw: Sequence representation returned by ``h5py``. Accepts bytes,
            numpy arrays, or nested iterables.

    Returns:
        Uppercase amino-acid sequence decoded to ASCII without NULL padding.
    """
    if isinstance(raw, bytes):
        return raw.decode("ascii").replace("\x00", "")
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in {"S", "a"}:
            return raw.tobytes().decode("ascii").replace("\x00", "")
        if raw.dtype.kind in {"U"}:
            return "".join(raw.tolist())
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        return "".join(_decode_h5_sequence(item) for item in raw)
    return str(raw)


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
        num_parsing_workers: Optional[int] = None,
    ) -> None:
        """Parse backbone files and optionally load cached features.

        Args:
            root: Path to a directory or file containing mmCIF/PDB chains or a
                preprocessed dataset.
            chain_ids: Optional iterable of chain identifiers to keep.
            length_cap: Maximum number of residues per chain; zero disables
                trimming.
            k: Number of nearest neighbours for graph construction when
                featurising on-the-fly.
            cache: Whether to memoize loaded records or preprocessed samples in
                memory.
            progress: Show a CLI progress bar when parsing raw structure files.
            num_parsing_workers: Optional number of worker processes used while
                reading structure files. ``None`` defaults to ``ProcessPoolExecutor``'s
                behaviour (typically the number of CPUs). Values ``<= 1`` fall back to
                sequential parsing.

        Raises:
            FileNotFoundError: If ``root`` does not exist.
            ValueError: If no valid backbone chains are discovered.
        """
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
        self._preprocessed_formats: List[str] = []
        self._preprocessed_metadata: List[Dict[str, Any]] = []
        self._source_length_cap: Optional[int] = None
        env_force_sequential = os.environ.get("GCPVQVAE_FORCE_SEQUENTIAL_LOAD")
        if env_force_sequential:
            self.num_parsing_workers = 1
        else:
            if num_parsing_workers is None or num_parsing_workers <= 0:
                self.num_parsing_workers = None
            else:
                self.num_parsing_workers = int(num_parsing_workers)

        if self.root.is_dir():
            manifest_path = self.root / PREPROCESSED_MANIFEST
            if manifest_path.is_file():
                self._init_from_preprocessed(manifest_path, chain_ids, length_cap, k)
                if not self._keys:
                    raise ValueError("Dataset does not contain any valid backbone chains")
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
            worker_count = self.num_parsing_workers
            use_parallel = (
                ProcessPoolExecutor is not None
                and as_completed is not None
                and (worker_count is None or worker_count > 1)
            )
            if use_parallel:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(
                            _load_records_for_dataset,
                            (str(file), length_cap, chain_filter),
                        ): file
                        for file in files
                    }
                    for future in as_completed(futures):
                        records = future.result()
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
        """Return the number of chains in the dataset."""
        return len(self._keys)

    def _store_records(self, records: Iterable[BackboneRecord]) -> None:
        """Cache newly parsed records and append their lookup keys.

        Args:
            records: Iterable of backbone records generated from raw structure
                files.
        """
        for record in records:
            key = (record.path, record.chain_id)
            self._keys.append(key)
            if self._cache_enabled:
                self._records[key] = record

    def _get_record(self, key: Tuple[str, str]) -> BackboneRecord:
        """Load a backbone record, using the cache when available.

        Args:
            key: Tuple ``(path, chain_id)`` uniquely identifying a chain.

        Returns:
            Backbone record with featurizable tensors normalized to the origin.

        Raises:
            KeyError: If the requested chain cannot be retrieved.
        """
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
        """Return featurized tensors for a backbone chain.

        Args:
            index: Dataset index referring to a ``(path, chain_id)`` pair.

        Returns:
            Mapping containing backbone coordinates ``(L, 3, 3)``, boolean masks,
            Hydra-ready node and edge features, rigid pose information, and
            metadata about the source chain.
        """
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
        """Initialise the dataset from a preprocessed manifest on disk.

        Args:
            manifest_path: Path to ``preprocessed_dataset.json``.
            chain_ids: Optional chain filter applied to the manifest entries.
            length_cap: Maximum residue count retained from the manifest.
            k: Expected k-NN parameter used to validate compatibility.

        Raises:
            ValueError: If the manifest version or layout is unsupported.
        """
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries")
        if isinstance(entries, list):
            self._init_from_torch_manifest(manifest_path, manifest, entries, chain_ids, length_cap, k)
            return

        chains = manifest.get("chains")
        if isinstance(chains, list):
            self._init_from_hdf5_manifest(manifest_path, chains, chain_ids, length_cap, k)
            return

        version = manifest.get("version")
        raise ValueError(
            f"Unsupported preprocessed dataset manifest format with version {version!r}"
        )

    def _init_from_torch_manifest(
        self,
        manifest_path: Path,
        manifest: Dict[str, Any],
        entries: List[Dict[str, Any]],
        chain_ids: Optional[Iterable[str]],
        length_cap: int,
        k: int,
    ) -> None:
        """Configure the dataset using Torch-generated preprocessing artefacts.

        Args:
            manifest_path: Path to the manifest JSON file.
            manifest: Parsed manifest contents.
            entries: List of entry dictionaries describing cached Torch tensors.
            chain_ids: Optional chain filter to apply.
            length_cap: Maximum number of residues retained per chain.
            k: Expected k-NN parameter used for compatibility checks.

        Raises:
            ValueError: If the manifest version mismatches or no chains remain
                after filtering.
            FileNotFoundError: If a referenced sample file is missing.
        """
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

        chain_filter = set(chain_ids) if chain_ids is not None else None

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

        self.length_cap = length_cap
        if self._source_length_cap is not None:
            self.length_cap = min(self.length_cap, self._source_length_cap)
        self.k = manifest_k if isinstance(manifest_k, int) and manifest_k > 0 else k
        self.chain_filter = chain_filter
        self._preprocessed = True
        self._preprocessed_files = []
        self._preprocessed_formats = []
        self._preprocessed_metadata = []
        self._keys = []

        for entry in filtered:
            file_rel = entry["file"]
            sample_path = manifest_path.parent / file_rel
            if not sample_path.exists():
                raise FileNotFoundError(sample_path)
            source_path = entry.get("source_path")
            if not isinstance(source_path, str):
                source_path = str(sample_path)
            chain_id = entry["chain_id"]
            key = (source_path, chain_id)
            self._keys.append(key)
            self._preprocessed_files.append(sample_path)
            self._preprocessed_formats.append("pt")
            self._preprocessed_metadata.append(
                {
                    "chain_id": chain_id,
                    "source_path": source_path,
                    "sequence": entry.get("sequence"),
                    "length": entry.get("length"),
                }
            )

    def _init_from_hdf5_manifest(
        self,
        manifest_path: Path,
        chains: List[Dict[str, Any]],
        chain_ids: Optional[Iterable[str]],
        length_cap: int,
        k: int,
    ) -> None:
        """Configure the dataset using HDF5-based preprocessing artefacts.

        Args:
            manifest_path: Path to the manifest JSON file.
            chains: List of chain dictionaries describing cached HDF5 samples.
            chain_ids: Optional chain filter to apply.
            length_cap: Maximum number of residues retained per chain.
            k: Stored k-NN parameter (for API parity with Torch preprocessing).

        Raises:
            ValueError: If no chains remain after filtering.
            FileNotFoundError: If a referenced sample file is missing.
        """
        chain_filter = set(chain_ids) if chain_ids is not None else None

        filtered: List[Dict[str, Any]] = []
        for entry in chains:
            if not isinstance(entry, dict):
                continue
            chain_id = entry.get("chain_id")
            file_rel = entry.get("h5_path")
            if not isinstance(chain_id, str) or not isinstance(file_rel, str):
                continue
            if chain_filter is not None and chain_id not in chain_filter:
                continue
            filtered.append(entry)

        if not filtered:
            raise ValueError("Dataset does not contain any valid backbone chains")

        lengths = [entry.get("length") for entry in filtered if isinstance(entry.get("length"), int)]
        self._source_length_cap = max(lengths) if lengths else None

        self.length_cap = length_cap
        if self._source_length_cap is not None:
            self.length_cap = min(self.length_cap, self._source_length_cap)
        self.k = k
        self.chain_filter = chain_filter
        self._preprocessed = True
        self._preprocessed_files = []
        self._preprocessed_formats = []
        self._preprocessed_metadata = []
        self._keys = []

        for entry in filtered:
            file_rel = entry["h5_path"]
            sample_path = manifest_path.parent / file_rel
            if not sample_path.exists():
                raise FileNotFoundError(sample_path)
            source_path = entry.get("source_path")
            if not isinstance(source_path, str):
                source_path = str(sample_path)
            chain_id = entry["chain_id"]
            key = (source_path, chain_id)
            self._keys.append(key)
            self._preprocessed_files.append(sample_path)
            self._preprocessed_formats.append("h5")
            self._preprocessed_metadata.append(
                {
                    "chain_id": chain_id,
                    "source_path": source_path,
                    "sequence": entry.get("sequence"),
                    "length": entry.get("length"),
                }
            )

    def _get_preprocessed_sample(
        self, index: int, key: Tuple[str, str]
    ) -> Dict[str, Tensor | Dict[str, object]]:
        """Load and trim a preprocessed sample from disk or cache.

        Args:
            index: Dataset index used to select the preprocessed file path.
            key: Tuple ``(source_path, chain_id)`` indexing the cache dictionary.

        Returns:
            Deep-copied sample mapping trimmed to ``self.length_cap`` residues.

        Raises:
            ValueError: If the preprocessed file format is unknown.
            TypeError: If a cached payload is not dictionary-like.
        """
        if self._cache_enabled and key in self._records:
            cached = self._records[key]
        else:
            sample_path = self._preprocessed_files[index]
            fmt = self._preprocessed_formats[index] if index < len(self._preprocessed_formats) else "pt"
            if fmt == "pt":
                cached = torch.load(sample_path, map_location="cpu")
                if not isinstance(cached, dict):
                    raise TypeError(f"Preprocessed sample at {sample_path} is not a mapping")
            elif fmt == "h5":
                metadata = self._preprocessed_metadata[index] if index < len(self._preprocessed_metadata) else {}
                cached = self._load_h5_sample(sample_path, key, metadata)
            else:
                raise ValueError(f"Unknown preprocessed sample format: {fmt}")
            if self._cache_enabled:
                self._records[key] = cached

        sample = _clone_nested(cached)
        if self.length_cap and self.length_cap > 0:
            sample = _trim_sample(sample, self.length_cap)
        return sample

    def _load_h5_sample(
        self,
        sample_path: Path,
        key: Tuple[str, str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Tensor | Dict[str, object]]:
        """Load an HDF5 sample and convert it to the standard feature dict.

        Args:
            sample_path: Path to the ``.h5`` file containing cached tensors.
            key: Tuple ``(source_path, chain_id)`` indicating the dataset entry.
            metadata: Supplemental metadata from the manifest describing the
                sample.

        Returns:
            Mapping identical to :meth:`__getitem__` output plus
            ``plddt_scores`` extracted from the file.

        Raises:
            RuntimeError: If ``h5py`` is not installed when loading HDF5 files.
            TypeError: If the stored payload is not a dictionary.
        """
        try:
            import h5py  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Loading preprocessed HDF5 datasets requires the 'h5py' package"
            ) from exc

        with h5py.File(sample_path, "r") as handle:
            seq_raw = handle["seq"][()]
            coords_raw = np.asarray(handle["N_CA_C_O_coord"][()])
            plddt_raw = np.asarray(handle["plddt_scores"][()], dtype=np.float32)

        seq_str = _decode_h5_sequence(seq_raw)
        seq_indices = [AA_TO_INDEX.get(aa, AA_TO_INDEX["X"]) for aa in seq_str]
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)

        coords_backbone = np.asarray(coords_raw[:, :3, :], dtype=np.float32)
        valid_atoms = ~np.isnan(coords_backbone)
        atom_mask = valid_atoms.all(axis=2)

        coords_filled = np.where(valid_atoms, coords_backbone, 0.0)
        coords_tensor = torch.from_numpy(coords_filled)
        atom_mask_tensor = torch.from_numpy(atom_mask.astype(np.bool_))
        mask_tensor = atom_mask_tensor.all(dim=1)
        ca_mask_tensor = atom_mask_tensor[:, 1]
        nan_mask_tensor = ~ca_mask_tensor

        ca_positions = coords_tensor[:, 1, :]
        if bool(ca_mask_tensor.any()):
            centroid = ca_positions[ca_mask_tensor].mean(dim=0)
        else:
            centroid = ca_positions.mean(dim=0)
        coords_tensor = coords_tensor - centroid.view(1, 1, 3)
        rotation = torch.eye(3, dtype=coords_tensor.dtype)
        translation = centroid

        residue_names = [ONE_TO_THREE.get(aa, "UNK") for aa in seq_str]
        residue_ids = [(idx + 1, "") for idx in range(len(seq_str))]

        record = BackboneRecord(
            path=metadata.get("source_path", str(sample_path)),
            chain_id=metadata.get("chain_id", key[1]),
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

        features = featurize_backbone(record, k=self.k)

        plddt_tensor = torch.from_numpy(plddt_raw)

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
                "source_h5": str(sample_path),
            },
            "plddt_scores": plddt_tensor,
        }

        return sample


def collate_backbones(batch: List[Dict[str, Tensor | Dict[str, object]]]) -> Dict[str, Tensor | List[Dict[str, object]]]:
    """Collate backbone samples into padded batch tensors.

    Args:
        batch: List of samples produced by :class:`BackboneDataset`. Each item
            must contain the keys emitted by :meth:`BackboneDataset.__getitem__`.

    Returns:
        Dictionary with batched tensors ready for model consumption. Notable
        shapes include ``coords`` with ``(B, L_max, 3, 3)``, ``node_scalars`` with
        ``(B, L_max, 6)``, and ``edge_index`` with ``(2, E_total)``. Metadata and
        sequence strings are returned as Python lists.

    Raises:
        ValueError: If ``batch`` is empty.

    Examples:
        >>> dataset = BackboneDataset("path/to/data", cache=False)
        >>> sample_batch = [dataset[0], dataset[1]]
        >>> collated = collate_backbones(sample_batch)
        >>> collated["coords"].shape
        torch.Size([2, collated["lengths"].max(), 3, 3])
    """
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
