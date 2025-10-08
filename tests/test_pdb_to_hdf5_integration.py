"""Integration tests covering preprocessing and training on real structures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

h5py = pytest.importorskip("h5py")

from gcpvqvae.data.dataset import BackboneDataset
from gcpvqvae.data.preprocess import preprocess_dataset, preprocess_structure
from gcpvqvae.system.train import train


_SUBSET_FILES = ["1A2N.cif", "1ACX.cif", "1B0I.cif"]


def _prepare_subset(tmp_path: Path, filenames: list[str] | None = None) -> Path:
    source_root = Path(__file__).resolve().parent / "test_data" / "cif_50"
    subset_root = tmp_path / "structures"
    subset_root.mkdir(parents=True, exist_ok=True)

    selected = filenames or _SUBSET_FILES
    for name in selected:
        shutil.copy2(source_root / name, subset_root / name)

    return subset_root


def _decode_sequence(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("ascii")
    if hasattr(raw, "tobytes"):
        return raw.tobytes().decode("ascii")
    return str(raw)


def test_preprocess_dataset_generates_hdf5_matching_structures(tmp_path: Path) -> None:
    subset_dir = _prepare_subset(tmp_path)
    output_dir = tmp_path / "processed"

    manifest_path, stats = preprocess_dataset(
        subset_dir,
        output_dir,
        min_len=0,
        max_workers=1,
        file_index=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chains = manifest["chains"]

    assert chains, "expected at least one chain to be preprocessed"
    assert stats.get("files_total", 0) == len(_SUBSET_FILES)
    assert stats.get("chains_written", 0) == len(chains)

    dataset = BackboneDataset(output_dir, k=4, progress=False)
    dataset_samples = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata = sample["metadata"]
        assert isinstance(metadata, dict)
        key = (metadata.get("path"), metadata.get("chain_id"))
        dataset_samples[key] = sample

    for entry in chains:
        h5_path = output_dir / entry["h5_path"]
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as handle:
            seq = _decode_sequence(handle["seq"][()])
            coords = handle["N_CA_C_O_coord"][()]
            plddt = handle["plddt_scores"][()]

        expected = preprocess_structure(entry["source_path"], entry["chain_id"])
        assert seq == expected.protein_seq
        np.testing.assert_allclose(coords, expected.coords, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(plddt, expected.plddt, atol=1e-6, equal_nan=True)

        key = (entry["source_path"], entry["chain_id"])
        sample = dataset_samples.get(key)
        assert sample is not None, f"missing dataset sample for {key}"
        assert sample["seq_str"] == expected.protein_seq
        np.testing.assert_allclose(
            sample["plddt_scores"].numpy(), expected.plddt, atol=1e-6, equal_nan=True
        )
        metadata = sample["metadata"]
        assert isinstance(metadata, dict)
        assert metadata.get("source_h5", "").endswith(entry["h5_path"])


def _make_training_config(output_dir: Path, data_root: Path) -> dict:
    return {
        "data": {
            "root": str(data_root),
            "k": 4,
            "num_dataloader_workers": 0,
            "cache": True,
        },
        "model": {
            "gcp": {
                "embedding": {
                    "node_scalar_dim": 6,
                    "node_vector_dim": 3,
                    "edge_scalar_dim": 8,
                    "edge_vector_dim": 1,
                    "output": {"scalar": 32, "vector": 4},
                },
                "message_passing": {"width": {"scalar": 32, "vector": 4}},
                "feed_forward": {"width": {"scalar": 64, "vector": 4}},
                "latent_dim": 64,
                "num_layers": 2,
            },
            "vq": {
                "num_codes": 32,
                "dim": 64,
                "beta": 0.25,
                "decay": 0.99,
                "kmeans_iters": 1,
            },
            "encoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 2,
            },
            "decoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 1,
            },
        },
        "train": {
            "seed": 777,
            "amp": False,
            "clip_grad": 1.0,
            "random_rotation": False,
            "checkpoint_interval": 1,
            "output_dir": str(output_dir),
            "log": {"interval": 1},
            "export": {"enabled": False},
            "stages": [
                {
                    "name": "pilot",
                    "length_cap": 128,
                    "batch_size": 1,
                    "base_lr": 0.001,
                    "min_lr": 1e-5,
                    "warmup_steps": 1,
                    "epochs": 1,
                    "accumulation_steps": 1,
                    "nan_mask_prob": 0.0,
                }
            ],
        },
    }


def test_training_pipeline_uses_preprocessed_hdf5(tmp_path: Path) -> None:
    subset_dir = _prepare_subset(tmp_path, filenames=["1A2N.cif", "1ACX.cif"])
    processed_root = tmp_path / "processed"
    preprocess_dataset(subset_dir, processed_root, min_len=0, max_workers=1, file_index=False)

    output_dir = tmp_path / "runs"
    config = _make_training_config(output_dir, processed_root)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train(str(config_path))

    checkpoint_dir = output_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert checkpoints, "training did not emit any checkpoints"
