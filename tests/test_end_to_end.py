"""End-to-end integration tests for the GCP-VQVAE pipeline."""

import shutil
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.models.gcpvqvae_model import GCPVQVAE


@pytest.fixture(scope="module")
def pilot_config():
    """Loads the pilot configuration."""
    with open("src/gcpvqvae/configs/pilot.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dummy_data_dir(tmp_path):
    """Creates a temporary directory with a dummy cif file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy("tests/data/dummy_A.cif", data_dir / "dummy_A.cif")
    return str(data_dir)


def test_training_smoke_test(pilot_config, dummy_data_dir):
    """
    Tests that a single training step can be completed without errors.
    This is a smoke test to ensure all components are connected correctly.
    """
    # Update config to use the temporary data directory
    pilot_config["data"]["root"] = dummy_data_dir
    pilot_config["train"]["total_steps"] = 2 # Only run for a couple of steps

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model and dataloader
    model = GCPVQVAE(pilot_config["model"]).to(device)
    model.train()

    dataset = BackboneDataset(**pilot_config["data"])
    dataloader = DataLoader(
        dataset,
        batch_size=2, # Use a batch size that is smaller than the dataset size
        collate_fn=collate_backbones
    )

    optimizer = torch.optim.AdamW(model.parameters())

    # Fetch one batch and run a training step
    try:
        batch = next(iter(dataloader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = model(batch)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    except Exception as e:
        pytest.fail(f"Training smoke test failed with an exception: {e}")


def test_encode_decode_roundtrip(pilot_config, tmp_path):
    """
    Tests that the encode and decode methods run without errors for mmCIF files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCPVQVAE(pilot_config["model"]).to(device)

    cif_path = "tests/data/dummy_A.cif"
    chain_id = "A"

    # Encode
    encode_result = model.encode(cif_path, chain_id)

    assert encode_result is not None
    assert "tokens" in encode_result
    assert encode_result["tokens"].ndim == 1
    assert encode_result["input_format"] == "cif"

    # Decode
    decode_result = model.decode(
        encode_result["tokens"],
        pose_header=encode_result["pose_header"]
    )

    assert decode_result is not None
    assert "coords" in decode_result

    # Check that the number of residues is consistent
    assert decode_result["coords"].shape[0] == encode_result["length"]
    assert decode_result["coords"].shape == (encode_result["length"], 3, 3)


def test_pdb_encode_decode_roundtrip(pilot_config, tmp_path):
    """
    Tests that the encode and decode methods run without errors for PDB files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCPVQVAE(pilot_config["model"]).to(device)

    pdb_path = "tests/data/dummy_A.pdb"
    chain_id = "A"

    # Encode
    encode_result = model.encode(pdb_path, chain_id)

    assert encode_result is not None
    assert "tokens" in encode_result
    assert encode_result["tokens"].ndim == 1
    assert encode_result["input_format"] == "pdb"

    # Decode
    decode_result = model.decode(
        encode_result["tokens"],
        pose_header=encode_result["pose_header"]
    )

    assert decode_result is not None
    assert "coords" in decode_result

    # Check that the number of residues is consistent
    assert decode_result["coords"].shape[0] == encode_result["length"]
    assert decode_result["coords"].shape == (encode_result["length"], 3, 3)
