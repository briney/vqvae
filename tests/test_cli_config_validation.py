from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from gcpvqvae.cli import gpcvq


def _write_config(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _base_config() -> dict:
    return {
        "data": {
            "root": "data/train",
            "k": 8,
            "num_dataloader_workers": 0,
            "cache": False,
        },
        "model": {
            "gcp": {
                "hidden_scalar_dim": 32,
                "hidden_vector_dim": 8,
                "edge_scalar_dim": 4,
                "edge_vector_dim": 4,
                "edge_vector_input_dim": 1,
                "latent_dim": 64,
                "layers": 2,
            },
            "vq": {
                "num_codes": 32,
                "dim": 64,
                "beta": 0.5,
                "decay": 0.95,
                "kmeans_iters": 1,
                "orthogonal_reg_weight": 0.0,
            },
            "encoder": {
                "model_dim": 128,
                "num_layers": 2,
                "num_heads": 8,
                "num_kv_heads": 2,
                "dropout": 0.0,
                "ffn_multiplier": 2.0,
            },
            "decoder": {
                "model_dim": 128,
                "num_layers": 2,
                "num_heads": 8,
                "num_kv_heads": 2,
                "dropout": 0.0,
                "ffn_multiplier": 2.0,
            },
            "rotation": {"input_dim": 128},
        },
        "train": {
            "stages": [
                {
                    "name": "stage1",
                    "length_cap": 128,
                    "batch_size": 4,
                    "base_lr": 0.001,
                    "min_lr": 0.00001,
                    "total_steps": 100,
                    "warmup_steps": 10,
                    "accumulation_steps": 1,
                }
            ]
        },
    }


def test_validate_config_reports_success(tmp_path):
    config = _base_config()
    path = _write_config(tmp_path / "valid.yaml", config)

    runner = CliRunner()
    result = runner.invoke(gpcvq, ["validate-config", str(path)])

    assert result.exit_code == 0, result.output
    assert "Status: VALID" in result.output
    assert "Model Parameters" in result.output
    assert "Vector quantiser" in result.output


def test_validate_config_flags_dimension_mismatch(tmp_path):
    config = _base_config()
    config["model"]["encoder"]["input_dim"] = 32  # inconsistent with gcp.latent_dim
    path = _write_config(tmp_path / "invalid.yaml", config)

    runner = CliRunner()
    result = runner.invoke(gpcvq, ["validate-config", str(path)])

    assert result.exit_code != 0
    assert "Status: INVALID" in result.output
    assert "model.encoder.input_dim" in result.output
    assert "does not match gcp.latent_dim" in result.output
