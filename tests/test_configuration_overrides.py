"""Tests for configuration override helpers."""

from pathlib import Path

import yaml

from gcpvqvae.system.configuration import compose_overrides


def _write_config(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_cli_overrides_update_logging_block(tmp_path):
    base = {
        "data": {"root": "data/train"},
        "model": {},
        "train": {
            "log": {"enabled": False, "project": None},
        },
    }

    path = _write_config(tmp_path / "config.yaml", base)

    overrides = ("train.log.enabled=true", "train.log.project=my-project")
    result = compose_overrides(path, overrides)

    assert result["train"]["log"]["enabled"] is True
    assert result["train"]["log"]["project"] == "my-project"
