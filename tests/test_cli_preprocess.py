from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner

from gcpvqvae.cli import gpcvq


def test_preprocess_dataset_help_lists_reference_options():
    runner = CliRunner()
    result = runner.invoke(gpcvq, ["preprocess-dataset", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--max-len",
        "--min-len",
        "--max-workers",
        "--use-cif",
        "--no-file-index",
        "--gap-threshold",
    ):
        assert flag in result.output


def test_preprocess_dataset_invokes_reference_driver(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    captured = {}

    def fake_driver(
        input_path: Path,
        output_path: Path,
        *,
        max_len,
        min_len,
        max_workers,
        use_cif,
        file_index,
        gap_threshold,
    ):
        captured["args"] = (input_path, output_path)
        captured["kwargs"] = {
            "max_len": max_len,
            "min_len": min_len,
            "max_workers": max_workers,
            "use_cif": use_cif,
            "file_index": file_index,
            "gap_threshold": gap_threshold,
        }
        manifest = output_path / "preprocessed_dataset.json"
        output_path.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        return manifest, Counter({"chains_written": 1})

    monkeypatch.setattr(
        "gcpvqvae.data.reference_preprocessing.preprocess_reference_dataset",
        fake_driver,
    )

    runner = CliRunner()
    result = runner.invoke(
        gpcvq,
        [
            "preprocess-dataset",
            str(input_dir),
            str(output_dir),
            "--max-len",
            "512",
            "--min-len",
            "64",
            "--max-workers",
            "8",
            "--use-cif",
            "--no-file-index",
            "--gap-threshold",
            "1.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == (input_dir, output_dir)
    assert captured["kwargs"] == {
        "max_len": 512,
        "min_len": 64,
        "max_workers": 8,
        "use_cif": True,
        "file_index": False,
        "gap_threshold": 1.5,
    }
    assert "Summary: Counter({'chains_written': 1})" in result.output
    assert "Preprocessed dataset written to" in result.output


@pytest.mark.parametrize(
    "args,message",
    [
        (("--min-len", "128", "--max-len", "64"), "--min-len cannot exceed"),
        (("--max-len", "0"), "--max-len must be a positive integer"),
        (("--min-len", "0"), "--min-len must be a positive integer"),
        (("--max-workers", "0"), "--max-workers must be a positive integer"),
        (("--gap-threshold", "0"), "--gap-threshold must be positive"),
    ],
)
def test_preprocess_dataset_reports_invalid_option_combinations(tmp_path, args, message):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        gpcvq,
        ["preprocess-dataset", str(input_dir), str(output_dir), *args],
    )

    assert result.exit_code != 0
    assert message in result.output
