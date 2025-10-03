from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterable

import gemmi
import h5py
import numpy as np
import pytest

from gcpvqvae.data.reference.preprocessing import (
    PreprocessedChain,
    _validate_length,
    _validate_missing_thresholds,
)
from gcpvqvae.data.reference_preprocessing import preprocess_reference_dataset

_EXPECTED_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "test_data" / "reference_h5_expected.py"
)
_EXPECTED_SPEC = importlib.util.spec_from_file_location(
    "reference_h5_expected", _EXPECTED_FIXTURE_PATH
)
if _EXPECTED_SPEC is None or _EXPECTED_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load fixture module at {_EXPECTED_FIXTURE_PATH}")
_EXPECTED_MODULE = importlib.util.module_from_spec(_EXPECTED_SPEC)
_EXPECTED_SPEC.loader.exec_module(_EXPECTED_MODULE)
VALID_CHAIN_A = _EXPECTED_MODULE.VALID_CHAIN_A
MISSING_CHAIN_A = _EXPECTED_MODULE.MISSING_CHAIN_A


def _make_chain(mask: Iterable[bool]) -> PreprocessedChain:
    mask_array = np.asarray(list(mask), dtype=bool)
    length = int(mask_array.shape[0])
    coords = np.zeros((length, 4, 3), dtype=np.float64)
    plddt = np.full((length,), 80.0, dtype=np.float64)

    for idx, missing in enumerate(mask_array):
        if missing:
            coords[idx, :, :] = np.nan
            plddt[idx] = np.nan
        else:
            base = float(idx)
            coords[idx, 0] = (base, 0.0, 0.0)
            coords[idx, 1] = (base + 1.2, 0.1, 0.0)
            coords[idx, 2] = (base + 2.3, 0.2, 0.0)
            coords[idx, 3] = (base + 3.5, 0.3, 0.0)

    sequence = "A" * length
    missing_residues = int(mask_array.sum())
    return PreprocessedChain(sequence, coords, plddt, missing_residues)


def _add_atom(residue: gemmi.Residue, name: str, position, occ: float = 1.0) -> None:
    atom = gemmi.Atom()
    atom.name = name
    atom.pos = gemmi.Position(*position)
    atom.occ = occ
    atom.b_iso = 50.0
    residue.add_atom(atom)


def _write_structure(
    path: Path,
    *,
    num_residues: int,
    chain_ids: Iterable[str] = ("A",),
    missing_ca: Iterable[int] | None = None,
) -> None:
    missing_set = set(missing_ca or [])

    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    for chain_offset, chain_id in enumerate(chain_ids):
        chain = gemmi.Chain(chain_id)
        for idx in range(num_residues):
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.het_flag = " "
            residue.seqid = gemmi.SeqId(str(idx + 1))

            base = 3.8 * (idx + chain_offset * num_residues)
            _add_atom(residue, "N", (base, 0.0, 0.0))
            if idx not in missing_set:
                _add_atom(residue, "CA", (base + 1.2, 0.0, 0.0))
            _add_atom(residue, "C", (base + 2.4, 0.0, 0.0))
            _add_atom(residue, "O", (base + 3.5, 0.0, 0.0))

            chain.add_residue(residue)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    doc = structure.make_mmcif_document()
    doc.write_file(str(path))


def _assert_h5_matches_expected(generated_path: Path, expected: dict[str, object]) -> None:
    with h5py.File(generated_path, "r") as generated:
        expected_keys = {"seq", "N_CA_C_O_coord", "plddt_scores"}
        assert set(generated.keys()) == expected_keys

        generated_seq = generated["seq"][()].decode("ascii")
        assert generated_seq == expected["seq"]

        generated_coords = generated["N_CA_C_O_coord"][()]
        reference_coords = np.asarray(expected["coords"], dtype=np.float64)
        assert np.array_equal(np.isnan(generated_coords), np.isnan(reference_coords))
        np.testing.assert_allclose(
            generated_coords,
            reference_coords,
            equal_nan=True,
        )

        generated_plddt = generated["plddt_scores"][()]
        reference_plddt = np.asarray(expected["plddt"], dtype=np.float64)
        assert np.array_equal(np.isnan(generated_plddt), np.isnan(reference_plddt))
        np.testing.assert_allclose(
            generated_plddt,
            reference_plddt,
            equal_nan=True,
        )


def test_validate_length_bounds() -> None:
    chain = _make_chain([False] * 10)
    ok, reason = _validate_length(len(chain.protein_seq), min_len=5, max_len=12)
    assert ok and reason is None

    ok, reason = _validate_length(len(chain.protein_seq), min_len=12, max_len=None)
    assert not ok and reason == "chains_too_short"

    ok, reason = _validate_length(len(chain.protein_seq), min_len=None, max_len=5)
    assert not ok and reason == "chains_too_long"


def test_validate_missing_thresholds() -> None:
    mask = [False] * 8 + [True, True]
    chain = _make_chain(mask)
    ok, reason, ratio, longest = _validate_missing_thresholds(chain)
    assert ok and reason is None
    assert pytest.approx(ratio) == 0.2
    assert longest == 2

    excessive_ratio_chain = _make_chain([False] * 7 + [True, True, True])
    ok, reason, ratio, longest = _validate_missing_thresholds(excessive_ratio_chain)
    assert not ok and reason == "missing_ratio_exceeded"
    assert pytest.approx(ratio) == 0.3
    assert longest == 3

    long_block_mask = [False] * 64 + [True] * 16
    long_block_chain = _make_chain(long_block_mask)
    ok, reason, ratio, longest = _validate_missing_thresholds(long_block_chain)
    assert not ok and reason == "missing_block_exceeded"
    assert pytest.approx(ratio) == 0.2
    assert longest == 16


def test_preprocess_reference_dataset_collects_stats(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    _write_structure(input_dir / "valid.cif", num_residues=10)
    _write_structure(input_dir / "short.cif", num_residues=4)
    _write_structure(input_dir / "ratio.cif", num_residues=10, missing_ca={7, 8, 9})
    _write_structure(
        input_dir / "block.cif",
        num_residues=80,
        missing_ca=set(range(10, 26)),
    )

    _write_structure(input_dir / "too_long.cif", num_residues=110)

    complex_path = input_dir / "complex.cif"
    _write_structure(complex_path, num_residues=6, chain_ids=["A", "B"])

    (input_dir / "invalid.cif").write_text("not a cif", encoding="utf-8")

    output_dir = tmp_path / "output"
    manifest_path, stats = preprocess_reference_dataset(
        input_dir,
        output_dir,
        max_len=90,
        min_len=5,
        max_workers=2,
        use_cif=True,
        file_index=True,
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_chains"] == 1
    assert len(manifest["chains"]) == 1
    entry = manifest["chains"][0]
    assert entry["chain_id"] == "A"
    assert entry["length"] == 10
    assert entry["missing_residues"] == 0
    assert entry["h5_path"].endswith(".h5")

    h5_path = output_dir / entry["h5_path"]
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as handle:
        assert set(handle.keys()) == {"seq", "N_CA_C_O_coord", "plddt_scores"}

    index_path = output_dir / "file_index.json"
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index_payload["files"]) == 7

    assert stats["files_total"] == 7
    assert stats["parsing_errors"] == 1
    assert stats["complexes"] == 1
    assert stats["chains_total"] == 5
    assert stats["chains_written"] == 1
    assert stats["chains_too_short"] == 1
    assert stats["chains_too_long"] == 1
    assert stats["missing_ratio_exceeded"] == 1
    assert stats["missing_block_exceeded"] == 1
    assert stats["missing_coordinates"] == 2
    assert stats["h5_processed"] == 1


def test_preprocess_reference_dataset_h5_matches_fixture(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_structure(input_dir / "valid.cif", num_residues=5)

    output_dir = tmp_path / "output"
    manifest_path, _ = preprocess_reference_dataset(
        input_dir,
        output_dir,
        use_cif=True,
        file_index=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["chains"][0]
    h5_path = output_dir / entry["h5_path"]
    _assert_h5_matches_expected(h5_path, VALID_CHAIN_A)


def test_preprocess_reference_dataset_h5_preserves_nans(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_structure(
        input_dir / "missing.cif",
        num_residues=10,
        missing_ca={8, 9},
    )

    output_dir = tmp_path / "output"
    manifest_path, _ = preprocess_reference_dataset(
        input_dir,
        output_dir,
        use_cif=True,
        file_index=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["chains"][0]
    assert entry["missing_residues"] == 2
    h5_path = output_dir / entry["h5_path"]
    _assert_h5_matches_expected(h5_path, MISSING_CHAIN_A)


def test_preprocess_reference_dataset_omits_index_when_requested(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_structure(input_dir / "simple.cif", num_residues=4)

    output_dir = tmp_path / "output"
    manifest_path, stats = preprocess_reference_dataset(
        input_dir,
        output_dir,
        use_cif=True,
        file_index=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["chains"][0]
    assert entry["h5_path"] == "simple_chain_id_A.h5"
    assert stats["h5_processed"] == 1
    assert not (output_dir / "file_index.json").exists()
