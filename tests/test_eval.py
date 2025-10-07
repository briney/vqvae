from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gemmi
import torch
from torch import nn
from torch.utils.data import Dataset

import yaml
import pytest

from gcpvqvae.data.preprocess import preprocess_backbone_dataset as preprocess_dataset
from gcpvqvae.system import eval as eval_module


class FakeDataset(Dataset):
    def __init__(
        self,
        _root: str,
        *,
        chain_ids=None,
        length_cap: int = 0,
        k: int = 0,
        cache: bool = True,
        progress: bool = True,
        num_workers=None,
    ) -> None:
        del chain_ids, length_cap, k, cache, progress, num_workers
        self.samples: List[Dict[str, torch.Tensor]] = []

        coords_a = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        coords_b = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                [[2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        self.samples.append(
            {
                "coords": coords_a,
                "mask": torch.tensor([True, True], dtype=torch.bool),
                "atom_mask": torch.ones((2, 3), dtype=torch.bool),
            }
        )
        self.samples.append(
            {
                "coords": coords_b,
                "mask": torch.tensor([True, True, True], dtype=torch.bool),
                "atom_mask": torch.ones((3, 3), dtype=torch.bool),
            }
        )

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


def _build_eval_structure(path: Path) -> None:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    chain = gemmi.Chain("A")
    for idx in range(1, 3):
        residue = gemmi.Residue()
        residue.name = "GLY"
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))
        base = 3.8 * (idx - 1)

        atom_n = gemmi.Atom()
        atom_n.name = "N"
        atom_n.pos = gemmi.Position(base, 0.0, 0.0)
        atom_n.occ = 1.0
        residue.add_atom(atom_n)

        atom_ca = gemmi.Atom()
        atom_ca.name = "CA"
        atom_ca.pos = gemmi.Position(base + 1.2, 0.3, 0.0)
        atom_ca.occ = 1.0
        residue.add_atom(atom_ca)

        atom_c = gemmi.Atom()
        atom_c.name = "C"
        atom_c.pos = gemmi.Position(base + 2.3, 0.1, 0.0)
        atom_c.occ = 1.0
        residue.add_atom(atom_c)

        chain.add_residue(residue)
    model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    doc = structure.make_mmcif_document()
    doc.write_file(str(path))


def fake_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["coords"].shape[0] for item in batch)
    batch_size = len(batch)

    coords = torch.zeros((batch_size, max_len, 3, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    atom_mask = torch.zeros((batch_size, max_len, 3), dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["coords"].shape[0]
        coords[i, :length] = item["coords"]
        mask[i, :length] = item["mask"]
        atom_mask[i, :length] = item["atom_mask"]

    return {"coords": coords, "mask": mask, "atom_mask": atom_mask}


class FakeModel(nn.Module):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        self.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
        self.vq = type("VQ", (), {"num_codes": 4})()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        coords = batch["coords"].to(self.dummy.device)
        mask = batch["mask"].to(self.dummy.device)
        decoded = coords + 0.1
        indices = torch.where(
            mask,
            torch.zeros_like(mask, dtype=torch.long),
            torch.full_like(mask, -1, dtype=torch.long),
        )
        vq_metrics = {"perplexity": torch.tensor(2.0, device=self.dummy.device)}
        return {
            "decoded": decoded,
            "mask": mask,
            "valid_mask": mask,
            "indices": indices,
            "vq_metrics": vq_metrics,
        }


def test_evaluate_reports_summary(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "toy.cif"
    _build_eval_structure(raw_path)

    processed_dir = tmp_path / "processed"
    preprocess_dataset(raw_dir, processed_dir, k=1, progress=False)

    def dataset_factory(root, **kwargs):
        assert Path(root) == processed_dir
        return FakeDataset(str(root), **kwargs)

    monkeypatch.setattr(eval_module, "BackboneDataset", dataset_factory)
    monkeypatch.setattr(eval_module, "collate_backbones", fake_collate)
    monkeypatch.setattr(eval_module, "GCPVQVAE", FakeModel)

    checkpoint_path = tmp_path / "ckpt.pt"
    model = FakeModel()
    torch.save({"model": model.state_dict(), "config": {}}, checkpoint_path)

    config = {
        "data": {
            "root": str(processed_dir),
            "k": 1,
            "num_dataloader_workers": 0,
            "cache": False,
        },
        "model": {"checkpoint": str(checkpoint_path)},
        "eval": {"batch_size": 2, "tm_score": True, "gdt_ts": True, "histogram_bins": 5},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    summary = eval_module.evaluate(str(config_path))

    assert summary["num_chains"] == 2
    assert summary["num_residues"] == 5
    assert summary["codebook"]["num_codes"] == 4
    assert summary["codebook"]["active_codes"] == 1
    assert summary["codebook"]["utilization"] == 0.25
    assert summary["codebook"]["perplexity_mean"] == 2.0
    assert summary["rmsd"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert summary["tm_score"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert summary["gdt_ts"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert summary["length_vs_rmsd"]["slope"] == pytest.approx(0.0, abs=1e-6)
