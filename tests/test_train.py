from pathlib import Path

import gemmi
import yaml

from gcpvqvae.data.dataset import BackboneDataset
from gcpvqvae.data.preprocessing import preprocess_dataset
from gcpvqvae.system.train import Trainer, train


def _build_toy_structure(path):
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)

    model = gemmi.Model("0")
    chain = gemmi.Chain("A")
    for idx, offset in enumerate((0.0, 3.8, 7.6), start=1):
        residue = gemmi.Residue()
        residue.name = "GLY"
        residue.het_flag = " "
        residue.seqid = gemmi.SeqId(str(idx))

        atom_n = gemmi.Atom()
        atom_n.name = "N"
        atom_n.pos = gemmi.Position(offset, 0.0, 0.0)
        atom_n.occ = 1.0
        residue.add_atom(atom_n)

        atom_ca = gemmi.Atom()
        atom_ca.name = "CA"
        atom_ca.pos = gemmi.Position(offset + 1.2, 0.5, (-1) ** idx * 0.3)
        atom_ca.occ = 1.0
        residue.add_atom(atom_ca)

        atom_c = gemmi.Atom()
        atom_c.name = "C"
        atom_c.pos = gemmi.Position(offset + 2.4, 0.1, 0.6)
        atom_c.occ = 1.0
        residue.add_atom(atom_c)

        chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    doc = structure.make_mmcif_document()
    doc.write_file(str(path))


def test_training_harness_runs_single_stage(tmp_path):
    data_path = tmp_path / "toy.cif"
    _build_toy_structure(data_path)

    output_dir = tmp_path / "runs"
    config = {
        "data": {
            "root": str(data_path),
            "k": 4,
            "num_dataloader_workers": 0,
            "cache": True,
        },
        "model": {
            "gcp": {
                "hidden_scalar_dim": 32,
                "hidden_vector_dim": 4,
                "edge_scalar_dim": 8,
                "edge_vector_dim": 1,
                "latent_dim": 64,
                "layers": 2,
            },
            "vq": {
                "num_codes": 64,
                "dim": 64,
                "beta": 0.25,
                "decay": 0.99,
                "kmeans_iters": 1,
            },
            "encoder": {
                "model_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 2,
            },
            "decoder": {
                "model_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 1,
            },
        },
        "train": {
            "seed": 123,
            "amp": False,
            "clip_grad": 1.0,
            "random_rotation": False,
            "checkpoint_interval": 1,
            "output_dir": str(output_dir),
            "log": {"interval": 1},
            "export": {"enabled": False},
            "stages": [
                {
                    "name": "test",
                    "length_cap": 512,
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

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    train(str(config_path))

    checkpoints = list((output_dir / "checkpoints").glob("*.pt"))
    assert checkpoints, "training did not produce checkpoints"


def test_training_with_preprocessed_dataset(tmp_path):
    data_path = tmp_path / "toy.cif"
    _build_toy_structure(data_path)

    processed_root = tmp_path / "processed"
    preprocess_dataset(data_path, processed_root, k=4, progress=False)

    output_dir = tmp_path / "runs"
    config = {
        "data": {
            "root": str(processed_root),
            "k": 4,
            "num_dataloader_workers": 0,
            "cache": True,
        },
        "model": {
            "gcp": {
                "hidden_scalar_dim": 32,
                "hidden_vector_dim": 4,
                "edge_scalar_dim": 8,
                "edge_vector_dim": 1,
                "latent_dim": 64,
                "layers": 2,
            },
            "vq": {
                "num_codes": 64,
                "dim": 64,
                "beta": 0.25,
                "decay": 0.99,
                "kmeans_iters": 1,
            },
            "encoder": {
                "model_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 2,
            },
            "decoder": {
                "model_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 1,
            },
        },
        "train": {
            "seed": 321,
            "amp": False,
            "clip_grad": 1.0,
            "random_rotation": False,
            "checkpoint_interval": 1,
            "output_dir": str(output_dir),
            "log": {"interval": 1},
            "export": {"enabled": False},
            "stages": [
                {
                    "name": "test",
                    "length_cap": 512,
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

    config_path = tmp_path / "config_preprocessed.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    train(str(config_path))

    checkpoints = list((output_dir / "checkpoints").glob("*.pt"))
    assert checkpoints, "training did not produce checkpoints"


def test_training_on_cif_dataset_decreases_loss(tmp_path, monkeypatch):
    data_root = Path(__file__).resolve().parent / "test_data" / "cif_50"

    output_dir = tmp_path / "runs"
    config = {
        "data": {
            "root": str(data_root),
            "k": 4,
            "num_dataloader_workers": 0,
            "cache": True,
        },
        "model": {
            "gcp": {
                "hidden_scalar_dim": 16,
                "hidden_vector_dim": 4,
                "edge_scalar_dim": 8,
                "edge_vector_dim": 1,
                "latent_dim": 16,
                "layers": 2,
            },
            "vq": {
                "num_codes": 16,
                "dim": 16,
                "beta": 0.25,
                "decay": 0.99,
                "kmeans_iters": 1,
            },
            "encoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
            },
            "decoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
            },
        },
        "train": {
            "seed": 7,
            "amp": False,
            "clip_grad": 1.0,
            "random_rotation": False,
            "checkpoint_interval": None,
            "output_dir": str(output_dir),
            "log": {"interval": 1},
            "export": {"enabled": False},
            "stages": [
                {
                    "name": "test",
                    "length_cap": 128,
                    "batch_size": 2,
                    "base_lr": 0.001,
                    "min_lr": 1e-4,
                    "warmup_steps": 1,
                    "total_steps": 4,
                    "accumulation_steps": 1,
                    "nan_mask_prob": 0.0,
                }
            ],
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    losses = []
    original = Trainer._log_stage_progress

    def capture_progress(self, stage, stage_step, total_steps, trackers, samples, residues, elapsed):
        losses.append(trackers["loss"].average)
        return original(self, stage, stage_step, total_steps, trackers, samples, residues, elapsed)

    monkeypatch.setattr(Trainer, "_log_stage_progress", capture_progress)

    train(str(config_path))

    checkpoints = list((output_dir / "checkpoints").glob("*.pt"))
    assert checkpoints, "training did not produce checkpoints"
    assert len(losses) >= 2, "training did not log multiple loss values"
    assert losses[-1] < losses[0], "training loss did not decrease"


def test_training_with_eval_and_export(tmp_path, monkeypatch):
    data_root = Path(__file__).resolve().parent / "test_data" / "cif_50"

    output_dir = tmp_path / "runs"
    exports_dir = output_dir / "exports"
    config = {
        "data": {
            "root": str(data_root),
            "k": 4,
            "num_dataloader_workers": 0,
            "cache": True,
        },
        "model": {
            "gcp": {
                "hidden_scalar_dim": 16,
                "hidden_vector_dim": 4,
                "edge_scalar_dim": 8,
                "edge_vector_dim": 1,
                "latent_dim": 16,
                "layers": 2,
            },
            "vq": {
                "num_codes": 16,
                "dim": 16,
                "beta": 0.25,
                "decay": 0.99,
                "kmeans_iters": 1,
            },
            "encoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
            },
            "decoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
            },
        },
        "train": {
            "seed": 17,
            "amp": False,
            "clip_grad": 1.0,
            "random_rotation": False,
            "checkpoint_interval": 1,
            "output_dir": str(output_dir),
            "log": {"interval": 1},
            "export": {
                "enabled": True,
                "directory": str(exports_dir),
                "every_n_steps": 1,
                "on_stage_end": True,
                "num_samples": 1,
            },
            "eval": {
                "interval": 1,
                "root": str(data_root),
                "batch_size": 1,
                "num_dataloader_workers": 0,
                "length_cap": 128,
                "k": 4,
                "cache": True,
                "show_progress": False,
                "tm_score": False,
                "gdt_ts": False,
            },
            "stages": [
                {
                    "name": "stage",
                    "length_cap": 128,
                    "batch_size": 2,
                    "base_lr": 0.001,
                    "min_lr": 1e-4,
                    "warmup_steps": 1,
                    "total_steps": 2,
                    "accumulation_steps": 1,
                    "nan_mask_prob": 0.0,
                }
            ],
        },
    }

    config_path = tmp_path / "config_eval_export.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    eval_calls = []

    def fake_eval(model, dataloader, runtime_cfg, logger):
        eval_calls.append(len(dataloader))
        return {"rmsd": {"mean": 1.0, "median": 1.0}}

    monkeypatch.setattr("gcpvqvae.system.train.run_model_evaluation", fake_eval)

    export_calls = []

    def fake_export(model, dataset, export_cfg, stage_name, global_step, root_dir, logger):
        export_calls.append((stage_name, global_step, export_cfg.num_samples))
        export_index = len(export_calls)
        export_path = root_dir / "exports" / f"{stage_name}_{global_step}_{export_index}.mmcif"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text("test", encoding="utf-8")

    monkeypatch.setattr("gcpvqvae.system.train._export_samples", fake_export)

    train(str(config_path))

    checkpoints = sorted((output_dir / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 2, "expected checkpoints to be written at each step"
    assert len(eval_calls) >= 2, "evaluation should run at every step"
    assert all(count > 0 for count in eval_calls), "evaluation dataloader should have batches"
    assert len(export_calls) >= 3, "exports should occur during training and at stage end"
    export_files = sorted(exports_dir.glob("*.mmcif"))
    assert len(export_files) == len(export_calls), "export hook should create outputs"
    assert sum(1 for _, step, _ in export_calls if step == 2) >= 2, "stage end should trigger an additional export"


def test_multi_stage_training_tracks_global_step(tmp_path, monkeypatch):
    data_root = Path(__file__).resolve().parent / "test_data" / "cif_50"

    output_dir = tmp_path / "runs"
    stage2_batch_size = 8
    stage2_accumulation = 2

    dataset = BackboneDataset(
        data_root,
        k=4,
        length_cap=64,
        cache=True,
        progress=False,
    )
    batches_per_epoch = -(-len(dataset) // stage2_batch_size)
    expected_stage2_steps = -(-batches_per_epoch // stage2_accumulation)
    expected_total_steps = 1 + expected_stage2_steps

    config = {
        "data": {
            "root": str(data_root),
            "k": 4,
            "num_dataloader_workers": 0,
            "cache": True,
        },
        "model": {
            "gcp": {
                "hidden_scalar_dim": 16,
                "hidden_vector_dim": 4,
                "edge_scalar_dim": 8,
                "edge_vector_dim": 1,
                "latent_dim": 16,
                "layers": 2,
            },
            "vq": {
                "num_codes": 16,
                "dim": 16,
                "beta": 0.25,
                "decay": 0.99,
                "kmeans_iters": 1,
            },
            "encoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
            },
            "decoder": {
                "model_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
            },
        },
        "train": {
            "seed": 101,
            "amp": False,
            "clip_grad": 0.0,
            "random_rotation": False,
            "checkpoint_interval": None,
            "output_dir": str(output_dir),
            "log": {"interval": 0},
            "export": {"enabled": False},
            "stages": [
                {
                    "name": "warmup",
                    "length_cap": 64,
                    "batch_size": stage2_batch_size,
                    "base_lr": 5e-4,
                    "min_lr": 5e-5,
                    "warmup_steps": 0,
                    "total_steps": 1,
                    "accumulation_steps": 1,
                    "nan_mask_prob": 0.0,
                },
                {
                    "name": "finetune",
                    "length_cap": 64,
                    "batch_size": stage2_batch_size,
                    "base_lr": 5e-4,
                    "min_lr": 5e-5,
                    "warmup_steps": 0,
                    "epochs": 1,
                    "accumulation_steps": stage2_accumulation,
                    "nan_mask_prob": 0.0,
                },
            ],
        },
    }

    checkpoint_calls = []

    original_save_checkpoint = Trainer._save_checkpoint

    def capture_save_checkpoint(self, stage):
        checkpoint_path = self.output_dir / "checkpoints" / f"{stage.name}_step{self.global_step:06d}.pt"
        checkpoint_calls.append(checkpoint_path.name)
        original_save_checkpoint(self, stage)

    monkeypatch.setattr(Trainer, "_save_checkpoint", capture_save_checkpoint)

    trainer = Trainer(config)
    trainer.run()

    assert trainer.global_step == expected_total_steps

    assert len(checkpoint_calls) == 1, "final stage should emit a checkpoint"
    assert checkpoint_calls[0].startswith("finetune_")
    assert checkpoint_calls[0].endswith(f"{expected_total_steps:06d}.pt")
