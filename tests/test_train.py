import gemmi
import yaml

from gcpvqvae.system.train import train


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
            "num_workers": 0,
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
            "log_interval": 1,
            "checkpoint_interval": 1,
            "output_dir": str(output_dir),
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
