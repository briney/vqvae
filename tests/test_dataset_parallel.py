import shutil
from pathlib import Path

import pytest

from gcpvqvae.data.dataset import BackboneDataset
from gcpvqvae.system.eval import _prepare_data_config as prepare_eval_data_config
from gcpvqvae.system.train import (
    _prepare_data_config as prepare_train_data_config,
    _prepare_eval_during_training_config,
)


def test_backbone_dataset_parallel_loading(tmp_path):
    source_dir = Path("tests/test_data/cif_50")
    for name in ("1A2N.cif", "1ACX.cif"):
        shutil.copy(source_dir / name, tmp_path / name)

    parallel_dataset = BackboneDataset(
        tmp_path,
        cache=False,
        progress=False,
        length_cap=128,
        num_parsing_workers=2,
    )
    sequential_dataset = BackboneDataset(
        tmp_path,
        cache=False,
        progress=False,
        length_cap=128,
        num_parsing_workers=1,
    )

    assert len(parallel_dataset) == len(sequential_dataset)
    assert parallel_dataset[0]["seq"].shape == sequential_dataset[0]["seq"].shape


def test_prepare_data_config_records_parsing_workers(tmp_path):
    cfg = prepare_train_data_config(
        {
            "root": str(tmp_path),
            "num_parsing_workers": 3,
        }
    )
    assert cfg.num_parsing_workers == 3


def test_prepare_eval_config_records_parsing_workers(tmp_path):
    cfg = prepare_eval_data_config(
        {
            "root": str(tmp_path),
            "num_parsing_workers": 4,
        }
    )
    assert cfg.num_parsing_workers == 4


def test_prepare_eval_during_training_config_records_parsing_workers():
    cfg = _prepare_eval_during_training_config({"num_parsing_workers": 5})
    assert cfg is not None
    assert cfg.num_parsing_workers == 5
