import os
from io import StringIO
from unittest.mock import patch, MagicMock

import hydra
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from src.data import sample_data, download_kaggle_dataset


def test_download_dataset_auth_raises():
    with pytest.raises(Exception):
        cfg = DictConfig({'kaggle' : {
                                        'username': 'username',
                                        'key': 'key',
        }})
        download_kaggle_dataset(cfg)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def get_cfg(cfg: DictConfig) -> DictConfig:
    ccfg = DictConfig('')
    return cfg


def test_download_dataset_download_successfully(capsys):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="main")
        download_kaggle_dataset(cfg)
        captured = capsys.readouterr()
        result = ('Downloading Kaggle dataset0\n'
                  'avito-demand-prediction\n'
                  'Downloading train.csv.zip to ../data\n'
                  '\n'
                  "Dataset for competition 'avito-demand-prediction' downloaded successfully!\n")
        assert captured.out == result


if __name__ == '__main__':
    cfg = get_cfg()
    print(cfg)
