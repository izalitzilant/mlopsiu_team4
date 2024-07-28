import os
from io import StringIO
from unittest.mock import patch, MagicMock

import pandas as pd

import hydra
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from src.data import sample_data_local, preprocess_data, download_kaggle_dataset
from src.validate_data import validate_initial_dataset


def download_kaggle_test():
    with initialize(config_path="../configs", job_name="download_kaggle_test", version_base=None):
        cfg = compose(config_name="main")
        data_path = os.path.join(cfg.paths.root_path, 'data')
        download_kaggle_dataset(cfg, data_path)
        assert os.path.exists(os.path.join(data_path, 'train.csv'))
        data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 1000000

def test_download_dataset():
    sample = sample_data_local()

    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 10000

def validate_initial_data_test():
    validate_initial_dataset()

def preprocess_data_test():
    sample = sample_data_local()
    preprocessed_data = preprocess_data(sample, refit=False)
    assert isinstance(preprocessed_data, pd.DataFrame)
    assert len(preprocessed_data) == 10000
    assert preprocessed_data.shape[1] == 126
