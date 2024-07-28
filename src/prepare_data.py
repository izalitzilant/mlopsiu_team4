import mlflow
import hydra
from hydra import compose, initialize
import os
from typing import Tuple
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
import dvc.api
import zenml
from zenml import step, pipeline, ArtifactConfig
from zenml.client import Client
from load_features import load_features
from transform_data_local import data_preparation_pipeline

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def prepare_data(cfg=None):
    cfg.datasets.version = cfg.data_version
    data_preparation_pipeline()

    X, y = load_features()
    assert isinstance(X, pd.DataFrame)


if __name__ == "__main__":
    prepare_data()