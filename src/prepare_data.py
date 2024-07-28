import argparse
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

def prepare_data(args):
    data_preparation_pipeline(args.data_version)

    X, y = load_features(name='features_target', version=f'{args.data_version}c', target_col='deal_probability')
    assert isinstance(X, pd.DataFrame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", type=str, default="4.1", help="Data Version", required=True)
    args = parser.parse_args()
    prepare_data(args)