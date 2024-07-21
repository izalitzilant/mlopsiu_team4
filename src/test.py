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

def read_datastore() -> Tuple[pd.DataFrame, str]:
    initialize(config_path="../configs", job_name="extract_data", version_base=None)
    cfg = compose(config_name="main")
    print("PATH", os.path.join(cfg.paths.root_path, 'data', 'samples', cfg.datasets.sample_filename))
    print("REPO", cfg.paths.root_path)
    url = dvc.api.get_url(
        path=os.path.join('data', 'samples', cfg.datasets.sample_filename),
        repo=os.path.join(cfg.paths.root_path),
        rev=str(cfg.datasets.version),
        remote=cfg.datasets.remote,
    )

    data = pd.read_csv(url)

    return data, str(cfg.datasets.version)

def dup():
    client = Client()
    list_of_artifacts = client.list_artifact_versions(name="features_target", tag='3.4b', sort_by="version").items
    list_of_artifacts.reverse()

    df1 = list_of_artifacts[0].load()

    list_of_artifacts = client.list_artifact_versions(name="features_target", tag='3.5b', sort_by="version").items
    list_of_artifacts.reverse()

    df2 = list_of_artifacts[0].load()

    print("DF 3.4", df1.shape)
    print(df1.iloc[:5, :5])
    
    print("\nDF 3.5", df2.shape)
    print(df2.iloc[:5, :5])
    
if __name__ == '__main__':
    dup()