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
    df1, _ = load_features("features_target", "4.1c", "deal_probability")
    df2, _ = load_features("features_target", "4.2c", "deal_probability")

    print("DF 4.1", df1.shape)
    print(df1.iloc[:5, :5])
    
    print("\nDF 4.2", df2.shape)
    print(df2.iloc[:5, :5])

def get_model_version(model_name, model_alias):
    client = mlflow.MlflowClient()
    if model_alias is not None:
        model_version = client.get_model_version_by_alias(model_name, model_alias).version
        print(model_version)

    #model_version = f'{model_version}'
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri=model_uri)

    print(model)

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    dup()
    return
    print(cfg.paths.root_path)
    print('new_path', os.getcwd())

    new_root_path = os.getcwd()

    print(os.path.join(new_root_path, 'configs', 'paths.yaml'))
    #OmegaConf.update(cfg, 'datasets.message', f"Added sample data for version {major}.{idx}")
    #OmegaConf.save({'datasets': cfg.datasets}, os.path.join(cfg.paths.root_path, 'configs', 'datasets.yaml'))

    
if __name__ == '__main__':
    main()
    #get_model_version("MLP_4", "challenger1")
    #dup()