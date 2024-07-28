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
from glob import glob
import shutil

def read_datastore() -> Tuple[pd.DataFrame, str]:
    with open(path, 'w') as f:
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

def move_to_mlartifacts():
    # move artifacts from mlruns to mlaritfacts folder, e.g. ./mlruns/0/0a025c23a34f40519b23ef4233def8e5/artifacts -> ./mlartifacts/0/0a025c23a34f40519b23ef4233def8e5/artifacts
    paths = glob(f"./mlruns/**/artifacts", recursive=True)
    print(f"Found {len(paths)} artifacts folders")
    for path in paths:
        full_path = os.path.abspath(path)
        new_path = full_path.replace("mlruns", "mlartifacts")
        print(f'Moving {full_path} to {new_path}')
        shutil.move(path, new_path)

def fix_mlartifacts_paths():
    # find all meta.yaml files and update the absolute path starting with "file:///mnt/c/mlopsiu_team4/mlruns/..." to "mlflow-artifacts:/..."

    paths = glob(f"./mlruns/**/meta.yaml", recursive=True)
    for path in paths:
        with open(path, 'r+') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "file:///mnt/c/mlopsiu_team4/mlruns/" in line:
                    new_line = line.replace("file:///mnt/c/mlopsiu_team4/mlruns/", "mlflow-artifacts:/")
                    lines[i] = new_line
                    print(f"Updated {line} to {new_line}")

            f.seek(0)
            f.writelines(lines)
            f.truncate()

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    dup()

    
if __name__ == '__main__':
    main()
