import os
import hydra
import zipfile

import pandas as pd
from omegaconf import DictConfig

import logging
import sys

def download_kaggle_dataset(cfg, path):
    print("Downloading Kaggle dataset")
    os.environ['KAGGLE_USERNAME'] = cfg.kaggle.username
    os.environ['KAGGLE_KEY'] = cfg.kaggle.key
    import kaggle
    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(cfg.kaggle.competition_name)
    try:
        kaggle.api.competition_download_file(cfg.kaggle.competition_name, 'train.csv',
                                             path=path)
        print(f"Dataset for competition '{cfg.kaggle.competition_name}' downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    for file in os.listdir(path):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(path, file), 'r') as zip_ref:
                zip_ref.extractall(path)
                os.remove(os.path.join(path, file))

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig = None) -> None:
    data_path = os.path.join(cfg.paths.root_path, 'data')
    data = pd.read_csv(os.path.join(data_path, 'train.csv'))

    if not os.path.exists(os.path.join(data_path, 'train.csv')) or cfg.kaggle.force_download:
        download_kaggle_dataset(cfg, data_path)

    sample_index_path = os.path.join(cfg.paths.root_path, cfg.datasets.sample_index_path)
    with open(sample_index_path, 'r+') as f:
        # get the sample batch indexes
        idx = int(f.read())
        start_idx = int(cfg.datasets.sample_size) * idx
        end_idx = int(cfg.datasets.sample_size) * (idx + 1)
        # check if it is the end of the set
        if end_idx >= len(data):
            end_idx = len(data)
        sample_data = data.iloc[start_idx:end_idx]

    samples_path = os.path.join(data_path, './samples/')
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    sample_data_path = str(os.path.join(samples_path, cfg.datasets.sample_filename))
    sample_data.to_csv(sample_data_path, index=False)

    return sample_data

if __name__ == "__main__":
    sample_data()
