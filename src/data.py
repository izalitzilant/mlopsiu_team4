import os
import hydra
import zipfile

import pandas as pd
from omegaconf import DictConfig


def download_kaggle_dataset(cfg):
    print("Downloading Kaggle dataset0")
    os.environ['KAGGLE_USERNAME'] = cfg.kaggle.username
    os.environ['KAGGLE_KEY'] = cfg.kaggle.key
    import kaggle
    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(cfg.kaggle.competition_name)
    try:
        kaggle.api.competition_download_file(cfg.kaggle.competition_name, cfg.datasets.file_name,
                                             path=cfg.datasets.download_path)
        print(f"Dataset for competition '{cfg.kaggle.competition_name}' downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    for file in os.listdir(cfg.datasets.download_path):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(cfg.datasets.download_path, file), 'r') as zip_ref:
                zip_ref.extractall(cfg.datasets.download_path)
                os.remove(os.path.join(cfg.datasets.download_path, file))

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig = None) -> None:
    download_kaggle_dataset(cfg)

    data = pd.read_csv(f'{cfg.datasets.download_path}/{cfg.datasets.file_name}')

    with open(cfg.datasets.sample_index_path, 'r+') as f:
        # get the sample batch indexes
        start_idx = int(f.read())
        end_idx = int(len(data) * cfg.datasets.sample_size_frac) + start_idx
        # check if it is the end of the set
        if end_idx >= len(data):
            end_idx = len(data)
        sample_data = data.iloc[start_idx: end_idx]
        # rewrite the index file
        f.truncate(0)
        f.seek(0)
        # reset if the set is ended
        if end_idx == len(data):
            end_idx = 0
        f.write(f'{end_idx}')

    if not os.path.exists(cfg.datasets.sample_output_dir):
        os.makedirs(cfg.datasets.sample_output_dir)

    sample_data_path = str(f'{cfg.datasets.sample_output_dir}/{cfg.datasets.sample_filename}')
    sample_data.to_csv(sample_data_path, index=False)

    with open(cfg.datasets.sample_tag_path, 'r') as f:
        tag = f.read()

    os.popen(f'sh {cfg.datasets.save_sample_script_path} {cfg.datasets.sample_output_dir} {cfg.datasets.sample_filename} {tag}')

if __name__ == "__main__":
    sample_data()
