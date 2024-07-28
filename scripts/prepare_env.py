import hydra
from hydra import compose, initialize
import os
from typing import Tuple
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
import shutil

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print('old_path', cfg.paths.root_path)
    print('new_path', os.getcwd())

    new_root_path = os.getcwd()

    print(os.path.join(new_root_path, 'configs', 'paths.yaml'))
    OmegaConf.update(cfg, 'paths.root_path', new_root_path)
    OmegaConf.save({'paths': cfg.paths}, os.path.join(new_root_path, 'configs', 'paths.yaml'))
    print('Successfully configured paths!')

    kaggle_key = cfg.kaggle_key
    kaggle_username = cfg.kaggle_username

    shutil.copyfile(os.path.join(new_root_path, 'configs', 'kaggle.yaml.sample'),
                    os.path.join(new_root_path, 'configs', 'kaggle.yaml'))

    kaggle_conf = OmegaConf.load(os.path.join(new_root_path, 'configs', 'kaggle.yaml'))
    OmegaConf.update(kaggle_conf, 'kaggle.key', kaggle_key)
    OmegaConf.update(kaggle_conf, 'kaggle.username', kaggle_username)
    OmegaConf.save(kaggle_conf, os.path.join(new_root_path, 'configs', 'kaggle.yaml'))
    print('Successfully configured kaggle secrets')
    
if __name__ == '__main__':
    main()