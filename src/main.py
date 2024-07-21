import random

import hydra
import numpy as np
import pandas as pd
import torch

from model import load_features, train, log_metadata


def run(args):
    cfg = args

    train_data_version = str(cfg.train_data_version)
    X_train, y_train = load_features(name="features_target", version=train_data_version, target_col=cfg.datasets.target_col)

    test_data_version = str(cfg.test_data_version)
    X_test, y_test = load_features(name="features_target", version=test_data_version, target_col=cfg.datasets.target_col)

    cfg.model.params.module__input_size = [X_train.shape[1]]

    gs = train(X_train, y_train, cfg=cfg)
    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()