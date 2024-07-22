from collections import defaultdict
import os
import random
from pathlib import Path
import importlib

import numpy as np
import sklearn.metrics
import torch
import zenml
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import open_dict
import sklearn

from model import load_features, train

def evaluate_model(gs, X_test, y_test, metrics_fns):
    predictions = gs.best_estimator_.predict(X_test)
    metrics = {}
    for metric_name, metric_fn in metrics_fns.items():
        metrics[metric_name] = metric_fn(y_test, predictions)
    
    return metrics

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    results_path = os.path.join(cfg.paths.root_path, "results")

    test_metrics = defaultdict(list)
    metrics_fns = {
        "MAE": sklearn.metrics.mean_absolute_error,
        "MSE": sklearn.metrics.mean_squared_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)),
    }
    
    print(f'Begin check for {cfg.model.model_name}')
    for seed in cfg.reproducibility.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cfg.random_state = seed
        cfg.model.params.module__seed = [seed]

        X_train, y_train = load_features(name="features_target", version=cfg.train_data_version, target_col=cfg.datasets.target_col)
        X_test, y_test = load_features(name="features_target", version=cfg.test_data_version, target_col=cfg.datasets.target_col)

        cfg.model.params.module__input_size = [X_train.shape[1]]
        gs = train(X_train, y_train, cfg=cfg)

        eval = evaluate_model(gs, X_test, y_test, metrics_fns)
        for metric_name, value in eval.items():
            test_metrics[metric_name].append(value)

    f = os.path.join(results_path, f"reproducibility_{cfg.model.model_name}.txt")
    with open(str(f), "w") as f:
        f.write(f"Test metrics for model {cfg.model.model_name}:\n")
        for metric_name, metrics in test_metrics.items():
            f.write(f"  {metric_name}: {metrics}\n")
            avg_metric = np.mean(metrics)
            var_metric = np.var(metrics)
            f.write(f"  Average {metric_name}: {avg_metric}\n")
            f.write(f"  Variance {metric_name}: {var_metric}\n\n")

if __name__ == "__main__":
    main()