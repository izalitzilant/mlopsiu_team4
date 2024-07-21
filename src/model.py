import importlib
import os
import random

import mlflow
import numpy as np
import pandas as pd
from zenml.client import Client
import torch
from skorch.callbacks import BatchScoring
from skorch.regressor import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from models import RMSELoss

def load_features(name, version, target_col):
    client = Client()
    print("Loading features from", name, version)
    artifacts = client.list_artifact_versions(name=name, tag=version, sort_by="version").items
    print("Number of artifacts", len(artifacts))
    artifacts.reverse()

    df = artifacts[0].load()

    print("DF Shape", df.shape)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y

def plot_loss(model, name, cfg, run):
    train_losses = model.history[:, "train_loss"]
    valid_losses = model.history[:, "valid_loss"]

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{cfg.model.model_name} Loss")
    plt.legend()

    path = f'{name}.png'
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path, artifact_path=cfg.model.artifact_path)
    os.remove(path)

    mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path=f"{cfg.model.artifact_path}/{path}", dst_path="results")

def train(X_train, y_train, cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    class_instance = getattr(importlib.import_module(cfg.model.module_name), cfg.model.class_name)
    optimizer = torch.optim.AdamW
    estimator = NeuralNetRegressor(module=class_instance, optimizer=optimizer, verbose=0, 
                                   criterion=RMSELoss, batch_size=512)

    param_grid = dict(cfg.model.params)

    scoring = list(cfg.model.metrics.values())
    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=cfg.cv_n_jobs,
        refit=evaluation_metric,
        cv=cfg.model.folds,
        verbose=1,
        return_train_score=True,
    )

    X_train = X_train.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32).reshape(-1, 1)

    gs.fit(X_train, y_train)

    return gs

def retrieve_model_with_alias(model_name, model_alias = "champion") -> mlflow.pyfunc.PyFuncModel:

    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")

    # best_model
    return best_model

def retrieve_model_with_version(model_name, model_version = "v1") -> mlflow.pyfunc.PyFuncModel:

    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # best_model
    return best_model

def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    cv_results = (
        pd.DataFrame(gs.cv_results_)
        .filter(regex=r"std_|mean_|param_")
        .sort_index(axis=1)
    )
    best_metrics_values = [
        result[1][gs.best_index_] for result in gs.cv_results_.items()
    ]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {
        k: v
        for k, v in zip(best_metrics_keys, best_metrics_values)
        if "mean" in k or "std" in k
    }

    # print(cv_results, cv_results.columns)

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id  # type: ignore

    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join([cfg.run_name, cfg.model.model_name, cfg.model.evaluation_metric, str(params[cv_evaluation_metric]).replace(".", "_")])  # type: ignore
    print("run name: ", run_name)

    if mlflow.active_run():
        mlflow.end_run()

    # Fake run
    with mlflow.start_run():
        pass

    # Parent run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        plot_loss(gs.best_estimator_, 'champion_loss', cfg, run)
        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_train, targets=cfg.datasets.target_col)  # type: ignore
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_test, targets=cfg.datasets.target_col)  # type: ignore
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        # Infer the model signature
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.float32).reshape(-1, 1)
        X_test_np = X_test.values.astype(np.float32)
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train_np))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg.model.artifact_path,
            signature=signature,
            input_example=X_train.iloc[0].to_numpy(),
            registered_model_name=cfg.model.model_name,
            pyfunc_predict_fn=cfg.model.pyfunc_predict_fn,
        )

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(
            name=cfg.model.model_name,
            version=model_info.registered_model_version,
            key="source",
            value="best_Grid_search_model",
        )

        # Evaluate the best model
        predictions = gs.best_estimator_.predict(X_test_np)  # type: ignore
        eval_data = pd.DataFrame(y_test)
        eval_data.columns = ["label"]
        eval_data["predictions"] = predictions

        results = mlflow.evaluate(
            data=eval_data,
            model_type="regressor",  # Correct model type for regression
            targets="label",
            predictions="predictions",
            evaluators=["default"],
        )

        mlflow.log_metrics(results.metrics)

        print(f"Best model metrics:\n{results.metrics}")

        for index, result in cv_results.iterrows():

            child_run_name = "_".join(["child", run_name, str(index)])  # type: ignore
            with mlflow.start_run(
                run_name=child_run_name, experiment_id=experiment_id, nested=True
            ) as child_run:  # , tags=best_metrics_dict):
                ps = result.filter(regex="param_").to_dict()
                ms = result.filter(regex="mean_").to_dict()
                stds = result.filter(regex="std_").to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_", ""): v for (k, v) in ps.items()}
                if 'max_epochs' in ps:
                    ps['max_epochs'] = int(ps['max_epochs'])

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name  # e.g. "sklearn.linear_model"
                class_name = cfg.model.class_name  # e.g. "LogisticRegression"

                # Load "module.submodule.MyClass"
                class_instance = getattr(
                    importlib.import_module(module_name), class_name
                )

                optimizer = torch.optim.AdamW
                estimator = NeuralNetRegressor(
                    module=class_instance, optimizer=optimizer, 
                    criterion=RMSELoss, batch_size=512, **ps
                )

                estimator.fit(X_train_np, y_train_np)

                plot_loss(estimator, f'run{index}_loss', cfg, child_run)
                
                train_losses = estimator.history[:, "train_loss"]
                valid_losses = estimator.history[:, "valid_loss"]

                for idx, train_loss in enumerate(train_losses):
                    mlflow.log_metric("train_loss", train_loss, step=idx)
                for idx, valid_loss in enumerate(valid_losses):
                    mlflow.log_metric("valid_loss", valid_loss, step=idx)

                if child_run.info.artifact_uri:
                    print("Downloading run artifacts")
                    try:
                        path = os.path.join(cfg.paths.root_path, 'results', f'{cfg.model.model_name}_{child_run.info.run_id}_{str(index)}')
                        os.mkdir(path)
                        mlflow.artifacts.download_artifacts(artifact_uri=child_run.info.artifact_uri, 
                                                            dst_path=path)
                    except:
                        print('Download failed!')

                signature = mlflow.models.infer_signature(
                    X_train, estimator.predict(X_train_np)
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path=cfg.model.artifact_path,
                    signature=signature,
                    input_example=X_train.iloc[0].to_numpy(),
                    registered_model_name=cfg.model.model_name,
                    pyfunc_predict_fn=cfg.model.pyfunc_predict_fn,
                )


                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                predictions = loaded_model.predict(X_test_np)  # type: ignore

                eval_data = pd.DataFrame(y_test)
                eval_data.columns = ["label"]
                eval_data["predictions"] = predictions

                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="regressor",
                    targets="label",
                    predictions="predictions",
                    evaluators=["default"],
                )

                print(f"Metrics:\n{results.metrics}")
