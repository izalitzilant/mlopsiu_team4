import pandas as pd
from typing import Tuple, Any
from typing_extensions import Annotated

import zenml
from zenml import step, pipeline, ArtifactConfig
from zenml.client import Client

from data import read_datastore, preprocess_data
from validate_features import validate_features

@step(enable_cache=False)
def extract_data() -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="extracted_data", tags=["data_preparation"])],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])]
]:
    data, version = read_datastore()
    print("Dataset version", version, data.shape)
    return data, version

@step(enable_cache=False)
def transform_data(data: pd.DataFrame) -> Annotated[pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])]:
    return preprocess_data(data)

@step(enable_cache=False)
def validate_data(data: pd.DataFrame, version: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="valid_input_features", tags=["data_preparation"])]:
    return validate_features(data, version)

def load_features(data: pd.DataFrame, version: str) -> None:
    version = version + 'c'
    print("Saving preprocessed dataset", version, data.shape)
    zenml.save_artifact(data, "features_target", tags=[version])

    client = Client()
    list_of_artifacts = client.list_artifact_versions(name="features_target", tag=version, sort_by="version").items
    list_of_artifacts.reverse()

    df = list_of_artifacts[0].load()
    print(df.info())

@step(enable_cache=False)
def load(data: pd.DataFrame, version: str) -> None:
    load_features(data, version)

@pipeline
def data_preparation_pipeline():
    data, version = extract_data()
    data = transform_data(data)
    data = validate_data(data, version)
    load(data, version)

if __name__ == "__main__":
    run = data_preparation_pipeline()
