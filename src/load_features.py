from zenml.client import Client
import numpy as np
import pandas as pd

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