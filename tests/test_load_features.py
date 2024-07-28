import os
from io import StringIO

import pandas as pd

import hydra
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from src.load_features import load_features
from src.validate_features import validate_features


def test_load_features():
    X, y = load_features(name='features_target', version='4.1c', target_col='deal_probability')
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 10000
    assert X.shape[1] == 125
    assert isinstance(y, pd.Series)
    assert len(y) == 10000