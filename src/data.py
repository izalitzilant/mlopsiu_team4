import os
from typing import Tuple
import hydra
from hydra import compose, initialize
import zipfile
import numpy as np

import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
import dvc.api

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import TfidfVectorizer

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

def parse_version(version):
    major, minor = version.split(".")
    major, minor = int(major), int(minor)
    return major, minor

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig = None) -> None:
    data_path = os.path.join(cfg.paths.root_path, 'data')

    if not os.path.exists(os.path.join(data_path, 'train.csv')) or cfg.kaggle.force_download:
        download_kaggle_dataset(cfg, data_path)

    data = pd.read_csv(os.path.join(data_path, 'train.csv'))

    major, idx= parse_version(cfg.datasets.version)
    
    start_idx = int(cfg.datasets.sample_size) * idx
    end_idx = int(cfg.datasets.sample_size) * (idx + 1)
    if end_idx >= len(data):
        end_idx = len(data)
    sample_data = data.iloc[start_idx:end_idx]

    if end_idx == len(data):
        idx = 0
        major += 1
    else:
        idx += 1

    OmegaConf.update(cfg, 'datasets.version', f"{major}.{idx}")
    OmegaConf.update(cfg, 'datasets.message', f"Added sample data for version {major}.{idx}")
    OmegaConf.save({'datasets': cfg.datasets}, os.path.join(cfg.paths.root_path, 'configs', 'datasets.yaml'))

    samples_path = os.path.join(data_path, './samples/')
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    sample_data_path = str(os.path.join(samples_path, cfg.datasets.sample_filename))
    sample_data.to_csv(sample_data_path, index=False)

    return sample_data

def read_datastore() -> Tuple[pd.DataFrame, str]:
    initialize(config_path="../configs", job_name="extract_data", version_base=None)
    cfg = compose(config_name="main")
    url = dvc.api.get_url(
        path=os.path.join(cfg.paths.root_path, 'data', 'samples', cfg.datasets.sample_filename),
        repo=cfg.paths.root_path,
        rev=str(cfg.datasets.version),
        remote=cfg.datasets.remote,
    )

    data = pd.read_csv(url)

    return data, str(cfg.datasets.version)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop('image', axis=1)
    data = data.drop(['item_id', 'user_id'], axis=1)
    data['activation_date'] = pd.to_datetime(data['activation_date'])

    # Imputations
    data['param_1'] = data['param_1'].fillna('missing')
    data['param_2'] = data['param_2'].fillna('missing')
    data['param_3'] = data['param_3'].fillna('missing')
    data['description'] = data['description'].fillna('Нет описания')

    category_price_median = data.groupby('category_name')['price'].median()
    data['price'] = data['price'].fillna(data['category_name'].map(category_price_median))

    max_image_top_1 = data['image_top_1'].max()
    data['image_top_1'] = data['image_top_1'].fillna(max_image_top_1 + 1)

    # Collapse categories
    top_categories = data['category_name'].value_counts(normalize=True).cumsum()
    top_categories = top_categories[top_categories <= 0.75].index
    data['category_name'] = data['category_name'].apply(lambda x: x if x in top_categories else 'Other')

    top_cities = data['city'].value_counts(normalize=True).cumsum()
    top_cities = top_cities[top_cities <= 0.75].index
    data['city'] = data['city'].apply(lambda x: x if x in top_cities else 'Other')

    top_param_1 = data['param_1'].value_counts(normalize=True).cumsum()
    top_param_1 = top_param_1[top_param_1 <= 0.75].index
    data['param_1'] = data['param_1'].apply(lambda x: x if x in top_param_1 or x == 'missing' else 'Other')

    top_param_2 = data['param_2'].value_counts(normalize=True).cumsum()
    top_param_2 = top_param_2[top_param_2 <= 0.75].index
    data['param_2'] = data['param_2'].apply(lambda x: x if x in top_param_2 or x == 'missing' else 'Other')

    top_param_3 = data['param_3'].value_counts(normalize=True).cumsum()
    top_param_3 = top_param_3[top_param_3 <= 0.75].index
    data['param_3'] = data['param_3'].apply(lambda x: x if x in top_param_3 or x == 'missing' else 'Other')

    def params_length(x):
        if x == 'missing':
            return 0
        return len(x)

    data['params_length'] = data['param_1'].apply(params_length) + data['param_2'].apply(params_length)\
          + data['param_3'].apply(params_length)
    
    # Categories encoding
    categorical_cols = ['region', 'category_name', 'city', 'parent_category_name', 'user_type', 'param_1', 'param_2', 'param_3']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)

    data['title_length'] = data['title'].str.len()
    data['description_length'] = data['description'].str.len()

    # TF-IDF for title and description
    text_features = ['title', 'description']
    nltk.download('stopwords')
    russian_stop_words = set(stopwords.words("russian"))
    mystem = Mystem()

    def preprocess_text(text):
        tokens = mystem.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in russian_stop_words\
                  and token != " " \
                  and token.strip() not in punctuation]
        text = " ".join(tokens)
        return text

    for feature in text_features:
        tfidf = TfidfVectorizer(max_features=32, max_df=0.95, min_df=0.05)
        data[feature] = data[feature].apply(preprocess_text)
        tfidf_result = tfidf.fit_transform(data[feature])
        tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf.get_feature_names_out(), index=data.index)
        data = pd.concat([data, tfidf_df], axis=1)
        data = data.drop(feature, axis=1)

    # Date features
    def sin_cos_transform(x, max_val):
        return np.sin(2 * np.pi * x / max_val), np.cos(2 * np.pi * x / max_val)


    day_of_week = data['activation_date'].dt.dayofweek
    data['day_of_week_sin'], data['day_of_week_cos'] = sin_cos_transform(day_of_week, 7)

    day_of_month = data['activation_date'].dt.day
    data['day_of_month_sin'], data['day_of_month_cos'] = sin_cos_transform(day_of_month, data['activation_date'].dt.days_in_month)

    data = data.drop('activation_date', axis=1)
    
    # Scaling
    scaler = MinMaxScaler()
    numerical_cols = ['price', 'item_seq_number', 'image_top_1', 'title_length', 'description_length', 'params_length']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data
    
if __name__ == "__main__":
    sample_data()
