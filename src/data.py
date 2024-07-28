import os
import pickle
from typing import Tuple

from sklearn.discriminant_analysis import StandardScaler
import hydra
from hydra import compose, initialize
import zipfile
import numpy as np

import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
import dvc.api
from sklearn.decomposition import PCA

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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

def sample_data_local(cfg: DictConfig = None) -> None:
    data_path = os.path.join(cfg.paths.root_path, 'data')
    samples_path = os.path.join(data_path, './samples/')
    sample_data = pd.read_csv(os.path.join(samples_path, cfg.datasets.sample_filename), parse_dates=["activation_date"])

    return sample_data

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig = None) -> None:
    data_path = os.path.join(cfg.paths.root_path, 'data')

    print('Data folder:', data_path)
    if not os.path.exists(os.path.join(data_path, 'train.csv')) or cfg.kaggle.force_download:
        print('Downloading dataset')
        download_kaggle_dataset(cfg, data_path)

    data = pd.read_csv(os.path.join(data_path, 'train.csv'))

    major, idx = parse_version(cfg.datasets.version)
    print(f'Dataset version {major}.{idx}')
    
    start_idx = int(cfg.datasets.sample_size) * idx
    end_idx = int(cfg.datasets.sample_size) * (idx + 1)
    if end_idx >= len(data):
        print('WARN: end_idx >= len(data)')
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

    print('Extract successful!')

    return sample_data

def read_datastore() -> Tuple[pd.DataFrame, str]:
    with initialize(config_path="../configs", job_name="extract_data", version_base=None):
        cfg = compose(config_name="main")
        url = dvc.api.get_url(
            path=os.path.join('data', 'samples', cfg.datasets.sample_filename),
            repo=cfg.paths.root_path,
            rev=str(cfg.datasets.version),
            remote=cfg.datasets.remote,
        )

        data = pd.read_csv(url)

        return data, str(cfg.datasets.version)
    
def read_datastore_local() -> Tuple[pd.DataFrame, str]:
    with initialize(config_path="../configs", job_name="extract_data", version_base=None):
        cfg = compose(config_name="main")
        data = sample_data_local(cfg)

        return data, str(cfg.datasets.version)
    
def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    return None

def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def preprocess_data(data: pd.DataFrame, refit=False) -> pd.DataFrame:
    with initialize(config_path="../configs", job_name="preprocess_data", version_base=None):
        cfg = compose(config_name="main")
        data = data.drop('image', axis=1)
        data = data.drop(['item_id', 'user_id'], axis=1)
        data['activation_date'] = pd.to_datetime(data['activation_date'])

        # Imputations
        data['param_1'] = data['param_1'].fillna('missing')
        data['param_2'] = data['param_2'].fillna('missing')
        data['param_3'] = data['param_3'].fillna('missing')

        def params_length(x):
            if x == 'missing':
                return 0
            return len(x)

        data['params_length'] = data['param_1'].apply(params_length) + data['param_2'].apply(params_length)\
            + data['param_3'].apply(params_length)

        data['description'] = data['description'].fillna('Нет описания')

        category_price_median = data.groupby('category_name')['price'].median()
        data['price'] = data['price'].fillna(data['category_name'].map(category_price_median))

        data['image_top_1'] = data['image_top_1'].fillna(3066 + 1)

        # Collapse categories
        top_categories = cfg.datasets.category_name
        data['category_name'] = data['category_name'].apply(lambda x: x if x in top_categories else 'Other')

        top_cities = cfg.datasets.city
        data['city'] = data['city'].apply(lambda x: x if x in top_cities else 'Other')

        top_param_1 = cfg.datasets.param_1
        data['param_1'] = data['param_1'].apply(lambda x: x if x in top_param_1 or x == 'missing' else 'Other')

        top_param_2 = cfg.datasets.param_2
        data['param_2'] = data['param_2'].apply(lambda x: x if x in top_param_2 or x == 'missing' else 'Other')

        top_param_3 = cfg.datasets.param_3
        data['param_3'] = data['param_3'].apply(lambda x: x if x in top_param_3 or x == 'missing' else 'Other')
        
        models_path = os.path.join(cfg.paths.root_path, 'models', 'transformers')
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        # Categories encoding
        categorical_cols = ['region', 'category_name', 'city', 'parent_category_name', 'user_type', 'param_1', 'param_2', 'param_3']
        ohe_path = os.path.join(models_path, 'onehotencoder.pkl')
        ohe = load_model(ohe_path) if not refit else None
        if ohe is None:
            ohe = OneHotEncoder(sparse_output=False, drop='first', dtype=int)
            ohe.fit(data[categorical_cols])
            save_model(ohe, ohe_path)
        encoded_categorical_cols = ohe.transform(data[categorical_cols])
        encoded_categorical_df = pd.DataFrame(encoded_categorical_cols, columns=ohe.get_feature_names_out(categorical_cols), index=data.index)
        data = pd.concat([data, encoded_categorical_df], axis=1)
        data = data.drop(categorical_cols, axis=1)

        data['title_length'] = data['title'].str.len()
        data['description_length'] = data['description'].str.len()

        # TF-IDF + PCA for title and description
        text_features = [('title', 8), ('description', 16)]
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

        for feature, pca_k in text_features:
            tfidf_path = os.path.join(models_path, f'tfidf_{feature}.pkl')
            pca_path = os.path.join(models_path, f'pca_{feature}.pkl')
            scaler_path = os.path.join(models_path, f'scaler_{feature}.pkl')

            
            data[feature] = data[feature].apply(preprocess_text)

            tfidf = load_model(tfidf_path) if not refit else None
            if tfidf is None:
                tfidf = TfidfVectorizer(max_features=128)
                tfidf.fit(data[feature])
                save_model(tfidf, tfidf_path)
            tfidf_result = tfidf.transform(data[feature])
            tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf.get_feature_names_out(), index=data.index)

            # PCA
            pca = load_model(pca_path) if not refit else None
            if pca is None:
                pca = PCA(n_components=pca_k)
                pca.fit(tfidf_df)
                save_model(pca, pca_path)
            pca_result = pca.transform(tfidf_df)

            scaler = load_model(scaler_path) if not refit else None
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(pca_result)
                save_model(scaler, scaler_path)
            pca_result = scaler.transform(pca_result)

            pca_df = pd.DataFrame(pca_result, columns=[f"{feature}_pca_{i}" for i in range(pca_k)], index=data.index)
            data = pd.concat([data, pca_df], axis=1)
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
        scaler_path = os.path.join(models_path, f'scaler_numerical.pkl')
        numerical_cols = ['price', 'item_seq_number', 'image_top_1', 'title_length', 'description_length', 'params_length']
        scaler = load_model(scaler_path) if not refit else None
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data[numerical_cols])
            save_model(scaler, scaler_path)
        data[numerical_cols] = scaler.transform(data[numerical_cols])

        # Sort columns for consistency
        data = data.reindex(sorted(data.columns), axis=1)

        return data
    
if __name__ == "__main__":
    sample_data()
