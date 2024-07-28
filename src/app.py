import os
import gradio as gr
import mlflow
from data import preprocess_data, read_datastore
import json
import requests
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose

def init_hydra():
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    return cfg

cfg = init_hydra()
raw_df, _ = read_datastore()
raw_df: pd.DataFrame = raw_df

# You need to define a parameter for each column in your raw dataset
def predict(region,
            city,
            parent_category_name,
            category_name,
            title,
            description,
            price,
            activation_date,
            user_type,
            item_seq_number=None,
            param_1 = None,
            param_2 = None,
            param_3 = None):
    
    # This will be a dict of column values for input data sample
    features = {
        "region": region,
        "city": city,
        "parent_category_name": parent_category_name,
        "category_name": category_name,
        "title": title,
        "description": description,
        "price": price,
        "item_seq_number": item_seq_number if item_seq_number else 1,
        "activation_date": activation_date,
        "user_type": user_type,
        "param_1": param_1 if param_1 and param_1 != '' else None,
        "param_2": param_2 if param_2 and param_2 != '' else None,
        "param_3": param_3 if param_3 and param_3 != '' else None,
        "image_top_1": None,
        "item_id": "dummy",
        "user_id": "dummy",
        "image": None
    }

    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    X = preprocess_data(raw_df, refit=False)
    
    # Convert it into JSON
    example = X.iloc[0,:]


    payload = json.dumps( 
        { "inputs": example.to_dict() }
    )

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:5001/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    if response.status_code != 200:
        return "Error: " + response.text

    pred = response.json()
    return f'{pred[0] * 100:.0f}%'

cities = raw_df['city'].unique().tolist()
regions = raw_df['region'].unique().tolist()
categories = raw_df['category_name'].unique().tolist()

examples_path = os.path.join(cfg.paths.root_path, 'data', 'examples')
if not os.path.exists(os.path.join(examples_path, 'log.csv')):
    os.makedirs(examples_path, exist_ok=True)
    cols = ['region', 'city', 'parent_category_name', 'category_name', 'title', 
                       'description', 'price', 'activation_date', 'user_type', 'item_seq_number', 
                       'param_1', 'param_2', 'param_3']
    raw_df = raw_df.dropna(subset=cols)
    raw_df.sample(10)[cols].to_csv(os.path.join(examples_path, 'log.csv'), index=False, header=False)
    
# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,

    title='Avito Advertisement Demand Prediction',
    
    # Here, the arguments in `predict` function
    # will be populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Dropdown(label="Region", choices=regions),
        gr.Dropdown(label="City", choices=cities),
        gr.Dropdown(label="Parent Category", choices=["Личные вещи", "Для дома и дачи", "Бытовая электроника", "Недвижимость", 
                                                       "Хобби и отдых", "Транспорт", "Услуги", "Животные", "Для бизнеса"]),
        gr.Dropdown(label="Category", choices=categories),
        gr.Text(label="Title"),
        gr.Text(label="Description"),   
        gr.Number(label="Price (in rubbles)", minimum=0), 
        gr.Text(label="Activation Date (YYYY-MM-DD)"),
        gr.Dropdown(label="User Type", choices=["Private", "Company", "Shop"]),
        gr.Number(label="Item Sequence Number (optional)", minimum=1),
        gr.Textbox(label="Parameter 1 (optional)"),
        gr.Textbox(label="Parameter 2 (optional)"),
        gr.Textbox(label="Parameter 3 (optional)"),
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="Deal Probability"),
    
    # This will provide the user with examples to test the API
    examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# Launch the web UI locally on port 5155
demo.launch(server_port=5155)