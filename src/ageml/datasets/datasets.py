from importlib import resources
import pandas as pd

DATA_MODULE = "ageml.datasets.data"

def load_csv_file_to_df(local_file):
    with resources.path(DATA_MODULE, local_file) as data:
        return pd.read_csv(data)

def load_avocado_sales():
    dataset = load_csv_file_to_df("avocados_demand_forecasting_dataset.csv")
    return dataset