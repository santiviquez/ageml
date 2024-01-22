from importlib import resources
import pandas as pd

DATA_MODULE = "src.ageml.datasets.data"

def load_csv_file_to_df(local_file):
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    local_file : str
        The name of the local CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file.
    """
    with resources.path(DATA_MODULE, local_file) as data:
        return pd.read_csv(data)

def load_avocado_sales():
    """
    Load the avocado sales dataset into a Pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the avocado sales dataset.
    """
    dataset = load_csv_file_to_df("avocados_demand_forecasting_dataset.csv")
    return dataset