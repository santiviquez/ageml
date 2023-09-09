import sys
sys.path.insert(1, './')
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from agingml import temporal_degradation_test as tdt
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import nannyml as nml

from agingml import performance_estimation_test as pet

dataset_url = 'data/yellow_trip_demand_forecasting_dataset.csv'
date_columns = ['date', 'inference_time']
categoric_columns = ['month', 'week', 'day_of_week', 'day', 'year', 'week_in_month']
dtype_categoric = dict([(c,'category') for c in categoric_columns])
# dataset = pd.read_csv(dataset_url , dtype=dtype_categoric, parse_dates=date_columns)
dataset = pd.read_csv(dataset_url)

dataset['timestamp'] = pd.to_datetime(dataset['inference_time'])
non_feature_cols = ['date', 'inference_time', 'demand']

data = dataset[dataset.columns[~dataset.columns.isin(non_feature_cols)]]
data = data.set_index('timestamp')
target = dataset['demand']
target.index = data.index


# Experiment set up
dataset = 'taxi'
n_train = 8800 # one year
n_test = 2200
n_prod = 4400
n_simulations = 1500
metric = mean_absolute_percentage_error
freq = 'D'
chunk_period='D'
models = [ElasticNet(), RandomForestRegressor(), LGBMRegressor(), MLPRegressor()]


for model in models:
    print(f'Running process for: {type(model).__name__}')
    aging_df = pd.read_parquet(f'results/aging/{dataset}/aging_{dataset}_{type(model).__name__}_{n_simulations}_simulations_{n_prod}_prod.parquet')
    aging_df.index = aging_df.index.rename('timestamp_index')

    # aging_df = aging_df[aging_df['model_age'] <= 800]
    test_aging_df = aging_df[aging_df['partition'] == 'test']
    test_mape = test_aging_df.groupby('simulation_id').apply(lambda group: mean_absolute_percentage_error(group.y, group.y_pred))
    test_mape = pd.DataFrame(test_mape, columns=['test_mape']).reset_index()
    d_aging_df = test_mape[test_mape['test_mape'] <= 0.1]

    valid_model_ids = d_aging_df['simulation_id'].drop_duplicates()
    aging_df = aging_df[aging_df['simulation_id'].isin(valid_model_ids)]

    pe_comparison, pe_result, realized_result = pet.evaluate_nannyml(data, aging_df, metric='mape', chunk_period=chunk_period)
    pe_comparison.to_parquet(f'results/performance_estimation/{dataset}/pe_comparison_{dataset}_{type(model).__name__}_{n_simulations}_simulations_{n_prod}_prod_{n_prod}_chunk{chunk_period}.parquet')
    for i in valid_model_ids:
        figure = pe_result[i].compare(realized_result[i]).plot()
        figure.write_image(f'figures/performance_estimation/{dataset}/pe_comparison_{dataset}_{type(model).__name__}_{n_simulations}_simulations_{n_prod}_prod_{i}_chunk{chunk_period}.svg', format='svg')

        figure = realized_result[i].plot()
        figure.write_image(f'figures/performance_estimation/{dataset}/realized_result_{dataset}_{type(model).__name__}_{n_simulations}_simulations_{n_prod}_prod_{i}_chunk{chunk_period}.svg', format='svg')