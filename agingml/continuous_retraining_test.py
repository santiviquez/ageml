import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
from skmisc.loess import loess
import optuna
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
import warnings
from agingml import temporal_degradation_test as tdt

optuna.logging.set_verbosity(0)
warnings.simplefilter("ignore", category=ExperimentalWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


def sample_train_start_point(data, n_test, n_prod, n_retrainings):
    idx_train_start = np.random.randint(0, len(data) - (2 * n_test * n_retrainings + n_prod))
    
    return idx_train_start


def train_test_prod_split(data, target, min_n_train, idx_train_start, n_test, n_prod, increment):
    idx_train_end = idx_train_start + min_n_train + n_test * increment
    
    idx_test_start = idx_train_end
    idx_test_end = idx_test_start + n_test
    
    idx_prod_start = idx_test_end
    idx_prod_end = idx_prod_start + n_prod
    
    # split data
    X_train = data.iloc[idx_train_start:idx_train_end]
    X_test = data.iloc[idx_test_start:idx_test_end]
    X_prod = data.iloc[idx_prod_start:idx_prod_end]

    # split targets
    y_train = target.iloc[idx_train_start:idx_train_end]
    y_test = target.iloc[idx_test_start:idx_test_end]
    y_prod = target.iloc[idx_prod_start:idx_prod_end]
    
    return X_train, X_test, X_prod, y_train, y_test, y_prod


def compute_model_errors(data, target, model, min_n_train, idx_train_start, n_test, n_prod, retraining_id):
    # create random split
    X_train, X_test, X_prod, y_train, y_test, y_prod = train_test_prod_split(data, target, min_n_train,
                                                                             idx_train_start, n_test,
                                                                             n_prod, retraining_id)
    # find optimal hyperparmeters
    if retraining_id == 0:
        optimal_params = tdt.hyperparameter_opt(X_train, y_train, model, n_trials=25)
        model.set_params(**optimal_params)
    
    # train the model
    model.fit(X_train, y_train)

    # flag valid models
    y_test_pred = model.predict(X_test)
    test_error = mean_absolute_percentage_error(y_test, y_test_pred) # TODO: allow other metrics
    if test_error < 0.2: # TODO: pass as parameter
        is_model_valid = True
    else:
        is_model_valid = False
    
    # train on all available data before production
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    model.fit(X, y)
    y_train_pred = model.predict(X)
    y_test_pred = model.predict(X_test)
    y_prod_pred = model.predict(X_prod)

    y_train_dict = {}
    y_test_dict = {}
    y_prod_dict = {}
    
    y_train_dict['y'] = y
    y_train_dict['y_pred'] = y_train_pred
    y_train_dict['partition'] = 'train'
    
    y_test_dict['y'] = y_test
    y_test_dict['y_pred'] = y_test_pred
    y_test_dict['partition'] = 'test'
    
    y_prod_dict['y'] = y_prod
    y_prod_dict['y_pred'] = y_prod_pred
    y_prod_dict['partition'] = 'prod'
        
    return y_train_dict, y_test_dict, y_prod_dict, is_model_valid


def generate_model_errors_dataframe(y_train_dict, y_test_dict, y_prod_dict, retraining_id, 
                                    simulation_id, is_model_valid):
    train_results_df = pd.DataFrame(y_train_dict)
    test_results_df = pd.DataFrame(y_test_dict)
    prod_results_df = pd.DataFrame(y_prod_dict)

    results_df = pd.concat([train_results_df, test_results_df, prod_results_df])
    results_df['timestamp'] = results_df.index
    #errors_df = errors_df.rename(columns={"demand": "error"})

    # check this
    results_df['model_age'] = (results_df.index - train_results_df.index[-1]).days
    results_df['is_model_valid'] = is_model_valid
    results_df['retraining_id'] = retraining_id
    results_df['simulation_id'] = simulation_id
    
    return results_df


def continuos_retraining_runner(data, target, model, min_n_train, n_test, n_prod, n_simulations, n_retrainings):
    
    # n_retrainings = np.floor((len(data) - n_train - n_val - n_test) / n_val).astype(int)

    # empty error lists
    simulation_results = []
    for simulation_id in tqdm(range(n_simulations)):
        idx_train_start = sample_train_start_point(data, n_test, n_prod, n_retrainings)
        model.set_params(random_state=np.random.randint(0, n_simulations))
        retraining_results = []
        # for every simulation compute the models errors, append the errors to the error lists
        for retraining_id in range(n_retrainings):
            train_errors, val_errors, test_errors, is_model_valid = compute_model_errors(data, target,
                                                                                         model,
                                                                                         min_n_train,
                                                                                         idx_train_start,
                                                                                         n_test,
                                                                                         n_prod,
                                                                                         retraining_id)        
            # generate a single error dataframe
            retraining_result = generate_model_errors_dataframe(train_errors, val_errors, test_errors,
                                                        retraining_id, simulation_id, is_model_valid)
            retraining_results.append(retraining_result)

        simulation_result = pd.concat(retraining_results)
        simulation_results.append(simulation_result)
    
    results = pd.concat(simulation_results)
    return results


def aggregate_errors_data(errors_df, metric, freq='D', only_valid_models=True):
    # TODO: clean this function
    # This function aggregates data in time frequencies and also checks if a model is valid
    
    freq_errors_df = errors_df.groupby(['partition', 'simulation_id', 'retraining_id', 
                                        pd.Grouper(key='timestamp', freq=freq)]) \
                              .apply(lambda group: metric(group.y, group.y_pred)) \
                              .rename("error").reset_index().sort_values(['simulation_id', 'timestamp'])

    last_train_dates_df = freq_errors_df[freq_errors_df['partition'] == 'train'] \
                            .groupby(['simulation_id', 'retraining_id']) \
                            .agg(last_val_date=('timestamp', 'max')) \
                            .reset_index()

    freq_errors_df = pd.merge(freq_errors_df, last_train_dates_df, on=['simulation_id', 'retraining_id'],
                              how='left')
    freq_errors_df['model_age'] = (freq_errors_df['timestamp'] - 
                                   freq_errors_df['last_val_date']) / np.timedelta64(1, freq)
        
    model_validity_df = errors_df[['simulation_id', 'retraining_id', 'is_model_valid']].drop_duplicates()
    freq_errors_df = pd.merge(freq_errors_df, model_validity_df, on=['simulation_id', 'retraining_id'],
                              how='left')
    
    test_results_df = errors_df[errors_df['partition'] == 'test']
    test_error = metric(test_results_df['y'], test_results_df['y_pred'])
    
    relative_errors = freq_errors_df['error'] / test_error
    freq_errors_df['error_rel'] = relative_errors
    
    if only_valid_models:    
        freq_errors_df = freq_errors_df[freq_errors_df['is_model_valid'] == True]
    
    return freq_errors_df


def get_trend_lines(data, quantiles, metric):
    trend_lines = []
    data = data[data['partition'] == 'prod']
    
    for q in quantiles:
        trend_line = {}
        q_df = data.groupby(['model_age'])[metric].agg(lambda x: x.quantile([q])).rename(metric).reset_index() 
        x = q_df['model_age']
        y = q_df[metric]

        l = loess(x, y, degree=2, span=0.1)
        l.fit()
        pred = l.predict(x, stderror=True)
        lowess = pred.values
        
        trend_line['quantile'] = q
        trend_line['model_age'] = x
        trend_line[metric] = y
        trend_lines.append(trend_line)
    
    trend_lines_df = pd.DataFrame(trend_lines)
    trend_lines_df = trend_lines_df.explode(['model_age', metric]).reset_index(drop=True)
    return trend_lines_df