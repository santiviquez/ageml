import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from skmisc.loess import loess
from tqdm import tqdm
import optuna
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import warnings



optuna.logging.set_verbosity(0)
warnings.simplefilter("ignore", category=ExperimentalWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


def train_test_prod_split(data, target, n_train, n_test, n_prod):
    # sample deplyoment point
    deployment_idx = np.random.randint(n_train + n_test, len(data) - n_prod)
    # deployment_idx = np.random.randint(n_train + n_test, len(data) - 2 * n_train)
    idx_test_end = deployment_idx 
    idx_test_start = deployment_idx - n_test
    
    idx_train_start = idx_test_start - n_train
    idx_train_end = idx_test_start
    
    idx_prod_start = np.random.randint(deployment_idx, len(data) - n_prod)
    # idx_prod_start = np.random.randint(deployment_idx, deployment_idx + 2 * n_train)
    idx_prod_end = idx_prod_start + n_prod

    # reference will be use to fit NannyML algorithms
    # it should contain the test set + the latest data before the prod set
    idx_reference_start = idx_test_start
    idx_reference_end = idx_prod_start
    
    # split data
    X_train = data.iloc[idx_train_start:idx_train_end]
    X_test = data.iloc[idx_test_start:idx_test_end]
    X_prod = data.iloc[idx_prod_start:idx_prod_end]
    X_reference = data.iloc[idx_reference_start:idx_reference_end]
    
    # split targets
    y_train = target.iloc[idx_train_start:idx_train_end]
    y_test = target.iloc[idx_test_start:idx_test_end]
    y_prod = target.iloc[idx_prod_start:idx_prod_end]
    y_reference = target.iloc[idx_reference_start:idx_reference_end]

    # TODO: return only 3 objects df_train, ... targets next to data
    return X_train, X_test, X_prod, X_reference, y_train, y_test, y_prod, y_reference


def hyperparameter_opt(X_train, y_train, model, n_trials):
    lgbm_params = {
        'num_leaves': optuna.distributions.IntDistribution(2, 2*12),
        'max_depth': optuna.distributions.IntDistribution(1, 13),
        'min_child_samples': optuna.distributions.IntDistribution(10, int(len(X_train) * 0.5)),
        'n_estimators': optuna.distributions.IntDistribution(100, 2000, 1),
        'learning_rate': optuna.distributions.FloatDistribution(0.0001, 0.3),
        'reg_alpha': optuna.distributions.FloatDistribution(0, 1000),
        'reg_lambda': optuna.distributions.FloatDistribution(0, 1000),
        'colsample_bytree': optuna.distributions.FloatDistribution(0, 1),
        'subsample': optuna.distributions.FloatDistribution(0, 1)
    }
    
    # TODO: create distributions for the other models
    elastic_net_params = {
        'alpha': optuna.distributions.FloatDistribution(0, 1000),
        'l1_ratio': optuna.distributions.FloatDistribution(0, 1),
        'max_iter': optuna.distributions.IntDistribution(1000, 2000)
    }
    random_forest_params = {
        'n_estimators': optuna.distributions.IntDistribution(100, 400, 1),
        'max_depth': optuna.distributions.IntDistribution(1, 13),
        'min_samples_split': optuna.distributions.IntDistribution(2, 10)

    }
    mlp_regressor_params = {
        'hidden_layer_sizes': optuna.distributions.IntDistribution(20, 150),
        'activation': optuna.distributions.CategoricalDistribution(['relu', 'tanh']),
        'solver': optuna.distributions.CategoricalDistribution(['lbfgs', 'sgd', 'adam']),
        'alpha': optuna.distributions.FloatDistribution(0.0001, 0.01),
        'learning_rate_init': optuna.distributions.FloatDistribution(0.0001, 0.01),
        'max_iter': optuna.distributions.IntDistribution(300, 1000)
    }

    if type(model).__name__ == 'LGBMRegressor':
        param_distributions = lgbm_params
    elif type(model).__name__ == 'ElasticNet':
        param_distributions = elastic_net_params
    elif type(model).__name__ == 'RandomForestRegressor':
        param_distributions = random_forest_params
    elif type(model).__name__ == 'MLPRegressor':
        param_distributions = mlp_regressor_params
        

    optuna_search = optuna.integration.OptunaSearchCV(
        model, param_distributions, n_trials=n_trials, verbose=0,
        cv=TimeSeriesSplit(n_splits=4), n_jobs=-1
    )

    optuna_search.fit(X_train, y_train)

    trial = optuna_search.study_.best_trial
    optimal_params = trial.params

    return optimal_params

def compute_model_errors(data, target, model, n_train, n_test, n_prod):
    # create random split
    X_train, X_test, X_prod, X_reference, y_train, y_test, y_prod, y_reference = train_test_prod_split(data, target, n_train, n_test, n_prod)
    
    # find optimal hyperparmeters
    optimal_params = hyperparameter_opt(X_train, y_train, model, n_trials=25)
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
    y_reference_pred = model.predict(X_reference)

    y_train_dict = {}
    y_test_dict = {}
    y_prod_dict = {}
    y_reference_dict = {}
    
    y_train_dict['y'] = y
    y_train_dict['y_pred'] = y_train_pred
    y_train_dict['partition'] = 'train'
    
    y_test_dict['y'] = y_test
    y_test_dict['y_pred'] = y_test_pred
    y_test_dict['partition'] = 'test'
    
    y_prod_dict['y'] = y_prod
    y_prod_dict['y_pred'] = y_prod_pred
    y_prod_dict['partition'] = 'prod'
    
    y_reference_dict['y'] = y_reference
    y_reference_dict['y_pred'] = y_reference_pred
    y_reference_dict['partition'] = 'reference'
        
    return y_train_dict, y_test_dict, y_prod_dict, y_reference_dict, is_model_valid


def generate_model_errors_dataframe(y_train_dict, y_test_dict, y_prod_dict, y_reference_dict, simulation_id, is_model_valid):
    train_results_df = pd.DataFrame(y_train_dict)
    test_results_df = pd.DataFrame(y_test_dict)
    prod_results_df = pd.DataFrame(y_prod_dict)
    reference_results_df = pd.DataFrame(y_reference_dict)

    results_df = pd.concat([train_results_df, test_results_df, reference_results_df, prod_results_df])
    results_df['timestamp'] = results_df.index
    #errors_df = errors_df.rename(columns={"demand": "error"})

    # check this
    results_df['model_age'] = (results_df.index - train_results_df.index[-1]).days
    results_df['is_model_valid'] = is_model_valid
    results_df['simulation_id'] = simulation_id
    
    return results_df


def evaluation_runner(data, target, model, n_train, n_test, n_prod, n_simulations=1):
    # empty error lists
    results_df_list = []
    
    # for every simulation compute the models errors, append the errors to the error lists
    for i in tqdm(range(n_simulations)):
        
        model.set_params(random_state=np.random.randint(0, n_simulations))
        y_train_results, y_test_results, y_reference_results, y_prod_results, is_model_valid = compute_model_errors(data, target, 
                                                                                                                    model, n_train, 
                                                                                                                    n_test, n_prod)
    
        # generate a single error dataframe
        results_df = generate_model_errors_dataframe(y_train_results, y_test_results, y_reference_results,
                                                     y_prod_results, i, is_model_valid)
        results_df_list.append(results_df)
    
    results_df = pd.concat(results_df_list)
    
    return results_df


def aggregate_errors_data(errors_df, metric, freq='D', only_valid_models=True):
    # TODO: clean this function
    # This function aggregates data in time frequencies and also checks if a model is valid
    
    freq_errors_df = errors_df.groupby(['partition', 'simulation_id', pd.Grouper(key='timestamp', freq=freq)]) \
                              .apply(lambda group: metric(group.y, group.y_pred)) \
                              .rename("error").reset_index().sort_values(['simulation_id', 'timestamp'])

    last_train_dates_df = freq_errors_df[freq_errors_df['partition'] == 'train'] \
                            .groupby(['simulation_id']) \
                            .agg(last_val_date=('timestamp', 'max')) \
                            .reset_index()

    freq_errors_df = pd.merge(freq_errors_df, last_train_dates_df, on='simulation_id', how='left')
    freq_errors_df['model_age'] = (freq_errors_df['timestamp'] - 
                                   freq_errors_df['last_val_date']) / np.timedelta64(1, freq)
        
    model_validity_df = errors_df[['simulation_id', 'is_model_valid']].drop_duplicates()
    freq_errors_df = pd.merge(freq_errors_df, model_validity_df, on='simulation_id', how='left')
    
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
        e = q_df['error']

        # l = loess(x, y, degree=2, span=0.05)
        # l.fit()
        # pred = l.predict(x, stderror=True)
        # lowess = pred.values
        
        trend_line['quantile'] = q
        trend_line['model_age'] = x
        trend_line['error'] = e
        # trend_line[metric] = lowess
        trend_lines.append(trend_line)
    
    trend_lines_df = pd.DataFrame(trend_lines)
    trend_lines_df = trend_lines_df.explode(['model_age', metric]).reset_index(drop=True)
    return trend_lines_df


def plot_aging_chart(aging_df, metric, freq, plot_name):

    trend_lines_df = get_trend_lines(data=aging_df, quantiles=[0.25, 0.50, 0.75], metric='error')

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(data=trend_lines_df, x='model_age', y='error', linewidth=1.5,
                palette=['#E8FF3A', 'black', '#FB4748'], hue='quantile', legend=False, ax=ax)

    sns.scatterplot(data=aging_df[aging_df['partition'] == 'prod'],
                    x='model_age', y='error', s=7, alpha=0.1, color='#3b0280', linewidth=0, ax=ax)

    ax.legend(title='Percentile', labels=['25th', 'Median', '75th'], loc='upper right')
    ax.set_xlabel(f'Model Age [{freq}]')
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(aging_df[aging_df['partition'] == 'prod']['error']))

    ax.set_title(plot_name)
    # plt.savefig(path, format='svg')
    # plt.show()

    return fig