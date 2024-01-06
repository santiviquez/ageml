import numpy as np
import pandas as pd
from tqdm import tqdm


class TemporalDegradation:
    def __init__(self,
                 timestamp_column_name: str,
                 target_column_name: str,
                 n_train_samples: int,
                 n_test_samples: int,
                 n_prod_samples: int,
                 n_simulations: int):

        self.timestamp_column_name = timestamp_column_name
        self.target_column_name = target_column_name
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.n_prod_samples = n_prod_samples
        self.n_simulations = n_simulations


    def _train_test_prod_split(self, X, y):
        
        self.X = X
        self.y = y

        # sample deplyoment point
        deployment_idx = np.random.randint(self.n_train_samples + self.n_test_samples, len(self.X) - self.n_prod_samples)
        idx_test_end = deployment_idx 
        idx_test_start = deployment_idx - self.n_test_samples
        
        idx_train_start = idx_test_start - self.n_train_samples
        idx_train_end = idx_test_start
        
        idx_prod_start = np.random.randint(deployment_idx, len(self.X) - self.n_prod_samples)
        idx_prod_end = idx_prod_start + self.n_prod_samples

        # reference will be use to fit NannyML algorithms
        # it should contain the test set + the latest data before the prod set
        # idx_reference_start = idx_test_start
        # idx_reference_end = idx_prod_start
        
        # split data
        X_train = self.X.iloc[idx_train_start:idx_train_end]
        X_test = self.X.iloc[idx_test_start:idx_test_end]
        X_prod = self.X.iloc[idx_prod_start:idx_prod_end]
        # X_reference = self.X.iloc[idx_reference_start:idx_reference_end]
        
        # split targets
        y_train = self.y.iloc[idx_train_start:idx_train_end]
        y_test = self.y.iloc[idx_test_start:idx_test_end]
        y_prod = self.y.iloc[idx_prod_start:idx_prod_end]
        # y_reference = self.y.iloc[idx_reference_start:idx_reference_end]

        return X_train, X_test, X_prod, y_train, y_test, y_prod

    def _compute_model_predictions(self, X_train, X_test, X_prod, y_train, y_test, y_prod, model):
        model.fit(X_train, y_train[self.target_column_name].values)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_prod_pred = model.predict(X_prod)

        y_train_dict = {f'{self.timestamp_column_name}': X_train.index, 'y':y_train[self.target_column_name].values, 'y_pred':y_train_pred, 'partition': 'train'}
        y_test_dict = {f'{self.timestamp_column_name}': X_test.index, 'y':y_test[self.target_column_name].values, 'y_pred':y_test_pred, 'partition': 'test'}
        y_prod_dict = {f'{self.timestamp_column_name}': X_prod.index, 'y':y_prod[self.target_column_name].values, 'y_pred':y_prod_pred, 'partition': 'prod'}

        train_results_df = pd.DataFrame(y_train_dict)
        test_results_df = pd.DataFrame(y_test_dict)
        prod_results_df = pd.DataFrame(y_prod_dict)

        results_df = pd.concat([train_results_df, test_results_df, prod_results_df])

        # this could be written in a nicer way
        last_train_timestamp = pd.to_datetime(train_results_df[self.timestamp_column_name].iloc[-1])
        results_df['model_age'] = (pd.to_datetime(results_df[self.timestamp_column_name]) - last_train_timestamp)#.days
        # results_df['simulation_id'] = simulation_id
        
        return results_df

    
    def run(self, data, model):
        
        self.data = data
        self.model = model

        self.data = self.data.set_index(self.timestamp_column_name)
        X = self.data.drop(columns=[self.target_column_name])
        y = self.data[[self.target_column_name]]

        results_df_list = []
        for i in tqdm(range(self.n_simulations)):

            X_train, X_test, X_prod, y_train, y_test, y_prod = self._train_test_prod_split(X, y)

            results_df = self._compute_model_predictions(X_train, X_test, X_prod, y_train, y_test, y_prod, model)
            results_df['simulation_id'] = i
            results_df_list.append(results_df)

        results_df = pd.concat(results_df_list)

        return results_df