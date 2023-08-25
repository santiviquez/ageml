import numpy as np
import pandas as pd
import nannyml as nml
from tqdm import tqdm


def evaluate_nannyml(data, aging_df, metric, chunk_period):
    simulation_ids = aging_df['simulation_id'].unique()
    nml_data = aging_df.merge(data, left_index=True, right_index=True, how='left')

    comparison_results = []
    pe_results = {}
    realized_results = {}
    constant_threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    
    for simulation_id in tqdm(simulation_ids):
        simulation_df = nml_data[nml_data['simulation_id'] == simulation_id]

        # get original reference set
        reference_df = simulation_df[simulation_df['partition'] == 'reference']

        # get original prod set
        analysis_df = simulation_df[simulation_df['partition'] == 'prod']

        # fit DLE from NannyML
        estimator = nml.DLE(
            feature_column_names=data.columns.tolist(),
            y_pred='y_pred',
            y_true='y',
            timestamp_column_name='timestamp',
            metrics=[metric],
            chunk_period=chunk_period,
            tune_hyperparameters=False,
            thresholds={'mape': constant_threshold}
        )

        estimator.fit(reference_df)
        
        # performance estimation results
        pe_result = estimator.estimate(analysis_df)
        
        # performance calculculator results
        calculator = nml.PerformanceCalculator(
            y_pred='y_pred',
            y_true='y',
            timestamp_column_name='timestamp',
            metrics=[metric],
            chunk_period=chunk_period,
            problem_type='regression'
        ).fit(reference_df)
        realized_result = calculator.calculate(analysis_df)

        pe_results[simulation_id] = pe_result
        realized_results[simulation_id] = realized_result
        
        comparison_result = pe_result.filter(period='analysis').to_df()[metric]
        comparison_result['estimated_alert'] = comparison_result['alert']
        comparison_result['realized_alert'] = np.where(comparison_result['realized'] > comparison_result['upper_threshold'], 
                                               True, False)

        comparison_result['simulation_id'] = simulation_id        
        comparison_results.append(comparison_result)
    
    
    return pd.concat(comparison_results), pe_results, realized_results