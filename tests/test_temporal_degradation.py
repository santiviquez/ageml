import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from ageml import TemporalDegradation


dataset_url = 'data/avocados_demand_forecasting_dataset.csv'
data = pd.read_csv(dataset_url)
data = data.drop(columns=['date'])
X = data
y = data['demand']

model = LinearRegression()

experiment = TemporalDegradation(
    timestamp_column_name='inference_time',
    target_column_name='demand',
    n_train_samples=52,
    n_test_samples=12,
    n_prod_samples=24,
    n_simulations=10)

X_train, X_test, X_prod, y_train, y_test, y_prod = experiment._train_test_prod_split(X, y)

assert len(X_train) == 52, "len(X_train) should be 52"
assert len(X_test) == 12, "len(X_test) should be 12"
assert len(X_prod) == 24, "len(X_prod) should be 24"

assert len(y_train) == 52, "len(y_train) should be 52"
assert len(y_test) == 12, "len(y_test) should be 12"
assert len(y_prod) == 24, "len(y_prod) should be 24"

experiment.run(data, model)

experiment.plot(freq='W', metric=mean_absolute_error, min_test_score=1e7, plot_name='Avocados sales prediction degradation plot')

results = experiment.get_results(freq='W', metric=mean_absolute_error, min_test_score=1e7)
print(results)

# unprocessed results
raw_results = experiment.get_raw_results()
print(raw_results)