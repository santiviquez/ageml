import pandas as pd
from sklearn.linear_model import LinearRegression

from ageml import TemporalDegradation


dataset_url = 'data/avocados_demand_forecasting_dataset.csv'
data = pd.read_csv(dataset_url)
data = data.drop(columns=['date'])
X = data
y = data[['demand']]

model = LinearRegression()

experiment = TemporalDegradation(
    timestamp_column_name='inference_time',
    target_column_name='demand',
    n_train_samples=30,
    n_test_samples=20,
    n_prod_samples=10,
    n_simulations=3)

X_train, X_test, X_prod, y_train, y_test, y_prod = experiment._train_test_prod_split(X, y)

assert len(X_train) == 30, "len(X_train) should be 30"
assert len(X_test) == 20, "len(X_test) should be 20"
assert len(X_prod) == 10, "len(X_prod) should be 10"

assert len(y_train) == 30, "len(y_train) should be 30"
assert len(y_test) == 20, "len(y_test) should be 20"
assert len(y_prod) == 10, "len(y_prod) should be 10"

results = experiment.run(data, model)

print(results)