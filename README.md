
# ageML
ageML is a framework designed to study the temporal performance degradation of machine learning models. The goal of this project is to facilitate the exploration of performance degradation by providing tools that allow users to easily test how their models would evolve over time when trained and tested on different periods of their data.

Disclaimer: This project is still in its early stages, so the code interface might change in the future, and some elements might be hardcoded. However, the idea is to improve it over time, making it more user-friendly.


<p align="center">
 <img src="figures/aging/model_aging_plot_avocados_lr.png" alt="temporal degradation plot of lgbm regressor on taxi dataset" width="600"/>
</p>

## Features
Currently, this project implements one test to study the "aging" process that machine learning models can experience when in production due to covariate or concept shift.

### Temporal Degradation Test
Examines how various models perform when trained on different samples of the same dataset. This framework is based on the aging framework developed by [Vela et al.](https://www.nature.com/articles/s41598-022-15245-z) in 2022.

<p align="center">
 <img src="figures/temporal_degradadation_test.svg" alt="temporal degradation test" width="600"/>
</p>

### WIP: Continuous Retraining Test
Simulates a fixed-schedule retraining process of a machine learning model in production.
<p align="center">
 <img src="figures/continuous_retraining_test.svg" alt="continuous retraining test" width="600"/>
</p>


## Installation
The package hasn't been published on PyPI yet, which means you cannot install it via the regular Python channels. Instead, you'll have to clone the repository and install it from your local copy.

```bash
git clone https://github.com/santiviquez/ageml.git
cd ageml
pip install .
```

## Quickstart

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from ageml import TemporalDegradation

experiment = TemporalDegradation(
    timestamp_column_name='inference_time',
    target_column_name='demand',
    n_train_samples=52,
    n_test_samples=12,
    n_prod_samples=24,
    n_simulations=10)

experiment.run(data, model=LinearRegression())

experiment.plot(
    freq='W',
    metric=mean_absolute_error,
    min_test_error=1e7,
    plot_name='Model Ageing Chart: Avocado Sales Prediction - LinearRegression')

results = experiment.get_results(
    freq='W',
    metric=mean_absolute_error,
    min_test_error=1e7)

print(results)
```

## Quickstart
Check out the [issues page](https://github.com/santiviquez/ageml/issues) if you want to start building this with me 😊

## Author
- [santiviquez](https://www.twitter.com/santiviquez)

