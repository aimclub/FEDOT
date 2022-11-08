import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_ridge_smoothing_pipeline
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def prepare_data(forecast_length, is_multi_ts):
    """
    Function to form InputData from file with time-series
    """
    columns_to_use = ['61_91', '56_86', '61_86', '66_86']
    target_column = '61_91'
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    file_path = os.path.join(str(fedot_project_root()), 'cases/data/arctic/topaz_multi_ts.csv')
    if is_multi_ts:
        data = InputData.from_csv_multi_time_series(
            file_path=file_path,
            task=task,
            columns_to_use=columns_to_use)
    else:
        data = InputData.from_csv_time_series(
            file_path=file_path,
            task=task,
            target_column=target_column)
    train_data, test_data = train_test_data_setup(data)
    return train_data, test_data, task


def run_multi_ts_forecast(forecast_length, is_multi_ts):
    """
    Function for run experiment with use multi_ts data type (is_multi_ts=True) for train set extension
    Or run experiment on one time-series (is_multi_ts=False)
    """
    train_data, test_data, task = prepare_data(forecast_length, is_multi_ts)
    # init model for the time series forecasting
    init_pipeline = ts_complex_ridge_smoothing_pipeline()
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=5,
                  n_jobs=1,
                  max_depth=5,
                  num_of_generations=20,
                  pop_size=15,
                  max_arity=4,
                  cv_folds=None,
                  validation_blocks=None,
                  initial_assumption=init_pipeline
                  )
    # fit model
    pipeline = model.fit(train_data)
    pipeline.show()

    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)

    # visualize results
    model.plot_prediction()

    print(f'MAE: {mean_absolute_error(target, forecast)}')
    print(f'RMSE: {mean_squared_error(target, forecast)}')
    print(f'MAPE: {mean_absolute_percentage_error(target, forecast)}')

    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))


if __name__ == '__main__':
    forecast_length = 60
    run_multi_ts_forecast(forecast_length, is_multi_ts=True)
    run_multi_ts_forecast(forecast_length, is_multi_ts=False)
