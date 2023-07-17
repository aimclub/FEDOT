import logging

import pandas as pd

from examples.advanced.time_series_forecasting.multistep import TS_DATASETS
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

logging.raiseExceptions = False


def get_ts_data(dataset='australia', horizon: int = 30, validation_blocks=None):
    time_series = pd.read_csv(TS_DATASETS[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if dataset not in ['australia']:
        idx = pd.to_datetime(time_series['idx'].values)
    else:
        # non datetime indexes
        idx = time_series['idx'].values
    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input, validation_blocks=validation_blocks)
    return train_data, test_data


def run_ts_forecasting_example(dataset='australia', horizon: int = 30, validation_blocks=2, timeout: float = None,
                               visualization=False, with_tuning=True):
    train_data, test_data = get_ts_data(dataset, horizon, validation_blocks)
    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=Task(TaskTypesEnum.ts_forecasting,
                                   TsForecastingParams(forecast_length=horizon)).task_params,
                  timeout=timeout,
                  n_jobs=1,
                  with_tuning=with_tuning,
                  cv_folds=2, validation_blocks=validation_blocks, preset='fast_train')

    # run AutoML model design in the same way
    pipeline = model.fit(train_data)

    # use model to obtain two-step in-sample forecast
    in_sample_forecast = model.predict(test_data)
    print('Metrics for two-step in-sample forecast: ',
          model.get_metrics(metric_names=['rmse', 'mae', 'mape']))

    # plot forecasting result
    if visualization:
        pipeline.show()
        model.plot_prediction()

    # use model to obtain one-step forecast
    train_data, test_data = get_ts_data(dataset, horizon)
    simple_forecast = model.forecast(test_data)
    print('Metrics for one-step forecast: ',
          model.get_metrics(metric_names=['rmse', 'mae', 'mape']))
    if visualization:
        model.plot_prediction()

    # use model to obtain two-step out-of-sample forecast
    out_of_sample_forecast = model.forecast(test_data, horizon=20)
    # we can not calculate metrics because we do not have enough future values
    if visualization:
        model.plot_prediction()

    return in_sample_forecast, simple_forecast, out_of_sample_forecast


if __name__ == '__main__':
    run_ts_forecasting_example(dataset='salaries', horizon=10, timeout=10., visualization=True)
