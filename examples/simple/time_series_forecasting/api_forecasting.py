import logging
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root

logging.raiseExceptions = False

_TS_EXAMPLES_DATA_PATH = fedot_project_root().joinpath('examples/data/ts')

TS_DATASETS = {
    'm4_daily': _TS_EXAMPLES_DATA_PATH.joinpath('M4Daily.csv'),
    'm4_monthly': _TS_EXAMPLES_DATA_PATH.joinpath('M4Monthly.csv'),
    'm4_quarterly': _TS_EXAMPLES_DATA_PATH.joinpath('M4Quarterly.csv'),
    'm4_weekly': _TS_EXAMPLES_DATA_PATH.joinpath('M4Weekly.csv'),
    'm4_yearly': _TS_EXAMPLES_DATA_PATH.joinpath('M4Yearly.csv'),
    'australia': _TS_EXAMPLES_DATA_PATH.joinpath('australia.csv'),
    'beer': _TS_EXAMPLES_DATA_PATH.joinpath('beer.csv'),
    'salaries': _TS_EXAMPLES_DATA_PATH.joinpath('salaries.csv'),
    'stackoverflow': _TS_EXAMPLES_DATA_PATH.joinpath('stackoverflow.csv'),
    'test_sea': fedot_project_root().joinpath('test', 'data', 'simple_sea_level.csv')
}


def get_ts_data(dataset='m4_monthly', horizon: int = 30, m4_id=None, validation_blocks=None):
    time_series = pd.read_csv(TS_DATASETS[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if 'm4' in dataset:
        if not m4_id:
            label = random.choice(np.unique(time_series['label']))
        else:
            label = m4_id
        print(label)
        time_series = time_series[time_series['label'] == label]
        idx = time_series['datetime'].values
    else:
        label = dataset
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
    return train_data, test_data, label


def run_ts_forecasting_example(dataset='australia', horizon: int = 30, timeout: float = None,
                               visualization=False, validation_blocks=2, with_tuning=True):
    train_data, test_data, label = get_ts_data(dataset, horizon, validation_blocks=validation_blocks)
    # init model for the time series forecasting

    model = Fedot(problem='ts_forecasting',
                  task_params=Task(TaskTypesEnum.ts_forecasting,
                                   TsForecastingParams(forecast_length=horizon)).task_params,
                  timeout=timeout,
                  n_jobs=-1,
                  metric='mae',
                  with_tuning=with_tuning)

    model.fit(train_data)

    pred_fedot = model.forecast(test_data)
    if visualization:
        model.current_pipeline.show()
        plt.plot(train_data.idx, train_data.features, label='features')
        plt.plot(test_data.idx, test_data.target, label='target')
        plt.plot(test_data.idx, pred_fedot, label='fedot')
        plt.grid()
        plt.legend()
        plt.show()

    return pred_fedot


if __name__ == '__main__':
    run_ts_forecasting_example(dataset='m4_monthly', horizon=14, timeout=2., validation_blocks=None, visualization=True)
