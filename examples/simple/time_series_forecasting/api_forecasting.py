import logging
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from examples.advanced.time_series_forecasting.multistep import TS_DATASETS
from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

logging.raiseExceptions = False


def get_ts_data(dataset='m4_monthly', horizon: int = 30, m4_id=None, validation_blocks=None):
    time_series = pd.read_csv(TS_DATASETS[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if not m4_id:
        label = random.choice(np.unique(time_series['label']))
    else:
        label = m4_id
    print(label)
    time_series = time_series[time_series['label'] == label]

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
                               visualization=False, with_tuning=True, validation_blocks=2):
    train_data, test_data, label = get_ts_data(dataset, horizon, validation_blocks=validation_blocks)
    # init model for the time series forecasting
    pipeline = PipelineBuilder().add_node('lagged') \
        .add_node('topological_features') \
        .add_node('lagged', branch_idx=1) \
        .join_branches('ridge').build()

    pipeline.fit(train_data)
    pred = np.ravel(pipeline.predict(test_data).predict)

    model = Fedot(problem='ts_forecasting',
                  task_params=Task(TaskTypesEnum.ts_forecasting,
                                   TsForecastingParams(forecast_length=horizon)).task_params,
                  timeout=2,
                  n_jobs=-1,
                  metric='mae',
                  with_tuning=True,
                  cv_folds=2)

    model.fit(train_data)

    pred_fedot = model.forecast(test_data)

    plt.plot(train_data.idx, train_data.features, label='features')
    plt.plot(test_data.idx, test_data.target, label='target')
    plt.plot(test_data.idx, pred, label='(lagged->topological_extractor)->ridge')
    plt.plot(test_data.idx, pred_fedot, label='fedot')
    plt.grid()
    plt.legend()
    plt.show()
    return None


if __name__ == '__main__':
    run_ts_forecasting_example(dataset='m4_monthly', validation_blocks=None, horizon=14, timeout=2., visualization=True)
