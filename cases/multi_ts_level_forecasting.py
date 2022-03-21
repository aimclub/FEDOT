import os
import datetime
from copy import deepcopy
from typing import Any, List
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from examples.simple.time_series_forecasting.ts_pipelines import *
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.api.main import Fedot
from examples.advanced.time_series_forecasting.multi_ts_arctic_forecasting import prepare_data, initial_pipeline


def run_multi_ts_forecast(forecast_length, multi_ts):
    train_data, test_data, task = prepare_data(forecast_length, multi_ts)

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=10,
                  initial_assumption=initial_pipeline(),
                  composer_params={
                      'max_depth': 4,
                      'num_of_generations': 20,
                      'timeout': 10,
                      'pop_size': 10,
                      'max_arity': 3,
                      'available_operations': ['lagged', 'smoothing', 'diff_filter', 'gaussian_filter',
                                               'ridge', 'lasso', 'linear', 'cut']
                  })

    # fit model
    pipeline = model.fit(train_data)
    pipeline.show()
    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)

    # visualize results
    plt.plot(np.ravel(test_data.idx), np.ravel(test_data.target), label='test')
    plt.plot(np.ravel(train_data.idx), np.ravel(train_data.target[:, 0]), label='history')
    plt.plot(np.ravel(test_data.idx), forecast, label='prediction_after_tuning')
    plt.legend()
    plt.show()

    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))


if __name__ == '__main__':
    forecast_length = 60
    #run_multi_ts_forecast(forecast_length, multi_ts=True)
    run_multi_ts_forecast(forecast_length, multi_ts=False)
