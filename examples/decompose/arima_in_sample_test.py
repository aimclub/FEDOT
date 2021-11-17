import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from examples.time_series.ts_forecasting_tuning import prepare_input_data
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from sklearn.metrics import mean_absolute_error
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def get_pipeline_arima():
    node_arima = PrimaryNode('arima')
    return Pipeline(node_arima)

def get_pipeline_lagged():
    node_lagged = PrimaryNode('lagged')
    node_lasso = SecondaryNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_lasso)


def wrap_into_input(forecast_length, time_series):
    """ Convert data for FEDOT framework """
    time_series = np.array(time_series)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    input_data = InputData(idx=np.arange(0, len(time_series)),
                           features=time_series, target=time_series,
                           task=task, data_type=DataTypesEnum.ts)

    return input_data


def run_test():
    df = pd.read_csv('../../cases/data/waves_mod.csv')
    test_size = 750
    variable = df['Hsig']
    train, test = variable[1: len(variable) - test_size], variable[len(variable) - test_size:]

    plt.plot(train)
    plt.plot(test)
    plt.show()

    input_data_short_train = wrap_into_input(forecast_length=2, time_series=train)
    input_data_short_test = wrap_into_input(forecast_length=2, time_series=variable)

    pipeline = get_pipeline_lagged()
    pipeline = pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                            loss_params=None, input_data=input_data_short_train,
                                            iterations=10, timeout=5,
                                            cv_folds=3, validation_blocks=20)
    pipeline.print_structure()
    pipeline.fit(input_data_short_train)
    short_val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                              input_data=input_data_short_test,
                                              horizon=test_size,
                                              force_fit=True)

    plt.plot(input_data_short_test.idx, input_data_short_test.target, label='Actual time series')
    plt.plot(np.arange(len(train), len(train) + len(short_val_predict)), short_val_predict,
             label='Forecast for 2 elements ahead')
    plt.legend()
    plt.show()

run_test()

'''def test_naive():
    df = pd.read_csv('../../cases/data/waves_mod.csv')
    test_size = 750
    variable = df['Hsig']
    train, test = variable[1: len(variable) - test_size], variable[len(variable) - test_size:]

    input_data_short_train = wrap_into_input(forecast_length=2, time_series=train)
    input_data_short_test = wrap_into_input(forecast_length=2, time_series=variable)

    pipeline = get_pipeline_arima()
    pipeline = pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                            loss_params=None, input_data=input_data_short_train,
                                            iterations=100, timeout=5,
                                            cv_folds=3, validation_blocks=20)
    pipeline.print_structure()
    pipeline.fit(input_data_short_train)
    prediction = []
    for i in range (0, 300):'''

