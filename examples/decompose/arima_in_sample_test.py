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


def get_pipeline():
    node_arima = PrimaryNode('arima')
    return Pipeline(node_arima)

def get_pipeline_lagged():
    node_lagged = PrimaryNode('lagged')
    node_lasso = SecondaryNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_lasso)


def in_sample_fit_predict(pipeline, train_input, predict_input, horizon) -> np.array:
    """ Fit pipeline and make predictions (in-sample forecasting) """
    pipeline.fit(train_input)

    predicted_main = in_sample_ts_forecast(pipeline=pipeline,
                                           input_data=predict_input,
                                           horizon=horizon)
    return predicted_main


def time_series_into_input(len_forecast, train_part, time_series):
    """ Function wrap univariate time series into InputData """
    train_input, _, task = prepare_input_data(len_forecast=len_forecast,
                                              train_data_features=train_part,
                                              train_data_target=train_part,
                                              test_data_features=train_part)
    # Create data for validation
    predict_input = InputData(idx=range(0, len(time_series)),
                              features=time_series,
                              target=time_series,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input


def run_test():
    df = pd.read_csv('../../cases/data/lena_levels/multi_sample.csv')
    len_forecast = 500
    validation_blocks = 2
    time_series = np.array(df['stage_max_mean'][2500:])
    horizon = len_forecast * validation_blocks
    train_part = time_series[:-horizon]
    test_part = time_series[-horizon:]

    train_input, predict_input = time_series_into_input(len_forecast,
                                                        train_part,
                                                        time_series)

    pipeline = get_pipeline()
    predicted_values = in_sample_fit_predict(pipeline, train_input,
                                             predict_input, horizon)

    ids_for_test = range(len(train_part), len(time_series))
    plt.plot(time_series, label='Actual time series')
    plt.plot(ids_for_test, predicted_values, label='ARIMA')
    plt.legend()
    plt.show()

    pipeline = get_pipeline_lagged()
    predicted_values = in_sample_fit_predict(pipeline, train_input,
                                             predict_input, horizon)

    ids_for_test = range(len(train_part), len(time_series))
    plt.plot(time_series, label='Actual time series')
    plt.plot(ids_for_test, predicted_values, label='lagged')
    plt.legend()
    plt.show()

run_test()