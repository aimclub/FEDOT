import warnings

import pandas as pd
import numpy as np

from copy import copy
from datetime import timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

warnings.filterwarnings('ignore')


def run_river_experiment(file_path):
    """ Function launch experiment for river level prediction. Tuner processes
    are available for such experiment.

    :param file_path: path to the file with river level data
    """
    # Last 350 values we will use for validation
    thr = 350

    # Read dataframe and prepare train and test data
    df = pd.read_csv(file_path)
    features = np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']])
    target = np.array(df['level_station_2'])

    task_regression = Task(TaskTypesEnum.regression)
    task_forecasting = Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(forecast_length=thr))

    # Divide into train and test
    features_train_reg = features[:-thr, :]
    target_train_reg = target[:-thr]
    features_test_reg = features[-thr:, :]
    target_test_reg = target[-thr:]

    # Data for regression part of the chain
    train_input_reg = InputData(idx=np.arange(0, len(features_train_reg)),
                                features=features_train_reg,
                                target=target_train_reg,
                                task=task_regression,
                                data_type=DataTypesEnum.table)

    start_forecast = len(features_train_reg)
    end_forecast = start_forecast + thr
    predict_input_reg = InputData(idx=np.arange(start_forecast, end_forecast),
                                  features=features_test_reg,
                                  target=target_test_reg,
                                  task=task_regression,
                                  data_type=DataTypesEnum.table)

    # Data for time series forecasting part of the chain
    target_train_ts = np.ravel(target_train_reg)
    target_test_ts = np.ravel(target_test_reg)

    train_input_ts = InputData(idx=np.arange(0, len(target_train_ts)),
                               features=target_train_ts,
                               target=target_train_ts,
                               task=task_forecasting,
                               data_type=DataTypesEnum.ts)
    predict_input_ts = InputData(idx=np.arange(start_forecast, end_forecast),
                                 features=target_train_ts,
                                 target=target_test_ts,
                                 task=task_forecasting,
                                 data_type=DataTypesEnum.ts)

    # Chain
    # one_hot_encoding -> dtreg  \
    #                             ridge
    #       lagged     -> linear /
    node_encoder = PrimaryNode('one_hot_encoding', node_data={'fit': train_input_reg,
                                                              'predict': predict_input_reg})
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_encoder])

    node_lagged = PrimaryNode('lagged', node_data={'fit': train_input_ts,
                                                   'predict': predict_input_ts})
    node_linear = SecondaryNode('linear', nodes_from=[node_lagged])

    node_final = SecondaryNode('ridge', nodes_from=[node_dtreg, node_linear])
    chain = Chain(node_final)

    # TODO finish this example


if __name__ == '__main__':
    run_river_experiment(file_path='../data/river_levels/station_levels.csv')
