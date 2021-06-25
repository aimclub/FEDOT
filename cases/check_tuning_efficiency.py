import os
import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from examples.ts_forecasting_tuning import get_complex_chain

warnings.filterwarnings('ignore')
np.random.seed(2020)


def convert_ts_into_input(array, forecast_len):
    """ Wrap time series into InputData block """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_len))

    input_data = InputData(idx=np.arange(0, len(array)), features=array,
                           target=array, task=task, data_type=DataTypesEnum.ts)

    return input_data


def get_linear_chain():
    node_lagged = PrimaryNode('lagged')
    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    chain = Chain(node_final)
    return chain


def get_non_linear_chain():
    node_lagged = PrimaryNode('lagged')
    node_final = SecondaryNode('treg', nodes_from=[node_lagged])
    chain = Chain(node_final)
    return chain


def get_chain(linear_chain):
    if linear_chain:
        chain = get_linear_chain()
    else:
        chain = get_non_linear_chain()

    return chain


def run_tuning_validation(ts, forecast_len, validation_blocks, linear_chain, vis):
    """ Experiment with time series tuning on several time series

    :param ts: one-dimensional numpy array - time series
    :param forecast_len: horizon for forecasting
    :param validation_blocks: number of validation blocks (in-sample forecasting)
    :param linear_chain: is it need to validate linear chain
    :param vis: is there a need to plot forecasts
    """

    horizon = forecast_len * validation_blocks

    # Part for training algorithm
    train_data = ts[:-horizon]
    test_data = ts[-horizon:]
    train_input = convert_ts_into_input(array=train_data, forecast_len=forecast_len)

    # Prepare entire time series for validation
    val_input = convert_ts_into_input(array=ts, forecast_len=forecast_len)

    chain = get_chain(linear_chain)
    chain_tuner = ChainTuner(chain=chain, task=train_input.task, iterations=10)
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=mean_squared_error,
                                         loss_params={'squared': False})

    # Fit chain on the entire dataset
    tuned_chain.fit(train_input)
    tuned_forecast = in_sample_ts_forecast(chain=tuned_chain, input_data=val_input,
                                           horizon=horizon)

    new_chain = get_chain(linear_chain)
    new_chain.fit(train_input)
    old_forecast = in_sample_ts_forecast(chain=new_chain, input_data=val_input,
                                         horizon=horizon)

    if vis:
        plt.plot(range(0, len(ts)), ts, label='Actual values')
        plt.plot(range(len(train_data), len(ts)), old_forecast, label='Before tuning')
        plt.plot(range(len(train_data), len(ts)), tuned_forecast, label='After tuning')
        plt.legend()
        plt.grid()
        plt.title(f'Forecast length was {forecast_len} and blocks {validation_blocks}')
        plt.show()

    # Calculate metrics
    mae_before = mean_absolute_error(test_data, old_forecast)
    mae_after = mean_absolute_error(test_data, tuned_forecast)

    return mae_before, mae_after


def launch(series_name, launches, time_series_type, chain_type, goods, equals, ts,
           forecast_len, validation_blocks, linear_chain=True, vis=False):
    """ Produce calculations for time series and logging results into variables
    'time_series_type', 'chain_type', 'goods', 'equals' lists
    """
    number_good_performs = 0
    number_equals = 0
    for i in range(0, launches):
        mae_before, mae_after = run_tuning_validation(ts, forecast_len, validation_blocks, linear_chain=linear_chain, vis=vis)

        if mae_before > mae_after:
            number_good_performs += 1
        elif mae_before == mae_after:
            number_equals += 1

        print(f'The number of launches when the tuning exceeded the initial chain: {number_good_performs}/{launches}')
        print(f'The number of launches when the tuning was equal to the initial chain: {number_equals}/{launches}')

    time_series_type.append(series_name)
    if linear_chain:
        chain_type.append('linear')
    else:
        chain_type.append('non-linear')
    goods.append(number_good_performs)
    equals.append(number_equals)

    return time_series_type, chain_type, goods, equals


if __name__ == '__main__':
    launches = 20

    time_series_type = []
    chain_type = []
    goods = []
    equals = []

    ##############################
    #   Sea height forecasting   #
    ##############################
    data_path = os.path.join(f'{fedot_project_root()}', 'notebooks', 'data', 'ts_sea_level.csv')
    df = pd.read_csv(data_path)
    time_series = np.array(df['Level'])

    ########################
    #     LINEAR CHAIN     #
    ########################
    time_series_type, chain_type, goods, equals = launch('Sea height', launches,
                                                         time_series_type, chain_type,
                                                         goods, equals, time_series, forecast_len=150,
                                                         validation_blocks=4, linear_chain=True, vis=False)

    ########################
    #   NON LINEAR CHAIN   #
    ########################
    time_series_type, chain_type, goods, equals = launch('Sea height', launches,
                                                         time_series_type, chain_type,
                                                         goods, equals, time_series, forecast_len=150,
                                                         validation_blocks=4, linear_chain=False, vis=False)
    ###############################
    #   Temperature forecasting   #
    ###############################
    df = pd.read_csv('../cases/data/time_series/temperature.csv')
    time_series = np.array(df['value'])

    ########################
    #     LINEAR CHAIN     #
    ########################
    time_series_type, chain_type, goods, equals = launch('Temp', launches,
                                                         time_series_type, chain_type,
                                                         goods, equals, time_series, forecast_len=200,
                                                         validation_blocks=40, linear_chain=True, vis=False)

    ########################
    #   NON LINEAR CHAIN   #
    ########################
    time_series_type, chain_type, goods, equals = launch('Temp', launches,
                                                         time_series_type, chain_type,
                                                         goods, equals, time_series, forecast_len=200,
                                                         validation_blocks=40, linear_chain=False, vis=False)

    ###############################
    #     Traffic forecasting     #
    ###############################
    df = pd.read_csv('../cases/data/time_series/traffic.csv')
    time_series = np.array(df['value'])

    ########################
    #     LINEAR CHAIN     #
    ########################
    time_series_type, chain_type, goods, equals = launch('Temp', launches,
                                                         time_series_type, chain_type,
                                                         goods, equals, time_series, forecast_len=200,
                                                         validation_blocks=40, linear_chain=True, vis=False)

    ########################
    #   NON LINEAR CHAIN   #
    ########################
    time_series_type, chain_type, goods, equals = launch('Temp', launches,
                                                         time_series_type, chain_type,
                                                         goods, equals, time_series, forecast_len=200,
                                                         validation_blocks=40, linear_chain=False, vis=False)

    df = pd.DataFrame({'Time series': time_series_type, 'Chain type': chain_type,
                       'Tuning exceeded default': goods, 'Tuning was equal to default': equals})
    df.to_csv(f'{launches}_check.csv', index=False)
