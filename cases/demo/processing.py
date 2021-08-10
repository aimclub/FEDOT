from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fedot api
from fedot.api.main import Fedot

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def prepare_unimodal_data(time_series: Union[np.array, pd.Series], forecast_length: int):
    """ Prepare data for time series forecasting

    :param time_series: array with univariate time series
    :param forecast_length: length of forecast
    """
    time_series = np.array(time_series)

    # Define task
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    # Wrapp data into InputData
    input_data = InputData(idx=np.arange(0, len(time_series)),
                           features=time_series,
                           target=time_series,
                           task=task,
                           data_type=DataTypesEnum.ts)

    return input_data


def prepare_multimodal_data(dataframe: pd.DataFrame, features: list, target: str,
                            forecast_length: int):
    """ Prepare MultiModal data for time series forecasting task

    :param dataframe: pandas DataFrame to process
    :param features: columns, which should be used as features in forecasting
    :param target: name of target column
    :param forecast_length: length of forecast
    TODO implement function
    """
    raise NotImplementedError()


def automl_through_api(train_input, predict_input, timeout: Union[int, float], vis=True):
    """ Running AutoML algorithm for identification and configuration of pipeline """

    composer_params = {'max_depth': 4,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 20,
                       'timeout': timeout,
                       'preset': 'light_tun',
                       'metric': 'rmse',
                       'cv_folds': 2,
                       'validation_blocks': 2}

    model = Fedot(problem='ts_forecasting',
                  composer_params=composer_params)

    # Run AutoML model design in the same way
    obtained_chain = model.fit(features=train_input)

    if vis is True:
        obtained_chain.show()

    # Use model to obtain forecast
    forecast = model.predict(features=predict_input)

    return forecast


def plot_diesel_and_wind(df):
    """ Function for visualisation dataframe """

    # Gap values are equal to -1.0
    diesel = np.array(df['diesel_fuel_kWh'])
    wind = np.array(df['wind_power_kWh'])

    plt.plot(df['datetime'], diesel, label='Diesel fuel')
    plt.plot(df['datetime'], wind, label='Wind power')
    plt.ylabel('Electricity generation, kilowatts per hour', fontsize=12)
    plt.xlabel('Datetime', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.grid()
    plt.show()
