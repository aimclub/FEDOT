from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# fedot api
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
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


def prepare_unimodal_for_validation(time_series: Union[np.array, pd.Series],
                                    forecast_length: int, validation_blocks: int):
    """ Prepare time series for in-sample forecasting """
    time_series = np.array(time_series)
    horizon = forecast_length * validation_blocks

    # Define task
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    train_part = time_series[:-horizon]

    # InputData for train
    train_input = InputData(idx=range(0, len(train_part)),
                            features=train_part,
                            target=train_part,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # InputData for validation
    validation_input = InputData(idx=range(0, len(time_series)),
                                 features=time_series,
                                 target=time_series,
                                 task=task,
                                 data_type=DataTypesEnum.ts)

    return train_input, validation_input


def automl_fit_forecast(train_input, predict_input, composer_params: dict,
                        vis=True, in_sample_forecasting=False, horizon: int = None):
    """ Running AutoML algorithm for identification and configuration of pipeline

    :param train_input: InputData for algorithm fitting
    :param predict_input: InputData for forecast
    :param composer_params: dictionary with hyperparameters
    :param vis: is there a need to display structure of obtained pipeline
    :param in_sample_forecasting: is it needed to make in_sample_forecasting
    :param horizon: forecast horizon for in sample forecasting
    (ignored if in_sample_forecasting is False)
    """

    model = Fedot(problem='ts_forecasting',
                  task_params=train_input.task.task_params,
                  composer_params=composer_params)

    # Run AutoML model design in the same way
    obtained_pipeline = model.fit(train_input)

    if vis is True:
        obtained_pipeline.print_structure()
        obtained_pipeline.show()

    # Use model to obtain forecast
    if in_sample_forecasting is True:
        # Perform in-sample forecast
        forecast = in_sample_ts_forecast(pipeline=obtained_pipeline,
                                         input_data=predict_input,
                                         horizon=horizon)
    else:
        forecast = model.predict(features=predict_input)

    return forecast, obtained_pipeline


def multi_automl_fit_forecast(train_input: dict, predict_input: dict,
                              composer_params: dict, target: np.array,
                              forecast_length: int, vis: bool = True,
                              verbose_level: int = 1):
    """ Multi modal forecasting

    :param train_input: dictionary with InputData classes for train
    :param predict_input: dictionary with InputData classes for test
    :param composer_params: dictionary with hyperparameters
    :param vis: is there a need to display structure of obtained pipeline
    :param target: numpy array (time series) for forecasting
    :param forecast_length: forecast length
    :param verbose_level: verbosity of logger
    """
    task_params = TsForecastingParams(forecast_length=forecast_length)
    model = Fedot(problem='ts_forecasting',
                  composer_params=composer_params,
                  task_params=task_params, verbose_level=verbose_level)
    # Run AutoML model design in the same way
    obtained_pipeline = model.fit(features=train_input, target=target)

    if vis:
        obtained_pipeline.show()
        obtained_pipeline.print_structure()

    forecast = model.predict(features=predict_input)

    return forecast, obtained_pipeline


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


def advanced_validation(predicted_values, forecast_length, validation_blocks,
                        source_time_series):
    """ Function for validation time series forecasts on several blocks """
    horizon = forecast_length * validation_blocks

    actual_values = np.ravel(source_time_series[-horizon:])
    pre_history = np.ravel(source_time_series[:-horizon])
    mse_metric = mean_squared_error(actual_values, predicted_values, squared=False)
    mae_metric = mean_absolute_error(actual_values, predicted_values)

    print(f'MAE - {mae_metric:.2f}')
    print(f'RMSE - {mse_metric:.2f}')

    # Plot time series forecasted
    plt.plot(range(0, len(source_time_series)), source_time_series, c='green', label='Actual time series')
    plt.plot(range(len(pre_history), len(source_time_series)), predicted_values, c='blue', label='Forecast')

    i = len(pre_history)
    for _ in range(0, validation_blocks):
        deviation = np.std(predicted_values)
        plt.plot([i, i], [min(actual_values) - deviation, max(actual_values) + deviation],
                 c='black', linewidth=1)
        i += forecast_length

    plt.legend(fontsize=15)
    start_view_point = len(source_time_series) - horizon - 50
    plt.xlim(start_view_point, len(source_time_series))
    plt.xlabel('Time index', fontsize=15)
    plt.show()
