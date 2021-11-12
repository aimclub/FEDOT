import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum


def wrap_into_input(forecast_length, time_series):
    """ Convert data for FEDOT framework """
    time_series = np.array(time_series)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    input_data = InputData(idx=np.arange(0, len(time_series)),
                           features=time_series, target=time_series,
                           task=task, data_type=DataTypesEnum.ts)

    return input_data


def save_forecast(forecast: np.array, actual: np.array, path: str):
    """ Save forecast in csv file """
    df = pd.DataFrame({'actual': actual, 'forecast': forecast})
    df.to_csv(path, index=False)


def display_metrics(forecast: np.array, actual: np.array):
    """ Calculate several metrics for validation part of time series """

    mae_value = mean_absolute_error(actual, forecast)
    rmse_value = mean_squared_error(actual, forecast, squared=False)
    smape_value = smape(actual, forecast)

    print(f'MAE metric: {mae_value:.2f}')
    print(f'RMSE metric: {rmse_value:.2f}')
    print(f'SMAPE metric: {smape_value:.2f}')


def smape(y_true, y_pred):
    return np.mean(100 *(2*np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))))