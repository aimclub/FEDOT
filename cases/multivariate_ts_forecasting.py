import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams


def train_test_split(dataframe: pd.DataFrame, forecast_horizon: int):
    train_df = []
    test_df = []
    for ts in list(dataframe['label'].unique()):
        ts_dataframe = dataframe[dataframe['label'] == ts]
        ts_dataframe = ts_dataframe.sort_values(by='datetime')

        train_len = len(ts_dataframe) - forecast_horizon
        train_df.append(ts_dataframe.head(train_len))
        test_df.append(ts_dataframe.tail(forecast_horizon))

    train_df = pd.concat(train_df)
    test_df = pd.concat(test_df)
    return train_df, test_df


def plot_results(full_df: pd.DataFrame, target_column: int, forecast: np.array,
                 forecast_horizon: int):
    """ Display forecasted vs actual values """
    target_df = full_df[full_df['label'] == target_column]
    datetime_xs = target_df['datetime']
    plt.plot(datetime_xs, target_df['value'], label='Actual values')
    plt.plot(datetime_xs.tail(forecast_horizon), forecast[:-forecast_horizon], label='Forecast')
    plt.xlabel('Datetime', fontsize=13)
    plt.ylabel('Sea surface height, m', fontsize=13)
    plt.legend()
    plt.grid()
    plt.show()


def launch_fedot_forecasting(target_column: int = 1, forecast_horizon: int = 50,
                             number_of_series_to_use: int = 25):
    """ Example how to launch FEDOT AutmoML for multivariate forecasting """
    path_to_file = os.path.join(os.path.curdir, 'data', 'multivariate_ssh.csv')
    df = pd.read_csv(path_to_file, parse_dates=['datetime'])
    train_df, test_df = train_test_split(df, forecast_horizon)

    # Prepare train data in a form of dictionary {'time series label': numpy array}
    train_data = {}
    feature_series = list(train_df['label'].unique())
    for feature_ts in feature_series[:number_of_series_to_use]:
        current_ts = train_df[train_df['label'] == feature_ts]
        train_data.update({str(feature_ts): np.array(current_ts['value'])})

    # Configure AutoML
    task_parameters = TsForecastingParams(forecast_length=forecast_horizon)
    model = Fedot(problem='ts_forecasting', task_params=task_parameters, timeout=3,
                  cv_folds=2, validation_blocks=2, with_tuning=False)
    target_series = train_df[train_df['label'] == target_column]
    obtained_pipeline = model.fit(features=train_data,
                                  target=np.array(target_series['value']))

    obtained_pipeline.show()

    # Use historical value to make forecast
    forecast = model.predict(train_data)

    plot_results(df, target_column, forecast, forecast_horizon)


if __name__ == '__main__':
    launch_fedot_forecasting(number_of_series_to_use=30)
