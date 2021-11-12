from typing import Union

import pandas as pd
import numpy as np

from cases.medical.wrappers import wrap_into_input, save_forecast, display_metrics
from fedot.api.main import Fedot
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from matplotlib import pyplot as plt


def launch_covid_forecasting(df, test_size, forecast_length, timeout):
    # Time series only for country
    first_row = np.array(df.iloc[0])
    ts = np.array(first_row[4:], dtype=int)

    test_len = round(len(ts) * test_size)
    train, test = ts[1: len(ts) - test_len], ts[len(ts) - test_len:]

    # Wrap train part into InputData
    train_ts = wrap_into_input(forecast_length, train)
    validation_ts = wrap_into_input(forecast_length, ts)

    task_parameters = TsForecastingParams(forecast_length=forecast_length)
    model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                  timeout=timeout, preset='ts_tun')
    pipeline = model.fit(train_ts)

    # Display some info about obtained pipeline
    pipeline.print_structure()
    pipeline.show()
    val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                        input_data=validation_ts,
                                        horizon=test_len)
    display_metrics(val_predict, test)
    # plot forecasting result
    plt.plot(validation_ts.idx, validation_ts.target, label='Actual time series')
    plt.plot(np.arange(len(train), len(train) + len(val_predict)), val_predict,
             label=f'Forecast for {forecast_length} elements ahead')
    plt.title(f'In-sample validation. Covid case')
    plt.xlabel('Time index')
    plt.legend()
    plt.show()

    save_forecast(forecast=val_predict, actual=test, path='covid_forecasts.csv')
    pipeline.save('')


def run_covid_experiment(timeout: Union[int, float] = 2, forecast_length: int = 30,
                         test_size: float = 0.3, only_important: bool = True):
    """ Run time series Fedot pipeline for covid data. For each time series perform experiments.

    :param timeout: number of minutes
    :param forecast_length: forecast horizon
    :param test_size: number of elements for validation (ratio)
    :param only_important: is there a need to prepare models only for time series from Russia and China
    """

    # Read time series data from file
    df = pd.read_csv('time_series_covid19_confirmed_global.csv')

    if only_important:
        # Prepare time series only
        countries_for_validation = ['Russia', 'China']
    else:
        countries_for_validation = df['Country/Region'].unique()

    for country in countries_for_validation:
        df_country = df[df['Country/Region'] == country]

        if len(df_country) == 1:
            print(f'Country{country}')
            # Time series only for country
            launch_covid_forecasting(df_country, test_size, forecast_length, timeout)
        else:
            # Time series for several cities in country
            for province in df_country['Province/State']:
                print(f'Province {province}, Country{country}')
                df_province = df[df['Province/State'] == province]
                launch_covid_forecasting(df_province, test_size, forecast_length, timeout)


if __name__ == '__main__':
    run_covid_experiment(timeout=2)
