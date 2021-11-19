import os
import time
from typing import Union

import pandas as pd
import numpy as np

from cases.medical.covid_analysis import TEST_RATIO
from cases.medical.wrappers import wrap_into_input, save_forecast, display_metrics
from fedot.api.main import Fedot
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams
from matplotlib import pyplot as plt


def launch_covid_forecasting(df, test_size, forecast_length, timeout, folder_to_save, vis):
    """ Perform time series forecasting based on Covid data """
    create_folder(folder_to_save)
    # Time series only for country
    first_row = np.array(df.iloc[0])
    dates_df = pd.DataFrame({'datetime': np.array(df.columns[4:], dtype=str)})
    dates_df['datetime'] = pd.to_datetime(dates_df['datetime'], format="%m/%d/%y")

    # Use different time series train part ratios
    for train_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        train_size_path = os.path.join(folder_to_save, str(train_size).replace('.', '_'))
        create_folder(train_size_path)

        ts = np.array(first_row[4:], dtype=int)

        test_len = round(len(ts) * test_size)
        if train_size + test_size > 1:
            raise ValueError('Train + test part should be equal to 1.0 or be smaller!')
        full_train_len = round(len(ts) * (train_size + test_size))

        # Take only cutted part of time series
        ts = ts[-full_train_len:]

        train, test = ts[1: len(ts) - test_len], ts[len(ts) - test_len:]
        # Dates for test set
        dates = dates_df['datetime'].iloc[-test_len:]

        for launch_id in range(0, 10):
            # Wrap train part into InputData
            train_ts = wrap_into_input(forecast_length, train)
            validation_ts = wrap_into_input(forecast_length, ts)

            task_parameters = TsForecastingParams(forecast_length=forecast_length)
            model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                          timeout=timeout, preset='ts_tun')
            pipeline = model.fit(train_ts)

            # Display some info about obtained pipeline
            pipeline.print_structure()
            file_name = ''.join(('pipeline_', str(launch_id), '.png'))
            path = os.path.join(train_size_path, file_name)
            pipeline.show(path)

            val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                                input_data=validation_ts,
                                                horizon=test_len)

            if vis:
                # Plot forecasting result
                plt.plot(dates_df['datetime'], validation_ts.target, label='Actual time series')
                plt.plot(dates, val_predict,
                         label=f'Forecast for {forecast_length} days ahead')
                plt.title(f'In-sample validation. Covid case')
                plt.xlabel('Datetime')
                plt.legend()
                plt.show()

            csv_name = ''.join(('covid_forecasts_', str(launch_id), '.csv'))
            csv_path = os.path.join(train_size_path, csv_name)
            save_forecast(dates=dates, forecast=val_predict, actual=test, path=csv_path)

            pipeline.save(os.path.join(train_size_path, ''))

            folders = os.listdir(train_size_path)
            for folder in folders:
                if folder.endswith(',PM'):
                    # Folder need to be renamed
                    old_name = os.path.join(train_size_path, folder)
                    new_name = os.path.join(train_size_path, ''.join(('pipeline_', str(launch_id))))
                    os.rename(old_name, new_name)


def run_covid_experiment(timeout: Union[int, float] = 2, forecast_length: int = 30,
                         test_size: float = TEST_RATIO, only_important: bool = True,
                         path_to_save: str = 'covid', vis: bool = False):
    """ Run time series Fedot pipeline for covid data. For each time series perform experiments.

    :param timeout: number of minutes
    :param forecast_length: forecast horizon
    :param test_size: number of elements for validation (ratio)
    :param only_important: is there a need to prepare models only for time series from Russia and China
    :param path_to_save: save results in the desired folder
    :param vis: is there is a need to visualise results
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
        folder_to_save = os.path.join(path_to_save, str(country))

        if len(df_country) == 1:
            print(f'Country {country}')
            # Time series only for country
            launch_covid_forecasting(df_country, test_size, forecast_length, timeout, folder_to_save, vis)
        else:
            # Time series for several cities in country
            for province in df_country['Province/State']:
                print(f'Province {province}, Country {country}')

                # Union of country name and province name
                country_province_name = ''.join((str(country), '_', str(country)))
                folder_to_save = os.path.join(path_to_save, country_province_name)

                df_province = df[df['Province/State'] == province]
                launch_covid_forecasting(df_province, test_size, forecast_length, timeout, folder_to_save, vis)


def create_folder(path):
    # Create new folder if it's not exists
    if os.path.isdir(path) is False:
        os.makedirs(path)


if __name__ == '__main__':
    run_covid_experiment(timeout=3, path_to_save='./covid')
