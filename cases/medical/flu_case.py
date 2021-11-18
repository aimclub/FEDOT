from typing import Union

import pandas as pd
import numpy as np
from datetime import datetime

from cases.medical.wrappers import wrap_into_input, save_forecast, display_metrics
from fedot.api.main import Fedot
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams
from matplotlib import pyplot as plt


def run_flu_experiment(timeout: Union[int, float] = 2, forecast_length: int = 30, test_size: int = 245):
    """ Launch time series model identification """

    # Read time series data from file
    df = pd.read_csv('inc_spb_daily_allyears_converted.csv', parse_dates=['datetime'])
    ts = np.array(df['Всего'])
    train, test = ts[1: len(ts) - test_size], ts[len(ts) - test_size:]
    dates = df['datetime'].iloc[-test_size:]

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
                                        horizon=test_size)
    display_metrics(val_predict, test)

    # plot forecasting result
    plt.plot(df['datetime'], df['Всего'], label='Actual time series')
    plt.plot(dates, val_predict,
             label=f'Forecast for {forecast_length} days ahead')
    plt.title(f'In-sample validation. Flu case')
    plt.xlabel('Datetime')
    plt.legend()
    plt.show()

    save_forecast(dates=dates, forecast=val_predict, actual=test, path='flu_forecasts.csv')
    pipeline.save('')


if __name__ == '__main__':
    run_flu_experiment(timeout=10)
