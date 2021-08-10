import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# FEDOT imports
from fedot.core.data.data_split import train_test_data_setup

# Additional custom functions
from cases.demo.processing import prepare_unimodal_data, plot_diesel_and_wind, \
    plot_results, automl_through_api

from pylab import rcParams
rcParams['figure.figsize'] = 15, 7


if __name__ == '__main__':
    forecast_length = 30
    df = pd.read_csv('pw_dataset.csv', parse_dates=['datetime'])

    # Make visualisation
    plot_diesel_and_wind(df)

    # Wrap time series data into InputData class
    ts = np.array(df['diesel_fuel_kWh'])
    input_ts = prepare_unimodal_data(ts, forecast_length)

    # Split data into train and test
    train_input, predict_input = train_test_data_setup(input_ts)

    # Make predictions
    forecast = automl_through_api(train_input, predict_input, timeout=10)

    mse_metric = mean_squared_error(predict_input.target, forecast, squared=False)
    mae_metric = mean_absolute_error(predict_input.target, forecast)
    print(f'MAE - {mae_metric:.2f}')
    print(f'RMSE - {mse_metric:.2f}')

    # Visualise predictions
    plot_results(actual_time_series=ts,
                 predicted_values=forecast,
                 len_train_data=len(ts) - forecast_length)
