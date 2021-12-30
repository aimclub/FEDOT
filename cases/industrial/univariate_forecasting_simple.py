import numpy as np
import pandas as pd
from pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Additional custom functions
from cases.industrial.processing import automl_fit_forecast, plot_diesel_and_wind, plot_results, prepare_unimodal_data
# FEDOT imports
from fedot.core.data.data_split import train_test_data_setup

rcParams['figure.figsize'] = 15, 7

if __name__ == '__main__':
    # Below is an example of univariate time series forecasting.
    # An example of how forecasts can be made is presented and a simple
    # validation is given on a single block which length is equal to the
    # length of the forecast horizon.

    # Define forecast horizon and read dataframe
    forecast_length = 30
    df = pd.read_csv('pw_dataset.csv', parse_dates=['datetime'])

    # Make visualisation
    plot_diesel_and_wind(df)

    # Wrap time series data into InputData class
    ts = np.array(df['diesel_fuel_kWh'])
    input_ts = prepare_unimodal_data(ts, forecast_length)

    # Split data into train and test
    train_input, predict_input = train_test_data_setup(input_ts)

    # Prepare parameters for algorithm launch
    # timeout 2 - means that AutoML algorithm will work for 2 minutes
    composer_params = {'max_depth': 3,
                       'max_arity': 4,
                       'pop_size': 20,
                       'num_of_generations': 20,
                       'timeout': 2,
                       'with_tuning': True,
                       'preset': 'best_quality',
                       'genetic_scheme': None,
                       'history_folder': None}
    forecast, obtained_pipeline = automl_fit_forecast(train_input, predict_input, composer_params,
                                                      vis=True, in_sample_forecasting=False)

    mse_metric = mean_squared_error(predict_input.target, forecast, squared=False)
    mae_metric = mean_absolute_error(predict_input.target, forecast)
    print(f'MAE - {mae_metric:.2f}')
    print(f'RMSE - {mse_metric:.2f}')

    # Visualise predictions
    plot_results(actual_time_series=ts,
                 predicted_values=forecast,
                 len_train_data=len(ts) - forecast_length)
