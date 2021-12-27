import numpy as np
import pandas as pd
from pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Additional custom functions
from cases.industrial.processing import multi_automl_fit_forecast, plot_results
from fedot.core.data.multi_modal import prepare_multimodal_data

rcParams['figure.figsize'] = 15, 7

if __name__ == '__main__':
    # Below is an example of multivariate time series forecasting.
    # An example of how forecasts can be made is presented and a simple
    # validation is given on a single block which length is equal to the
    # length of the forecast horizon.

    # Define forecast horizon and read dataframe
    forecast_length = 20
    df = pd.read_csv('pw_dataset.csv', parse_dates=['datetime'])

    # Wrap time series data into InputData class
    features_to_use = ['wind_power_kWh', 'diesel_time_h', 'wind_time_h',
                       'velocity_max_msec', 'velocity_mean_msec', 'tmp_grad',
                       'diesel_fuel_kWh']
    ts = np.array(df['diesel_fuel_kWh'])
    mm_train, mm_test, = prepare_multimodal_data(dataframe=df,
                                                 features=features_to_use,
                                                 forecast_length=forecast_length)

    # Prepare parameters for algorithm launch
    # timeout 5 - means that AutoML algorithm will work for 5 minutes
    composer_params = {'max_depth': 6,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 20,
                       'timeout': 0.5,
                       'preset': 'best_quality',
                       'metric': 'rmse',
                       'cv_folds': None,
                       'validation_blocks': None}
    forecast, obtained_pipeline = multi_automl_fit_forecast(mm_train, mm_test,
                                                            composer_params,
                                                            ts, forecast_length,
                                                            vis=True)

    mse_metric = mean_squared_error(ts[-forecast_length:], forecast, squared=False)
    mae_metric = mean_absolute_error(ts[-forecast_length:], forecast)
    print(f'MAE - {mae_metric:.2f}')
    print(f'RMSE - {mse_metric:.2f}')

    # Save obtained pipeline
    obtained_pipeline.save('best')

    # Visualise predictions
    plot_results(actual_time_series=ts,
                 predicted_values=forecast,
                 len_train_data=len(ts) - forecast_length)
