import numpy as np
import pandas as pd
from pylab import rcParams

# Additional custom functions
from cases.industrial.processing import advanced_validation, automl_fit_forecast, plot_diesel_and_wind, \
    prepare_unimodal_for_validation

rcParams['figure.figsize'] = 15, 7

if __name__ == '__main__':
    # Below is an example of univariate time series forecasting.
    # An example of how forecasts can be made is presented and an advanced
    # validation is given on a several validation blocks which length is equal
    # to the length of the forecast horizon * number of validation blocks.

    # Define forecast horizon and read dataframe
    forecast_length = 20
    validation_blocks = 2
    df = pd.read_csv('pw_dataset.csv', parse_dates=['datetime'])

    # Make visualisation
    plot_diesel_and_wind(df)

    # Wrap time series data into InputData class
    ts = np.array(df['diesel_fuel_kWh'])
    # Get data for time series implementation
    train_input, validation_input = prepare_unimodal_for_validation(ts, forecast_length,
                                                                    validation_blocks)

    # Prepare parameters for algorithm launch
    # timeout 2 - means that AutoML algorithm will work for 2 minutes
    composer_params = {'max_depth': 4,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 100,
                       'timeout': 2,
                       'preset': 'best_quality',
                       'metric': 'rmse',
                       'cv_folds': 2,
                       'validation_blocks': 2}
    forecast, obtained_pipeline = automl_fit_forecast(train_input, validation_input, composer_params,
                                                      vis=True, in_sample_forecasting=True,
                                                      horizon=forecast_length * validation_blocks)

    # Perform in-sample validation and display metrics
    advanced_validation(forecast, forecast_length, validation_blocks, ts)
