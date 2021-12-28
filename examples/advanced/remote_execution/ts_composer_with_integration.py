import os
import random

import numpy as np
import pandas as pd

from cases.industrial.processing import multi_automl_fit_forecast, plot_diesel_and_wind, plot_results
from fedot.core.data.multi_modal import prepare_multimodal_data
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.datamall_client import DEFAULT_CONNECT_PARAMS, DEFAULT_EXEC_PARAMS, \
    DataMallClient
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

random.seed(1)
np.random.seed(1)


# WARNING - THIS SCRIPT CAN BE EVALUATED ONLY WITH THE ACCESS TO DATAMALL SYSTEM

def clip_dataframe(df, forecast_horizon, history_size):
    # Take last n elements from dataframe to train
    dataframe_cutted = df.tail(history_size + forecast_horizon)

    return dataframe_cutted


def run_automl(df: pd.DataFrame, features_to_use: list, target_series: str,
               forecast_horizon: int = 10, history_size: int = 397,
               timeout: int = 1):
    """ Launch AutoML FEDOT algorithm for time series forecasting task """

    folder = os.path.join(fedot_project_root(), 'cases', 'industrial')

    connect_params = DEFAULT_CONNECT_PARAMS
    exec_params = DEFAULT_EXEC_PARAMS

    client = DataMallClient(connect_params, exec_params, output_path=os.path.join(folder, 'remote'))

    remote_task_params = RemoteTaskParams(
        mode='remote',
        dataset_name='pw_dataset',
        max_parallel=20,
        var_names=features_to_use
    )

    evaluator = RemoteEvaluator()
    evaluator.init(
        client=client,
        remote_task_params=remote_task_params
    )

    dataframe_cutted = clip_dataframe(df, forecast_horizon, history_size)

    ts = np.array(dataframe_cutted[target_series])
    mm_train, mm_test, = prepare_multimodal_data(dataframe=dataframe_cutted,
                                                 features=features_to_use,
                                                 forecast_length=forecast_horizon)
    # Prepare parameters for algorithm launch
    # timeout 2 - means that AutoML algorithm will work for 2 minutes
    composer_params = {'max_depth': 6,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 100,
                       'timeout': timeout,
                       'preset': 'fast_train',
                       'metric': 'rmse',
                       'cv_folds': None,
                       'validation_blocks': None}
    forecast, obtained_pipeline = multi_automl_fit_forecast(mm_train, mm_test,
                                                            composer_params,
                                                            ts, forecast_horizon,
                                                            vis=False)

    return forecast, obtained_pipeline


def plot_automl_forecast(df, forecast_horizon, history_size, forecast):
    dataframe_cutted = clip_dataframe(df, forecast_horizon, history_size)
    ts = np.array(dataframe_cutted[target_series])

    # Visualise predictions
    plot_results(actual_time_series=ts,
                 predicted_values=forecast,
                 len_train_data=len(ts) - forecast_horizon)


df = pd.read_csv('../../../cases/industrial/pw_dataset.csv', parse_dates=['datetime'])

# Make visualisation
plot_diesel_and_wind(df)

features_to_use = ['wind_power_kWh', 'diesel_time_h', 'wind_time_h',
                   'velocity_max_msec', 'velocity_mean_msec', 'tmp_grad',
                   'diesel_fuel_kWh']
target_series = 'diesel_fuel_kWh'

forecast, obtained_pipeline = run_automl(df=df, features_to_use=features_to_use,
                                         target_series=target_series,
                                         forecast_horizon=30,
                                         history_size=200,
                                         timeout=1)

obtained_pipeline.show()

plot_automl_forecast(df, forecast_horizon=30, history_size=200, forecast=forecast)

forecast_horizon = 60
history_size = 300
timeout = 2
forecast, obtained_pipeline = run_automl(df=df, features_to_use=features_to_use,
                                         target_series=target_series,
                                         forecast_horizon=forecast_horizon,
                                         history_size=history_size,
                                         timeout=timeout)

obtained_pipeline.show()

plot_automl_forecast(df, forecast_horizon, history_size, forecast=forecast)
