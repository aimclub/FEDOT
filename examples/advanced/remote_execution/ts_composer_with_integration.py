import os
import random

import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.test_client import TestClient
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

random.seed(1)
np.random.seed(1)


# WARNING - THIS SCRIPT CAN BE EVALUATED ONLY WITH THE ACCESS TO DATAMALL SYSTEM

def clip_dataframe(df, forecast_horizon, history_size):
    # Take last n elements from dataframe to train
    dataframe_cutted = df.tail(history_size + forecast_horizon)

    return dataframe_cutted


def run_automl(data: MultiModalData, features_to_use,
               forecast_horizon: int = 10, history_size: int = 397,
               timeout: int = 1):
    """ Launch AutoML FEDOT algorithm for time series forecasting task """

    folder = os.path.join(fedot_project_root(), 'cases', 'data', 'metocean')

    connect_params = {}
    exec_params = {
        'container_input_path': os.path.join(fedot_project_root(), 'cases', 'data', 'metocean'),
        'container_output_path': os.path.join(fedot_project_root(), 'cases', 'data', 'metocean', 'remote'),
        'container_config_path': os.path.join(fedot_project_root(), 'cases', 'data', 'metocean', '.'),
        'container_image': "test",
        'timeout': 1
    }
    client = TestClient(connect_params, exec_params, output_path=os.path.join(folder, 'remote'))

    remote_task_params = RemoteTaskParams(
        mode='remote',
        dataset_name='metocean_data_train',
        max_parallel=20,
        var_names=features_to_use,
        target='sea_height'
    )

    # the following client and params can be used with DataMall system
    # connect_params = DEFAULT_CONNECT_PARAMS
    # exec_params = DEFAULT_EXEC_PARAMS
    # DataMallClient(connect_params, exec_params, output_path=os.path.join(folder, 'remote'))

    evaluator = RemoteEvaluator()
    evaluator.init(
        client=client,
        remote_task_params=remote_task_params
    )

    # Prepare parameters for algorithm launch
    composer_params = {'max_depth': 6,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 100,
                       'preset': 'fast_train',
                       'metric': 'rmse',
                       'cv_folds': None,
                       'validation_blocks': None}

    automl = Fedot(problem='ts_forecasting', timeout=timeout, **composer_params)

    obtained_pipeline = automl.fit(data)
    forecast = automl.forecast(data)
    return forecast, obtained_pipeline


features_to_use = ['wind_speed', 'sea_height']

data = MultiModalData.from_csv_time_series(
    file_path=f'{fedot_project_root()}/cases/data/metocean/metocean_data_train.csv',
    var_names=features_to_use,
    target_column='sea_height',
    idx_column='datetime')

forecast, obtained_pipeline = run_automl(data=data, features_to_use=['wind_speed', 'sea_height'],
                                         forecast_horizon=30,
                                         history_size=200,
                                         timeout=5)

obtained_pipeline.show()
