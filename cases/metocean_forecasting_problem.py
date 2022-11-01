import logging
import os

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root


def prepare_input_data(train_file_path, test_file_path, history_size: int = 10000):
    """ Function for preparing InputData for train and test algorithm

    :param train_file_path: path to the csv file for training
    :param test_file_path: path to the csv file for validation
    :param history_size: length of time series

    :return dataset_to_train: InputData for train
    :return dataset_to_validate: InputData for validation
    """
    # Load train and test dataframes
    full_path_train = os.path.join(str(fedot_project_root()), train_file_path)
    full_path_test = os.path.join(str(fedot_project_root()), test_file_path)
    df_train = pd.read_csv(full_path_train)
    df_test = pd.read_csv(full_path_test)

    ws_history = np.ravel(np.array(df_train['wind_speed']))[:history_size]
    ssh_history = np.ravel(np.array(df_train['sea_height']))[:history_size]
    ssh_obs = np.ravel(np.array(df_test['sea_height']))[:history_size]

    return ssh_history, ws_history, ssh_obs


def run_metocean_forecasting_problem(train_file_path, test_file_path,
                                     forecast_length=1, visualization=False, timeout=5):
    # Prepare data for train and test
    ssh_history, ws_history, ssh_obs = prepare_input_data(train_file_path, test_file_path)

    historical_data = {
        'ws': ws_history,  # additional variable
        'ssh': ssh_history,  # target variable
    }

    fedot = Fedot(problem='ts_forecasting',
                  task_params=TsForecastingParams(forecast_length=forecast_length),
                  timeout=timeout, logging_level=logging.DEBUG)

    pipeline = fedot.fit(features=historical_data, target=ssh_history)
    fedot.forecast(historical_data)
    metric = fedot.get_metrics(target=ssh_obs)

    if visualization:
        pipeline.show()
        fedot.plot_prediction(target='ssh')

    return metric


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'

    run_metocean_forecasting_problem(file_path_train, file_path_test,
                                     forecast_length=12, timeout=10,
                                     visualization=True)
