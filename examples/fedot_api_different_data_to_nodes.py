import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from examples.ts_forecasting_with_exogenous import prepare_input_data


def run_different_data_sources_api(path_to_file, len_forecast):
    """ Example of using FEDOT api for time series forecasting with different
    data sources connected to primary nodes

    :param path_to_file: path to the dataframe with time series
    :param len_forecast: length of the forecast to made
    """
    df = pd.read_csv(path_to_file)
    time_series = np.array(df['Level'])
    exog_variable = np.array(df['Neighboring level'])

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Exog feature
    train_data_exog = exog_variable[:-len_forecast]
    test_data_exog = exog_variable[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    # Exogenous time series
    train_input_exog, predict_input_exog, _ = prepare_input_data(len_forecast=len_forecast,
                                                                 train_data_features=train_data_exog,
                                                                 train_data_target=train_data,
                                                                 test_data_features=test_data_exog)

    # Define chain - we can add names to nodes as like this for example
    node_lagged = PrimaryNode('lagged', node_name='lagged')
    node_exog = PrimaryNode('exog', node_name='exogenous')
    node_final = SecondaryNode('ridge', node_name='final',
                               nodes_from=[node_lagged, node_exog])
    predefined_model = Chain(node_final)

    # Trace data to different nodes

    model = Fedot(problem='ts_forecasting', task_params=task.task_params)
    # TODO - remove it may be


if __name__ == '__main__':
    path = '../notebooks/jupyter_media/time_series_forecasting/sea_level.csv'
    run_different_data_sources_api(path_to_file=path,
                                   len_forecast=250)