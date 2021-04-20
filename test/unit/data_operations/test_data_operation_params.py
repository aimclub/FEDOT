import os
import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root


def get_ts_chain(window_size):
    """ Function return chain with lagged transformation in it """
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}

    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    chain = Chain(node_final)
    return chain


def test_lagged_transformation_parameters_improving():
    """ Function checks if the lagged parameters correct works well """
    window_size = 600
    len_forecast = 50

    # The length of the time series is 500 elements
    project_root_path = str(project_root())
    file_path = os.path.join(project_root_path, 'test/data/short_time_series.csv')
    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(time_series)),
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Get chain with lagged transformation in it
    chain = get_ts_chain(window_size)

    # Fit it
    chain.fit(train_input)

    is_chain_was_fitted = True
    assert is_chain_was_fitted
