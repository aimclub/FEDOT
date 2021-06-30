import os

import numpy as np
import pandas as pd

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_ts_pipeline(window_size):
    """ Function return pipeline with lagged transformation in it """
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}

    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)
    return pipeline


def test_lagged_with_invalid_params_fit_correctly():
    """ The function define a pipeline with incorrect parameters in the lagged
    transformation. During the training of the pipeline, the parameter 'window_size'
    is corrected
    """
    window_size = 600
    len_forecast = 50

    # The length of the time series is 500 elements
    project_root_path = str(fedot_project_root())
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

    # Get pipeline with lagged transformation in it
    pipeline = get_ts_pipeline(window_size)

    # Fit it
    pipeline.fit(train_input)

    is_pipeline_was_fitted = True
    assert is_pipeline_was_fitted
