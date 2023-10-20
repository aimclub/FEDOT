from copy import copy

import numpy as np
import pandas as pd
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_ts_pipeline(window_size):
    """ Function return pipeline with lagged transformation in it """
    node_lagged = PipelineNode('lagged')
    node_lagged.parameters = {'window_size': window_size}

    node_final = PipelineNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)
    return pipeline


def get_rasnac_pipeline():
    node_ransac = PipelineNode('ransac_lin_reg')
    node_ransac.parameters = {'residual_threshold': 0.0002}
    node_final = PipelineNode('linear', nodes_from=[node_ransac])
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
    file_path = fedot_project_root().joinpath('test/data/short_time_series.csv')
    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    ts_input = InputData(idx=np.arange(0, len(time_series)), features=time_series,
                         target=time_series, task=task, data_type=DataTypesEnum.ts)

    # Get pipeline with lagged transformation in it
    pipeline = get_ts_pipeline(window_size)

    # Fit it
    with pytest.raises(ValueError):
        pipeline.fit(ts_input)


def test_ransac_with_invalid_params_fit_correctly():
    """ Check that on a small dataset the RANSAC anomaly search algorithm can
    adjust the values of hyperparameters

    As stated in the sklearn documentation, min_samples is determined by default
    based on how many features are in the dataset
    Therefore, problems can arise when there are more attributes in a dataset
    than the number of objects
    """

    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

    data = InputData.from_csv(data_path)
    train, test = train_test_data_setup(data)
    train.task.task_type = TaskTypesEnum.regression
    ransac_pipeline = get_rasnac_pipeline()
    ransac_pipeline.fit(train)
    predicted = ransac_pipeline.predict(train)

    assert ransac_pipeline.is_fitted
    assert predicted is not None


def test_params_filter_with_non_default():
    """
    Check custom_params returns updated parameters only for changed keys.
    Default params for operation setting as string 'default_params'
    """
    input_data = InputData(idx=np.arange(0, 3),
                           features=np.array([[1, 0, 2],
                                              [2, 0, 3],
                                              [3, 1, 4]]),
                           target=np.array([[1], [0], [1]]),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    # Params are default for now - 'default_params'
    node_knn = PipelineNode('knn')
    default_params = copy(node_knn.parameters)

    node_knn.fit(input_data)
    updated_params = node_knn.parameters

    assert default_params == {}
    assert 'n_neighbors' in list(updated_params.keys())
    assert len(list(updated_params.keys())) == 1
