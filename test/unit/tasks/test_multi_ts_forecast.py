import os
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_multi_ts_data(forecast_length: int = 5, validation_blocks: Optional[int] = None):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'test/data/synthetic_multi_ts.csv')
    data = InputData.from_csv_multi_time_series(
        file_path=file_path,
        task=task)
    train_data, test_data = train_test_data_setup(data, validation_blocks=validation_blocks)
    return train_data, test_data


def get_simple_pipeline():
    node_lagged_1 = PipelineNode("lagged")
    node_lagged_1.parameters = {'window_size': 20}

    node_smooth = PipelineNode("smoothing")
    node_lagged_2 = PipelineNode("lagged", nodes_from=[node_smooth])
    node_lagged_2.parameters = {'window_size': 10}

    node_ridge = PipelineNode("ridge", nodes_from=[node_lagged_1])
    node_lasso = PipelineNode("lasso", nodes_from=[node_lagged_2])

    node_final = PipelineNode("ridge", nodes_from=[node_ridge, node_lasso])
    pipeline = Pipeline(node_final)
    return pipeline


def get_linear_pipeline():
    node_smoothed = PipelineNode("smoothing")
    node_lagged = PipelineNode("lagged", nodes_from=[node_smoothed])
    node_lagged.parameters = {'window_size': 3}
    node_linear = PipelineNode("linear", nodes_from=[node_lagged])
    node_lagged2 = PipelineNode("lagged")
    node_lagged2.parameters = {'window_size': 5}
    node_linear2 = PipelineNode("linear", nodes_from=[node_lagged2])
    node_linear3 = PipelineNode("linear", nodes_from=[node_linear, node_linear2])
    pipeline = Pipeline(node_linear3)
    return pipeline


def test_multi_ts_predict():
    train_data, test_data = get_multi_ts_data()
    pipeline = get_linear_pipeline()
    pipeline.fit(train_data)
    prediction = np.ravel(pipeline.predict(test_data).predict)
    assert np.allclose(np.ravel(prediction), np.ravel(test_data.target))
