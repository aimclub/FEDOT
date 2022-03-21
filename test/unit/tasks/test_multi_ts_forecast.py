import os

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_data():
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=20))
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'test/data/short_multi_ts.csv')
    data = InputData.from_csv_multi_time_series(
        file_path=file_path,
        task=task)
    train_data, test_data = train_test_data_setup(data)
    return train_data, test_data


def get_simple_pipeline():
    node_lagged_1 = PrimaryNode("lagged")
    node_lagged_1.custom_params = {'window_size': 50}

    node_smoth = PrimaryNode("smoothing")
    node_lagged_2 = SecondaryNode("lagged", nodes_from=[node_smoth])
    node_lagged_2.custom_params = {'window_size': 30}

    node_ridge = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode("lasso", nodes_from=[node_lagged_2])

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge, node_lasso])
    pipeline = Pipeline(node_final)
    return pipeline


def test_multi_ts_forecasting():
    train_data, test_data = get_data()
    pipeline = get_simple_pipeline()
    pipeline.fit(train_data)
    prediction = np.ravel(pipeline.predict(test_data).predict)
    assert len(prediction) == len(test_data.target)