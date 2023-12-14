import json
import os
from functools import reduce

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from fedot.core.composer.metrics import RMSE
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.atomized_model.atomized_decompose import AtomizedTimeSeriesDecompose
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.integration.utilities.test_pipeline_import_export import create_correct_path, create_func_delete_files


def get_data():
    time = np.linspace(0, 1.5, 700)
    time_series = np.polyval((1, 1, 1, 1), time)

    data = InputData(idx=np.arange(len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(int(len(time_series) // 10))),
                     data_type=DataTypesEnum.ts)
    train, test = train_test_data_setup(data, validation_blocks=2)
    return train, test


def get_pipeline(lagged=True):
    node = PipelineNode('lagged')
    node = PipelineNode('ridge', nodes_from=[node])
    node = PipelineNode(AtomizedTimeSeriesDecompose(), nodes_from=[node])
    return Pipeline(node)


def test_atomized_lagged_time_series_decompose():
    train, test = get_data()
    pipeline = get_pipeline()
    pipeline.fit(train)
    predict = pipeline.predict(test)
    print(1)
