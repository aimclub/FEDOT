import json
import os
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
from sklearn.metrics import mean_squared_error

from fedot.core.composer.metrics import RMSE
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.atomized_model.atomized_decompose import AtomizedTimeSeriesDecompose
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.atomized_model.atomized_ts_sampler import AtomizedTimeSeriesDataSample
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.integration.utilities.test_pipeline_import_export import create_correct_path, create_func_delete_files


def get_data():
    time = np.linspace(0, 10, 50)
    time_series = np.sin(time)
    # time_series = np.polyval((10, 2, 2, 2), time)
    # time_series = np.diff(time_series)

    data = InputData(idx=np.arange(len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(max(20, int(len(time_series) // 10)))),
                     data_type=DataTypesEnum.ts)
    train, test = train_test_data_setup(data, validation_blocks=1)
    return train, test


def get_pipeline(include_atomized: bool, model: str = 'rfr'):
    node = PipelineNode('lagged')
    if include_atomized:
        pipeline = Pipeline(PipelineNode(model))
        node = PipelineNode(AtomizedTimeSeriesDataSample(pipeline), nodes_from=[node])
    else:
        node = PipelineNode(model, nodes_from=[node])
    return Pipeline(node)


def test_atomized_lagged_ts_sampler():
    train, test = get_data()

    atomized_pipeline = get_pipeline(True)
    atomized_pipeline.fit(train)
    atomized_predict = atomized_pipeline.predict(test)

    simple_pipeline = get_pipeline(True)
    simple_pipeline.fit(train)
    simple_predict = simple_pipeline.predict(test)

    assert np.allclose(simple_predict.predict, atomized_predict.predict)
