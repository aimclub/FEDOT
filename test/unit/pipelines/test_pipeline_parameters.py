import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository import tasks
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from test.unit.tasks.test_forecasting import get_simple_ts_pipeline


def small_ts_dataset():
    forecast_len = 2
    original_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    task_parameters = tasks.TsForecastingParams(forecast_length=forecast_len)
    task = tasks.Task(tasks.TaskTypesEnum.ts_forecasting, task_parameters)

    input_ts = InputData(idx=np.arange(0, len(original_array)),
                         features=original_array,
                         target=original_array,
                         task=task, data_type=DataTypesEnum.ts)

    return input_ts


def test_parameters_changed_correct():
    """ Check the parameters change correctly and are they updated correctly """
    input_ts = small_ts_dataset()

    # Get simple pipeline for time series forecasting
    ts_pipeline = get_simple_ts_pipeline()

    # Fit pipeline with inconceivably incorrect parameters (window_size will be corrected to 2)
    ts_pipeline.fit(input_ts)

    # Correct window_size parameter to new value
    ts_pipeline.nodes[1].parameters = {'window_size': 3}

    content_params = ts_pipeline.nodes[1].content['params']
    custom_params = ts_pipeline.nodes[1].parameters
    descriptive_id = ts_pipeline.nodes[1].descriptive_id
    assert content_params == custom_params
    assert '3' in descriptive_id
