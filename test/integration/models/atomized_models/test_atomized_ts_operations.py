import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from typing import Type

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_differ import AtomizedTimeSeriesDiffer
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_sampler import AtomizedTimeSeriesSampler
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_scaler import AtomizedTimeSeriesScaler
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast


def get_data(fit_length: int, validation_blocks: int = 10, forecasting_length: int = 20):
    time = np.linspace(0, 10, fit_length)
    dt = time[1] - time[0]
    start = time[-1] + dt
    stop = start + validation_blocks * forecasting_length * dt
    time = np.concatenate([time, np.arange(start, stop, dt)])
    time_series = np.sin(time)

    data = InputData(idx=np.arange(len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecasting_length)),
                     data_type=DataTypesEnum.ts)
    train, test = train_test_data_setup(data, validation_blocks=validation_blocks)
    return train, test


def get_pipeline(atomized: Type[AtomizedModel] = None, model: str = 'rfr'):
    node = PipelineNode('lagged')
    if atomized is not None:
        pipeline = Pipeline(PipelineNode(model))
        node = PipelineNode(atomized(pipeline), nodes_from=[node])
    else:
        node = PipelineNode(model, nodes_from=[node])
    return Pipeline(node)


def predict(pipeline: Pipeline, train: InputData, test: InputData):
    pipeline.fit(train)
    return in_sample_ts_forecast(pipeline, test, len(test.target))


@pytest.mark.parametrize(('data_length', 'atomized_class'),
                         [(100, AtomizedTimeSeriesSampler),
                          (1000, AtomizedTimeSeriesScaler),
                          (1000, AtomizedTimeSeriesDiffer),
                          ])
def test_atomized_operations(data_length: int, atomized_class: Type[AtomizedModel]):
    train, test = get_data(data_length)
    atomized_predict = predict(get_pipeline(atomized_class), train, test)
    simple_predict = predict(get_pipeline(), train, test)
    # pd.DataFrame([test.target, simple_predict, atomized_predict], index=['target', 'simple', 'atomized']).T.plot()
    # print(mean_squared_error(test.target, simple_predict) - mean_squared_error(test.target, atomized_predict))
    assert mean_squared_error(test.target, simple_predict) >= mean_squared_error(test.target, atomized_predict)
