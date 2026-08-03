from types import SimpleNamespace

import numpy as np

from fedot.core.data.input_data.data import OutputData
from fedot.core.data.multimodal.supplementary_data import SupplementaryData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def _make_ts_output():
    return OutputData(
        idx=np.arange(2),
        predict=np.array([[1.0], [2.0]]),
        target=None,
        task=Task(TaskTypesEnum.ts_forecasting,
                  TsForecastingParams(forecast_length=2)),
        data_type=DataTypesEnum.ts,
    )


def test_pipeline_postprocess_uses_typed_postprocess_plan():
    pipeline = Pipeline()

    result = pipeline._postprocess(_make_ts_output(), output_mode='labels')

    assert result.predict.tolist() == [1.0, 2.0]


def test_pipeline_fit_assigns_data_and_delegates_to_internal_fit():
    pipeline = Pipeline()
    assigned = []
    pipeline._assign_data_to_nodes = lambda data: assigned.append(data) or data
    pipeline._fit = lambda tensor_data=None, **kwargs: ('fitted', tensor_data)
    tensor_data = SimpleNamespace(name='tensor-data')

    result = pipeline.fit(tensor_data)

    assert result == ('fitted', tensor_data)
    assert len(assigned) == 1
    assert assigned[0] is not tensor_data


def test_pipeline_predict_assigns_data_and_delegates_to_root_node():
    pipeline = Pipeline()
    assigned = []
    captured = {}
    pipeline._assign_data_to_nodes = lambda data: assigned.append(data) or data
    pipeline._postprocess = lambda result, output_mode: (
        'postprocessed', result, output_mode)

    original_is_fitted = Pipeline.is_fitted
    original_root_node = Pipeline.root_node
    Pipeline.is_fitted = property(lambda self: True)
    Pipeline.root_node = property(lambda self: SimpleNamespace(
        predict=lambda tensor_data, output_mode, predictions_cache, fold_id: (
            captured.update({
                'tensor_data': tensor_data,
                'output_mode': output_mode,
            }),
            _make_ts_output(),
        )[1]
    ))

    try:
        tensor_data = SimpleNamespace(name='tensor-data')
        result = pipeline.predict(tensor_data, output_mode='labels')
    finally:
        Pipeline.is_fitted = original_is_fitted
        Pipeline.root_node = original_root_node

    assert len(assigned) == 1
    assert assigned[0] is not tensor_data
    assert captured['tensor_data'] is assigned[0]
    assert captured['output_mode'] == 'labels'
    assert result[0] == 'postprocessed'
    assert result[2] == 'labels'
