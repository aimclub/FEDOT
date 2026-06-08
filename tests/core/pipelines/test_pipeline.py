from types import SimpleNamespace

import numpy as np

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.multimodal.supplementary_data import SupplementaryData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class _StubPreprocessor:
    def __init__(self):
        self.calls = []

    def restore_index(self, copied_input_data, result):
        self.calls.append('restore_index')
        return result

    def apply_inverse_target_encoding(self, prediction):
        self.calls.append('inverse_target_encoding')
        return prediction + 1


def _make_ts_output():
    return OutputData(
        idx=np.arange(2),
        predict=np.array([[1.0], [2.0]]),
        target=None,
        task=Task(TaskTypesEnum.ts_forecasting,
                  TsForecastingParams(forecast_length=2)),
        data_type=DataTypesEnum.ts,
    )


def _make_classification_input(is_auto_preprocessed: bool):
    supplementary_data = SupplementaryData(
        is_auto_preprocessed=is_auto_preprocessed)
    return InputData(
        idx=np.arange(2),
        features=np.array([[1.0], [2.0]]),
        target=np.array([[0.0], [1.0]]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        supplementary_data=supplementary_data,
    )


def test_pipeline_postprocess_uses_typed_postprocess_plan():
    pipeline = Pipeline(use_input_preprocessing=False)
    pipeline.preprocessor = _StubPreprocessor()

    result = pipeline._postprocess(
        None, _make_ts_output(), output_mode='labels')

    assert pipeline.preprocessor.calls == [
        'restore_index', 'inverse_target_encoding']
    assert result.predict.tolist() == [2.0, 3.0]


def test_pipeline_fit_skips_preprocessing_when_input_is_marked_auto_preprocessed():
    pipeline = Pipeline(use_input_preprocessing=False)
    pipeline._preprocess = lambda *args, **kwargs: (_ for _ in ()
                                                    ).throw(AssertionError('_preprocess should not be called'))
    pipeline._assign_data_to_nodes = lambda data: data
    pipeline._fit = lambda input_data=None, predictions_cache=None, fold_id=None: 'ok'
    input_data = _make_classification_input(is_auto_preprocessed=True)

    result = pipeline.fit(input_data)

    assert result == 'ok'


def test_pipeline_fit_tensordata_uses_tensor_runtime_preprocess_path():
    pipeline = Pipeline(use_input_preprocessing=False)
    pipeline.preprocessor = _StubPreprocessor()
    calls = []

    def fake_prepare_tensordata(tensor_data, *, is_fit_stage, is_optional, pipeline=None):
        calls.append((is_fit_stage, is_optional, pipeline))
        return f'tensor-{len(calls)}'

    pipeline.preprocessor.prepare_tensordata = fake_prepare_tensordata
    pipeline.preprocessor.convert_indexes_for_fit = lambda pipeline, data: (
        'fit-indexed', data)
    pipeline.preprocessor.reduce_memory_size = lambda data: ('reduced', data)
    pipeline._assign_data_to_nodes = lambda data: data
    pipeline._fit = lambda input_data=None, predictions_cache=None, fold_id=None: input_data

    original_bridge = getattr(
        __import__(
            'fedot.core.pipelines.pipeline',
            fromlist=['tensordata_to_input_data']),
        'tensordata_to_input_data')
    import fedot.core.pipelines.pipeline as pipeline_module
    pipeline_module.tensordata_to_input_data = lambda tensor_data: (
        'input', tensor_data)

    try:
        result = pipeline.fit_tensordata(
            SimpleNamespace(features=np.array([[1.0]])))
    finally:
        pipeline_module.tensordata_to_input_data = original_bridge

    assert calls == [(True, False, None), (True, True, pipeline)]
    assert result == ('reduced', ('fit-indexed', ('input', 'tensor-2')))


def test_pipeline_predict_tensordata_uses_tensor_runtime_preprocess_path():
    pipeline = Pipeline(use_input_preprocessing=False)
    pipeline.preprocessor = _StubPreprocessor()
    calls = []

    def fake_prepare_tensordata(tensor_data, *, is_fit_stage, is_optional, pipeline=None):
        calls.append((is_fit_stage, is_optional, pipeline))
        return f'tensor-{len(calls)}'

    pipeline.preprocessor.prepare_tensordata = fake_prepare_tensordata
    pipeline.preprocessor.convert_indexes_for_predict = lambda pipeline, data: (
        'predict-indexed', data)
    pipeline.preprocessor.update_indices_for_time_series = lambda data: (
        'ts-updated', data)
    pipeline.preprocessor.reduce_memory_size = lambda data: ('reduced', data)
    pipeline._assign_data_to_nodes = lambda data: data
    pipeline._postprocess = lambda copied_input_data, result, output_mode: (
        'postprocessed', copied_input_data, result, output_mode)

    original_bridge = getattr(
        __import__(
            'fedot.core.pipelines.pipeline',
            fromlist=['tensordata_to_input_data']),
        'tensordata_to_input_data')
    import fedot.core.pipelines.pipeline as pipeline_module
    pipeline_module.tensordata_to_input_data = lambda tensor_data: (
        'input', tensor_data)

    original_is_fitted = Pipeline.is_fitted
    original_root_node = Pipeline.root_node
    Pipeline.is_fitted = property(lambda self: True)
    Pipeline.root_node = property(lambda self: SimpleNamespace(
        predict=lambda input_data, output_mode, predictions_cache, fold_id: _make_ts_output()
    ))

    try:
        result = pipeline.predict_tensordata(SimpleNamespace(
            features=np.array([[1.0]])), output_mode='labels')
    finally:
        pipeline_module.tensordata_to_input_data = original_bridge
        Pipeline.is_fitted = original_is_fitted
        Pipeline.root_node = original_root_node

    assert calls == [(False, False, None), (False, True, pipeline)]
    assert result[0] == 'postprocessed'
    assert result[1] == (
        'reduced', ('ts-updated', ('predict-indexed', ('input', 'tensor-2'))))
    assert result[3] == 'labels'
