import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.supplementary_data import SupplementaryData
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
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
        data_type=DataTypesEnum.ts,
    )



def _make_classification_input(is_auto_preprocessed: bool):
    supplementary_data = SupplementaryData(is_auto_preprocessed=is_auto_preprocessed)
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

    result = pipeline._postprocess(None, _make_ts_output(), output_mode='labels')

    assert pipeline.preprocessor.calls == ['restore_index', 'inverse_target_encoding']
    assert result.predict.tolist() == [2.0, 3.0]



def test_pipeline_fit_skips_preprocessing_when_input_is_marked_auto_preprocessed():
    pipeline = Pipeline(use_input_preprocessing=False)
    pipeline._preprocess = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('_preprocess should not be called'))
    pipeline._assign_data_to_nodes = lambda data: data
    pipeline._fit = lambda input_data=None, predictions_cache=None, fold_id=None: 'ok'
    input_data = _make_classification_input(is_auto_preprocessed=True)

    result = pipeline.fit(input_data)

    assert result == 'ok'
