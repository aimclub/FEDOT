from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import fedot.api.api_utils.api_composer as composer_module
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.pipelines.ensembling.config import ChunkedEnsembleConfig
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _FakeCache:
    def __init__(self, cache_dir=None, use_stats=False):
        self.cache_dir = cache_dir
        self.use_stats = use_stats
        self.was_reset = False

    def reset(self):
        self.was_reset = True


class _FakeParams(dict):
    timeout = 1
    n_jobs = -1


class _FakeChunk:
    def __init__(self, size=3):
        self.task = type('TaskContainer', (), {'task_type': TaskTypesEnum.classification})()
        self.idx = list(range(size))
        self.target = np.zeros(size, dtype=int)
        self.features = np.zeros((size, 1))
        self.data_type = DataTypesEnum.table


class _FakePipeline:
    is_fitted = True
    use_input_preprocessing = False
    preprocessor = None

    def fit(self, data, n_jobs):
        self.is_fitted = True

    def predict(self, input_data, output_mode='default', predictions_cache=None, fold_id=None):
        return OutputData(
            idx=input_data.idx,
            features=input_data.features,
            task=input_data.task,
            data_type=input_data.data_type,
            target=input_data.target,
            predict=np.zeros(len(input_data.idx), dtype=int),
        )


class _FakeChunkPipeline:
    def __init__(self, is_fitted: bool):
        self.is_fitted = is_fitted
        self.fit_calls = []

    def fit(self, data, n_jobs):
        self.fit_calls.append((data, n_jobs))
        self.is_fitted = True

    def fit_tensordata(self, data, n_jobs):
        self.fit_calls.append((data, n_jobs))
        self.is_fitted = True


class _FakeHistory:
    def __init__(self, is_empty: bool):
        self._is_empty = is_empty

    def is_empty(self):
        return self._is_empty


def test_api_composer_init_cache_uses_typed_cache_plan(monkeypatch):
    monkeypatch.setattr(composer_module, 'OperationsCache', _FakeCache)
    monkeypatch.setattr(composer_module, 'PreprocessingCache', _FakeCache)
    monkeypatch.setattr(composer_module, 'PredictionsCache', _FakeCache)

    params = _FakeParams(
        use_operations_cache=True,
        use_preprocessing_cache=True,
        use_predictions_cache=True,
        use_input_preprocessing=False,
        cache_dir='cache_dir',
        use_stats=True,
    )

    composer = ApiComposer(params, metrics=['f1'])

    assert isinstance(composer.operations_cache, _FakeCache)
    assert composer.operations_cache.was_reset is True
    assert composer.preprocessing_cache is None
    assert isinstance(composer.predictions_cache, _FakeCache)
    assert composer.predictions_cache.was_reset is True


def test_obtain_ensemble_model_uses_predefined_model_for_chunks(monkeypatch):
    class _FakeEnsemble:
        def __init__(self, pipelines, validation_metric, ensemble_method, pipeline_infos, **kwargs):
            self.pipelines = pipelines
            self.pipeline_infos = pipeline_infos
            self.validation_metric = validation_metric

    captured_calls = []
    api_preprocessor = SimpleNamespace(name='api-preprocessor')

    class _FakePredefinedModel:
        def __init__(self, predefined_model, data, log, use_input_preprocessing=True, api_preprocessor=None):
            captured_calls.append(SimpleNamespace(
                predefined_model=predefined_model,
                data=data,
                api_preprocessor=api_preprocessor,
            ))

        def fit(self):
            return _FakePipeline()

    monkeypatch.setattr(composer_module, 'PipelineEnsemble', _FakeEnsemble)
    monkeypatch.setattr(composer_module, 'PredefinedModel', _FakePredefinedModel)

    params = _FakeParams(use_input_preprocessing=True)
    composer = ApiComposer(params, metrics=['f1'])

    def _forbidden_obtain_model(*args, **kwargs):
        raise AssertionError('obtain_model must not be called when predefined_model is set')

    monkeypatch.setattr(composer, 'obtain_model', _forbidden_obtain_model, raising=False)

    chunks = [_FakeChunk(), _FakeChunk()]
    ensemble, best_models, histories = composer.obtain_ensemble_model(
        chunks,
        predefined_model='logit',
        api_preprocessor=api_preprocessor,
    )

    assert len(captured_calls) == len(chunks)
    assert [call.predefined_model for call in captured_calls] == ['logit', 'logit']
    assert [call.data for call in captured_calls] == chunks
    assert all(call.api_preprocessor is not api_preprocessor for call in captured_calls)
    assert captured_calls[0].api_preprocessor is not captured_calls[1].api_preprocessor
    assert len(ensemble.pipelines) == len(chunks)
    assert ensemble.validation_metric == 'f1'
    assert len(best_models) == len(chunks)
    assert histories == []


def test_obtain_ensemble_model_uses_external_validation_for_chunk_composition(monkeypatch):
    class _FakeEnsemble:
        def __init__(self, pipelines, validation_metric, ensemble_method, pipeline_infos, **kwargs):
            self.pipelines = pipelines
            self.pipeline_infos = pipeline_infos
            self.validation_metric = validation_metric

    monkeypatch.setattr(composer_module, 'PipelineEnsemble', _FakeEnsemble)
    monkeypatch.setattr(composer_module, 'calculate_validation_metrics', lambda **kwargs: {'f1': -1.0})

    params = _FakeParams(use_input_preprocessing=False)
    composer = ApiComposer(params, metrics=['f1'])
    captured_calls = []

    def _fake_obtain_model_with_external_validation(train_data, validation_data):
        captured_calls.append(SimpleNamespace(train_data=train_data, validation_data=validation_data))
        return _FakePipeline(), [_FakePipeline()], None

    monkeypatch.setattr(composer, 'obtain_model_with_external_validation',
                        _fake_obtain_model_with_external_validation, raising=False)

    chunks = [_FakeChunk(), _FakeChunk()]
    validation_data = _FakeChunk(size=2)
    ensemble, best_models, histories = composer.obtain_ensemble_model(
        chunks,
        validation_data=validation_data,
    )

    assert [call.train_data for call in captured_calls] == chunks
    assert [call.validation_data for call in captured_calls] == [validation_data, validation_data]
    assert len(ensemble.pipelines) == len(chunks)
    assert len(best_models) == len(chunks)
    assert histories == []


def test_obtain_ensemble_model_raises_when_successful_chunk_threshold_is_not_met(monkeypatch):
    class _FakeEnsemble:
        def __init__(self, pipelines, validation_metric, ensemble_method, pipeline_infos, **kwargs):
            self.pipelines = pipelines
            self.pipeline_infos = pipeline_infos
            self.validation_metric = validation_metric

    monkeypatch.setattr(composer_module, 'PipelineEnsemble', _FakeEnsemble)

    params = _FakeParams(use_input_preprocessing=False)
    composer = ApiComposer(params, metrics=['f1'])

    def _fake_obtain_model_with_external_validation(train_data, validation_data):
        if train_data.idx[0] == 0:
            raise ValueError('synthetic chunk failure')
        return _FakePipeline(), [_FakePipeline()], None

    monkeypatch.setattr(composer, 'obtain_model_with_external_validation',
                        _fake_obtain_model_with_external_validation, raising=False)

    first_chunk = _FakeChunk()
    second_chunk = _FakeChunk()
    second_chunk.idx = [10, 11, 12]
    validation_data = _FakeChunk(size=2)

    try:
        composer.obtain_ensemble_model(
            [first_chunk, second_chunk],
            validation_data=validation_data,
            chunked_ensemble_config=ChunkedEnsembleConfig(min_successful_chunks=2),
        )
    except ValueError as ex:
        message = str(ex)
    else:
        raise AssertionError('Expected ValueError when successful chunk threshold is not met')

    assert 'at least 2 successful chunks' in message
    assert 'synthetic chunk failure' in message
    assert "'chunk_idx': 0" in message


def test_obtain_ensemble_model_accepts_single_success_when_threshold_is_met(monkeypatch):
    class _FakeEnsemble:
        def __init__(self, pipelines, validation_metric, ensemble_method, pipeline_infos, **kwargs):
            self.pipelines = pipelines
            self.pipeline_infos = pipeline_infos
            self.validation_metric = validation_metric

    monkeypatch.setattr(composer_module, 'PipelineEnsemble', _FakeEnsemble)
    monkeypatch.setattr(composer_module, 'calculate_validation_metrics', lambda **kwargs: {'f1': -1.0})

    params = _FakeParams(use_input_preprocessing=False)
    composer = ApiComposer(params, metrics=['f1'])

    def _fake_obtain_model_with_external_validation(train_data, validation_data):
        if train_data.idx[0] == 0:
            raise ValueError('synthetic chunk failure')
        return _FakePipeline(), [_FakePipeline()], None

    monkeypatch.setattr(composer, 'obtain_model_with_external_validation',
                        _fake_obtain_model_with_external_validation, raising=False)

    first_chunk = _FakeChunk()
    second_chunk = _FakeChunk()
    second_chunk.idx = [10, 11, 12]
    validation_data = _FakeChunk(size=2)

    ensemble, best_models, histories = composer.obtain_ensemble_model(
        [first_chunk, second_chunk],
        validation_data=validation_data,
        chunked_ensemble_config=ChunkedEnsembleConfig(min_successful_chunks=1),
    )

    assert len(ensemble.pipelines) == 1
    assert len(best_models) == 1
    assert histories == []


def _dummy_composer():
    composer = ApiComposer.__new__(ApiComposer)
    composer.params = _FakeParams()
    composer.params.n_jobs = 2
    return composer


def _dummy_tensor_composer(initial_assumption=None):
    composer = _dummy_composer()
    composer.params['initial_assumption'] = initial_assumption
    composer.params['preset'] = 'auto'
    composer.params.data = {'cv_folds': 3}
    composer.operations_cache = 'operations-cache'
    composer.preprocessing_cache = 'preprocessing-cache'
    composer.log = SimpleNamespace(message=lambda *_args, **_kwargs: None)
    composer.timer = SimpleNamespace(
        launch_assumption_fit=lambda n_folds: nullcontext(),
        assumption_fit_spend_time_single_fold=SimpleNamespace(total_seconds=lambda: 1.2),
        assumption_fit_spend_time=SimpleNamespace(total_seconds=lambda: 3.6),
    )
    return composer


def _classification_input(n_samples: int = 4) -> InputData:
    return InputData(
        idx=np.arange(n_samples),
        features=np.zeros((n_samples, 1)),
        target=np.zeros(n_samples, dtype=int),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_tensor_initial_assumption_is_required_for_composition():
    composer = _dummy_tensor_composer(initial_assumption=None)

    with pytest.raises(NotImplementedError, match='without initial assumption'):
        composer.propose_and_fit_initial_assumption(SimpleNamespace(name='tensor-data'))


def test_tensor_initial_assumption_uses_user_pipeline_without_auto_builder(monkeypatch):
    captured = {}
    initial_pipeline = SimpleNamespace(name='initial')
    fitted_pipeline = SimpleNamespace(name='fitted')

    class _FakeAssumptionsHandler:
        def __init__(self, data):
            captured['data'] = data

        def propose_assumptions_with_tensordata(self, initial_assumption):
            captured['initial_assumption'] = initial_assumption
            return [initial_assumption]

        def fit_assumption_and_check_correctness_with_tensordata(
                self, pipeline, operations_cache=None, preprocessing_cache=None, eval_n_jobs=-1):
            captured['fit_pipeline'] = pipeline
            captured['operations_cache'] = operations_cache
            captured['preprocessing_cache'] = preprocessing_cache
            captured['eval_n_jobs'] = eval_n_jobs
            return fitted_pipeline

        def propose_preset(self, preset, timer, n_jobs):
            captured['preset'] = preset
            captured['preset_n_jobs'] = n_jobs
            return 'fast_train'

    monkeypatch.setattr(composer_module, 'AssumptionsHandler', _FakeAssumptionsHandler)

    tensor_data = SimpleNamespace(name='tensor-data')
    composer = _dummy_tensor_composer(initial_assumption=initial_pipeline)
    composer.params.n_jobs = 4

    initial_assumptions, fitted_assumption = composer.propose_and_fit_initial_assumption(tensor_data)

    assert captured['data'] is tensor_data
    assert captured['initial_assumption'] is initial_pipeline
    assert captured['fit_pipeline'] is not initial_pipeline
    assert captured['fit_pipeline'].name == 'initial'
    assert captured['operations_cache'] == 'operations-cache'
    assert captured['preprocessing_cache'] == 'preprocessing-cache'
    assert captured['eval_n_jobs'] == 4
    assert captured['preset'] == 'auto'
    assert captured['preset_n_jobs'] == 4
    assert composer.params['preset'] == 'fast_train'
    assert initial_assumptions == [initial_pipeline]
    assert fitted_assumption is fitted_pipeline


def test_chunk_pipeline_is_fitted_after_successful_composition_history():
    data = _classification_input()
    pipeline = _FakeChunkPipeline(is_fitted=False)
    composer = _dummy_composer()

    result = composer._fit_chunk_pipeline_for_ensemble(pipeline, data, _FakeHistory(is_empty=False))

    assert result is pipeline
    assert pipeline.is_fitted is True
    assert pipeline.fit_calls == [(data, 2)]


def test_chunk_pipeline_is_not_refitted_without_composition_history_when_already_fitted():
    data = _classification_input()
    pipeline = _FakeChunkPipeline(is_fitted=True)
    composer = _dummy_composer()

    composer._fit_chunk_pipeline_for_ensemble(pipeline, data, _FakeHistory(is_empty=True))

    assert pipeline.fit_calls == []
