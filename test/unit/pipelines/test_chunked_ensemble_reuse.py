import numpy as np
from sklearn.datasets import load_iris

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.params import ApiParams
from fedot.core.data.input_data.data import InputData
from fedot.core.pipelines.ensembling.config import ChunkedEnsembleConfig, validate_chunked_ensemble_config
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_chunked_ensemble_config_accepts_reuse_previous_best_initial_population():
    config = validate_chunked_ensemble_config({
        'reuse_previous_best_initial_population': True,
    })

    assert config.reuse_previous_best_initial_population is True
    assert config.to_dict()['reuse_previous_best_initial_population'] is True


def test_prepare_reused_initial_pipelines_returns_unfitted_copies():
    pipeline = Pipeline(PipelineNode('logit'))
    pipeline.root_node.fitted_operation = object()

    reusable = ApiComposer._prepare_reused_initial_pipelines([pipeline])

    assert len(reusable) == 1
    assert reusable[0] is not pipeline
    assert not reusable[0].is_fitted
    assert pipeline.is_fitted


def test_chunked_ensemble_reuses_previous_best_pipelines_when_enabled(monkeypatch):
    data = _get_iris_data()
    chunks = [
        data.subset_by_positions(np.arange(0, 40)),
        data.subset_by_positions(np.arange(40, 80)),
        data.subset_by_positions(np.arange(80, 120)),
    ]
    validation_data = data.subset_by_positions(np.arange(120, 150))
    composer = _build_test_composer()
    received_initial_pipelines = []

    def fake_obtain_model_with_external_validation(train_data, validation_data, initial_pipelines=None):
        received_initial_pipelines.append(initial_pipelines)
        pipeline = Pipeline(PipelineNode('logit'))
        candidate = Pipeline(PipelineNode('dt'))
        candidate.root_node.fitted_operation = object()
        return pipeline, [candidate], None

    def fake_fit_chunk_pipeline_for_ensemble(pipeline, chunk_data, history):
        pipeline.root_node.fitted_operation = object()
        return pipeline

    monkeypatch.setattr(
        composer,
        'obtain_model_with_external_validation',
        fake_obtain_model_with_external_validation,
    )
    monkeypatch.setattr(
        composer,
        '_fit_chunk_pipeline_for_ensemble',
        fake_fit_chunk_pipeline_for_ensemble,
    )

    _, best_models, _ = composer.obtain_ensemble_model(
        train_data_list=chunks,
        validation_data=validation_data,
        chunked_ensemble_config=ChunkedEnsembleConfig(
            reuse_previous_best_initial_population=True,
        ),
    )

    assert received_initial_pipelines[0] is None
    assert len(received_initial_pipelines[1]) == 1
    assert not received_initial_pipelines[1][0].is_fitted
    assert received_initial_pipelines[1][0] is not best_models[0][0]
    assert len(received_initial_pipelines[2]) == 1


def test_chunked_ensemble_does_not_reuse_previous_best_pipelines_by_default(monkeypatch):
    data = _get_iris_data()
    chunks = [
        data.subset_by_positions(np.arange(0, 40)),
        data.subset_by_positions(np.arange(40, 80)),
    ]
    validation_data = data.subset_by_positions(np.arange(120, 150))
    composer = _build_test_composer()
    received_initial_pipelines = []

    def fake_obtain_model_with_external_validation(train_data, validation_data, initial_pipelines=None):
        received_initial_pipelines.append(initial_pipelines)
        pipeline = Pipeline(PipelineNode('logit'))
        candidate = Pipeline(PipelineNode('dt'))
        return pipeline, [candidate], None

    def fake_fit_chunk_pipeline_for_ensemble(pipeline, chunk_data, history):
        pipeline.root_node.fitted_operation = object()
        return pipeline

    monkeypatch.setattr(
        composer,
        'obtain_model_with_external_validation',
        fake_obtain_model_with_external_validation,
    )
    monkeypatch.setattr(
        composer,
        '_fit_chunk_pipeline_for_ensemble',
        fake_fit_chunk_pipeline_for_ensemble,
    )

    composer.obtain_ensemble_model(
        train_data_list=chunks,
        validation_data=validation_data,
        chunked_ensemble_config=ChunkedEnsembleConfig(),
    )

    assert received_initial_pipelines == [None, None]


def _build_test_composer() -> ApiComposer:
    params = ApiParams(
        input_params={
            'with_tuning': False,
            'use_operations_cache': False,
            'use_preprocessing_cache': False,
            'use_predictions_cache': False,
        },
        problem='classification',
        timeout=0.1,
    )
    return ApiComposer(params, [ClassificationMetricsEnum.ROCAUC])


def _get_iris_data() -> InputData:
    dataset = load_iris()
    return InputData(
        idx=np.arange(len(dataset.target)),
        features=dataset.data,
        target=dataset.target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )
