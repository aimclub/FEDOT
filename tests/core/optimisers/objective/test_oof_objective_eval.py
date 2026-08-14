import logging

import numpy as np
import pytest
from sklearn.datasets import load_iris

from fedot import Fedot, create_data
from fedot.core.data.input_data.data import InputData
from fedot.core.data.multimodal.supplementary_data import SupplementaryData
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.optimisers.objective.evaluation_contracts import EvaluationComplete, EvaluationReused
from fedot.core.optimisers.objective.oof_objective_eval import PipelineOOFObjectiveEvaluate
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.oof import PipelineOOFExecutor, build_oof_splits
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements_rules import PipelineEvaluationMode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _iris_input_data() -> InputData:
    dataset = load_iris()
    return InputData(
        idx=np.arange(len(dataset.target)),
        features=dataset.data,
        target=dataset.target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(),
    )


def _iris_tensor_data():
    dataset = load_iris()
    return create_data(dataset.data, target=dataset.target, use_cache=False)


@pytest.mark.unit
def test_oof_splits_cover_each_row_once():
    data = _iris_input_data()

    splits = build_oof_splits(data, cv_folds=3, random_seed=42)
    test_ids = np.concatenate([split.test_ids for split in splits])

    assert len(splits) == 3
    assert np.array_equal(np.sort(test_ids), np.arange(len(data.target)))


@pytest.mark.unit
def test_oof_objective_evaluate_with_data_operation():
    data = _iris_tensor_data()
    pipeline = Pipeline(PipelineNode('logit', nodes_from=[PipelineNode('scaling')]))
    objective_eval = PipelineOOFObjectiveEvaluate(
        objective=MetricsObjective(ClassificationMetricsEnum.ROCAUC),
        tensor_data=data,
        cv_folds=3,
    )

    fitness = objective_eval.evaluate(pipeline)

    assert fitness.valid
    assert fitness.value is not None
    assert not pipeline.is_fitted


@pytest.mark.unit
def test_oof_objective_evaluate_with_several_parents_and_metrics():
    data = _iris_tensor_data()
    pipeline = Pipeline(PipelineNode('logit', nodes_from=[PipelineNode('rf'), PipelineNode('scaling')]))
    objective_eval = PipelineOOFObjectiveEvaluate(
        objective=MetricsObjective(
            [ClassificationMetricsEnum.ROCAUC, ClassificationMetricsEnum.accuracy],
            is_multi_objective=True,
        ),
        tensor_data=data,
        cv_folds=3,
    )

    fitness = objective_eval.evaluate(pipeline)

    assert fitness.valid
    assert len(fitness.values) == 2


@pytest.mark.unit
def test_oof_evaluator_exposes_complete_and_reused_outcomes():
    evaluator = PipelineOOFObjectiveEvaluate(
        objective=MetricsObjective(ClassificationMetricsEnum.ROCAUC),
        tensor_data=_iris_tensor_data(),
        cv_folds=3,
    )

    first = evaluator.evaluate_result(Pipeline(PipelineNode('logit')))
    reused = evaluator.evaluate_result(Pipeline(PipelineNode('logit')))

    assert isinstance(first, EvaluationComplete)
    assert first.expected_folds == 1
    assert isinstance(reused, EvaluationReused)
    evaluator.clear_results()
    assert evaluator.last_outcome is None


@pytest.mark.unit
def test_each_oof_prediction_uses_a_model_without_its_rows(monkeypatch):
    data = _iris_input_data()
    pipeline = Pipeline(PipelineNode('logit', nodes_from=[PipelineNode('scaling')]))
    fitted_rows = {}

    for node in pipeline.nodes:
        original_fit = node.operation.fit
        original_predict = node.operation.predict

        def record_fit(*args, _node=node, _fit=original_fit, **kwargs):
            fitted_rows[(_node.descriptive_id, kwargs['fold_id'])] = set(kwargs['data'].idx)
            return _fit(*args, **kwargs)

        def assert_predict_is_oof(*args, _node=node, _predict=original_predict, **kwargs):
            train_rows = fitted_rows[(_node.descriptive_id, kwargs['fold_id'])]
            assert train_rows.isdisjoint(set(kwargs['data'].idx))
            return _predict(*args, **kwargs)

        monkeypatch.setattr(node.operation, 'fit', record_fit)
        monkeypatch.setattr(node.operation, 'predict', assert_predict_is_oof)

    output = PipelineOOFExecutor(pipeline, data, cv_folds=3).predict_oof(output_mode='labels')

    assert len(output.predict) == len(data.target)
    assert len(fitted_rows) == 3 * len(pipeline.nodes)


@pytest.mark.unit
def test_data_operation_uses_same_fold_fit_for_secondary_train_input(monkeypatch):
    data = _iris_input_data()
    scaling_node = PipelineNode('scaling')
    model_node = PipelineNode('logit', nodes_from=[scaling_node])
    pipeline = Pipeline(model_node)
    transformed_by_train_rows = {}

    original_scaling_fit = scaling_node.operation.fit
    original_model_fit = model_node.operation.fit

    def record_scaling_fit(*args, **kwargs):
        fitted_operation, transformed = original_scaling_fit(*args, **kwargs)
        transformed_by_train_rows[frozenset(kwargs['data'].idx)] = np.asarray(transformed.predict)
        return fitted_operation, transformed

    def assert_fold_consistent_input(*args, **kwargs):
        expected = transformed_by_train_rows[frozenset(kwargs['data'].idx)]
        assert np.allclose(kwargs['data'].features, expected)
        return original_model_fit(*args, **kwargs)

    monkeypatch.setattr(scaling_node.operation, 'fit', record_scaling_fit)
    monkeypatch.setattr(model_node.operation, 'fit', assert_fold_consistent_input)

    PipelineOOFExecutor(pipeline, data, cv_folds=3).predict_oof()


@pytest.mark.unit
def test_oof_evaluation_mode_available_from_api():
    data = _iris_input_data()
    initial_assumption = Pipeline(PipelineNode('logit'))
    model = Fedot(
        problem='classification',
        timeout=0.1,
        preset='fast_train',
        max_depth=2,
        max_arity=2,
        pop_size=2,
        num_of_generations=1,
        cv_folds=3,
        evaluation_mode='oof',
        available_operations=['logit'],
        initial_assumption=initial_assumption,
        with_tuning=False,
        show_progress=False,
        logging_level=logging.CRITICAL,
    )

    train_data = model.create_data(data.features, target=data.target)
    pipeline = model.fit(train_data)

    assert isinstance(pipeline, Pipeline)
    assert model.params.composer_requirements.evaluation_mode is PipelineEvaluationMode.oof
