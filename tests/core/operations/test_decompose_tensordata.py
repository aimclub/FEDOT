from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

from fedot.core.caching.hasher import Hasher
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.data_operations.decompose import (
    DecomposerClassImplementation, DecomposerRegImplementation,
)
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tensor_input import prepare_tensor_input
from fedot.core.pipelines.verification_rules import has_no_conflicts_in_decompose
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def data(target, prediction, task=TaskTypesEnum.regression, device='cpu'):
    return TensorData(features=torch.arange(len(target) * 2, dtype=torch.float64).reshape(-1, 2),
                      target=target, predict=prediction, idx=torch.arange(len(target)),
                      task=Task(task), data_type=DataTypesEnum.tabular).to(device)


@pytest.mark.parametrize('columns', [0, 1, 2])
def test_regression_residuals_preserve_multioutput_and_do_not_mutate_input(columns):
    target = torch.tensor([[100., 50.], [80., 40.]], dtype=torch.float64)
    prediction = torch.tensor([[90., 45.], [85., 30.]], dtype=torch.float64)
    if columns < 2:
        target, prediction = target[:, :1], prediction[:, :1]
    if columns == 0:
        target, prediction = target[:, 0], prediction[:, 0]
    source = data(target, prediction)
    original = deepcopy(source)
    fitted = DecomposerRegImplementation().fit(source)
    result = fitted.transform_for_fit(source)
    torch.testing.assert_close(result.target, (target - prediction).reshape(2, -1))
    assert result.predict is None
    assert result.features is source.features
    assert source == original
    inference = fitted.transform(replace(source, target=None))
    assert inference.target is None and inference.predict is None
    torch.testing.assert_close(fitted.transform(source).target, source.target)


@pytest.mark.parametrize('full_probs', [False, True])
def test_binary_classes_and_probability_columns(full_probs):
    target = torch.tensor([2, 7, 2])
    positive = torch.tensor([.2, .8, .6], dtype=torch.float64)
    prediction = torch.stack((1 - positive, positive), dim=1) if full_probs else positive
    source = data(target, prediction, TaskTypesEnum.classification)
    fitted = DecomposerClassImplementation().fit(source)
    result = fitted.transform_for_fit(source)
    torch.testing.assert_close(result.target[:, 0], (target == 7).to(positive.dtype) - positive)
    assert result.task.task_type is TaskTypesEnum.regression
    assert source.task.task_type is TaskTypesEnum.classification
    inference = fitted.transform(replace(source, target=None))
    assert inference.target is None and inference.task.task_type is TaskTypesEnum.regression


def test_next_regressor_learns_signed_probability_errors_as_numeric_targets():
    source = data(torch.tensor([2, 7]), torch.tensor([.2, .2], dtype=torch.float64),
                  TaskTypesEnum.classification)
    source = replace(source, features=torch.tensor([[0.], [1.]], dtype=torch.float64))
    residual = DecomposerClassImplementation().fit(source).transform_for_fit(source)
    expected = torch.tensor([-.2, .8], dtype=torch.float64)
    learner = PipelineNode('linear')
    training = learner.fit(residual)
    assert training.task.task_type is TaskTypesEnum.regression
    torch.testing.assert_close(training.target[:, 0], expected)
    torch.testing.assert_close(training.predict.reshape(-1), expected)
    torch.testing.assert_close(learner.predict(replace(residual, target=None)).predict.reshape(-1), expected)


def test_multiclass_fitted_vocabulary_survives_later_subset():
    target = torch.tensor([2, 7, 11])
    probabilities = torch.tensor([[.8, .1, .1], [.1, .7, .2], [.1, .2, .7]])
    source = data(target, probabilities, TaskTypesEnum.classification)
    fitted = pickle.loads(pickle.dumps(DecomposerClassImplementation().fit(source)))
    subset = source.subset_rows(torch.tensor([0, 1]))
    result = fitted.transform_for_fit(subset)
    torch.testing.assert_close(result.target, torch.eye(3)[:2] - probabilities[:2])
    assert result.target.shape == (2, 3)


@pytest.mark.parametrize('full_probs', [False, True])
def test_binary_multioutput_keeps_one_residual_per_target(full_probs):
    target = torch.tensor([[2, 10], [7, 20], [2, 20]])
    probability = torch.tensor([[.2, .3], [.8, .6], [.6, .7]])
    prediction = torch.stack((1 - probability, probability), dim=-1) if full_probs else probability
    source = data(target, prediction, TaskTypesEnum.classification)
    result = DecomposerClassImplementation().fit(source).transform_for_fit(source)
    expected = (target == torch.tensor([7, 20])).float() - probability
    torch.testing.assert_close(result.target, expected)


def test_invalid_regression_broadcast_and_missing_class_axis_are_rejected():
    source = data(torch.tensor([1., 2., 3.]), torch.ones(3, 3))
    with pytest.raises(ValueError, match='matching target'):
        DecomposerRegImplementation().fit(source).transform_for_fit(source)
    source = replace(source, target=torch.tensor([2, 7, 11]), task=Task(TaskTypesEnum.classification),
                     predict=torch.ones(3, 4))
    with pytest.raises(ValueError, match='vocabulary'):
        DecomposerClassImplementation().fit(source).transform_for_fit(source)


@pytest.mark.parametrize('reverse', [False, True])
def test_preparation_aligns_prediction_and_preserves_feature_parent_schema(reverse):
    source = data(torch.tensor([100., 80., 60.]), torch.tensor([90., 85., 50.]))
    source = replace(source, categorical_idx=[1], numerical_idx=[0], idx_mapping={0: 5, 1: 9},
                     categorical_features=np.array([['red'], ['blue'], ['green']]), categorical_source_idx=[9],
                     features_names=['value', 'color'])
    model = PipelineNode('linear')
    feature = PipelineNode('scaling')
    feature_data = replace(source.subset_rows(torch.tensor([2, 0])), predict=None)
    parents, outputs = [model, feature], [source, feature_data]
    if reverse:
        parents.reverse(); outputs.reverse()
    prepared = prepare_tensor_input('decompose', parents, outputs)
    torch.testing.assert_close(prepared.idx, torch.tensor([0, 2]))
    torch.testing.assert_close(prepared.predict, torch.tensor([90., 50.]))
    assert prepared.idx_mapping == source.idx_mapping
    assert prepared.features_names == source.features_names
    assert prepared.categorical_features[:, 0].tolist() == ['red', 'green']
    result = DecomposerRegImplementation().fit(prepared).transform_for_fit(prepared)
    torch.testing.assert_close(result.target[:, 0], torch.tensor([10., 10.]))
    assert not hasattr(result, 'target_role') and not hasattr(result, 'flow_depth')


def test_model_as_feature_parent_uses_predictions_as_numeric_features():
    source = data(torch.tensor([100., 80.]), torch.tensor([90., 85.]))
    shallow = PipelineNode('linear')
    deeper = PipelineNode('ridge', nodes_from=[shallow])
    prepared = prepare_tensor_input('decompose', [shallow, deeper],
                                    [replace(source, predict=torch.tensor([20., 30.])), source])
    torch.testing.assert_close(prepared.features[:, 0], torch.tensor([20., 30.]))
    torch.testing.assert_close(prepared.predict, source.predict)
    assert prepared.numerical_idx == [0] and prepared.categorical_idx == []


def test_decompose_input_identity_includes_baseline_prediction_values():
    source = data(torch.tensor([1., 2.]), torch.tensor([.8, 1.7]))
    assert Hasher.hash(source) != Hasher.hash(replace(source, predict=source.predict + .1))


def test_prediction_hash_sampling_uses_sample_axis_even_for_three_dimensional_features():
    source = TensorData(features=torch.zeros(3, 12, 2), predict=torch.arange(3.),
                        task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.tabular)
    assert Hasher.hash(source) != Hasher.hash(replace(source, predict=source.predict + 1))


@pytest.mark.parametrize('classification', [False, True])
def test_complete_pipeline_returns_to_main_objective_and_survives_pickle(classification):
    features = torch.randn(48, 3, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    target = 2 * features[:, 0] + features[:, 1].square()
    if classification:
        target = (target > target.median()).long() * 5 + 2
    source = TensorData(features=features, target=target, idx=torch.arange(48), data_type=DataTypesEnum.tabular,
                        task=Task(TaskTypesEnum.classification if classification else TaskTypesEnum.regression))
    primary = PipelineNode('scaling')
    baseline = PipelineNode('logit' if classification else 'ridge', nodes_from=[primary])
    residual = PipelineNode('class_decompose' if classification else 'decompose', nodes_from=[primary, baseline])
    correction = PipelineNode('linear', nodes_from=[residual])
    root = PipelineNode('logit' if classification else 'ridge', nodes_from=[correction, baseline])
    pipeline = Pipeline(root)
    assert has_no_conflicts_in_decompose(pipeline)
    training = pipeline.fit(source)
    correction_output = correction.fit(source)
    assert correction_output.task.task_type is TaskTypesEnum.regression
    torch.testing.assert_close(correction_output.target, residual.fit(source).target)
    torch.testing.assert_close(training.target, target)
    assert training.task == source.task
    fitted_copy = pickle.loads(pickle.dumps(pipeline))
    inference = replace(source, target=None)
    torch.testing.assert_close(pipeline.predict(inference).predict, fitted_copy.predict(inference).predict)
    assert pipeline.predict(inference).task == source.task
    assert source.target is target


def test_fork_within_residual_branch_preserves_residual_target_before_outer_merge():
    features = torch.randn(32, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    source = TensorData(features=features, target=features[:, 0].square() + 100, idx=torch.arange(32),
                        task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.tabular)
    primary = PipelineNode('scaling')
    baseline = PipelineNode('ridge', nodes_from=[primary])
    residual = PipelineNode('decompose', nodes_from=[primary, baseline])
    left, right = PipelineNode('linear', nodes_from=[residual]), PipelineNode('ridge', nodes_from=[residual])
    ensemble = PipelineNode('ridge', nodes_from=[left, right])
    pipeline = Pipeline(PipelineNode('ridge', nodes_from=[baseline, ensemble]))
    training = pipeline.fit(source)
    expected = residual.fit(source).target
    torch.testing.assert_close(ensemble.fit(source).target, expected)
    torch.testing.assert_close(training.target, source.target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('classification', [False, True])
def test_preparation_and_residual_math_stay_on_cuda(classification, monkeypatch):
    source = data(torch.tensor([2, 7, 11]) if classification else torch.tensor([100., 80., 60.]),
                  torch.eye(3) * .7 + .1 if classification else torch.tensor([90., 85., 50.]),
                  TaskTypesEnum.classification if classification else TaskTypesEnum.regression, 'cuda')
    model, feature = PipelineNode('logit' if classification else 'linear'), PipelineNode('scaling')

    def forbidden(*args, **kwargs):
        pytest.fail('Decompose must not convert numerical tensors through CPU/NumPy')

    with monkeypatch.context() as patch:
        for method in ['cpu', 'numpy', 'tolist']:
            patch.setattr(torch.Tensor, method, forbidden)
        prepared = prepare_tensor_input('class_decompose' if classification else 'decompose',
                                        [model, feature], [source, replace(source, predict=None)])
        implementation = DecomposerClassImplementation() if classification else DecomposerRegImplementation()
        result = implementation.fit(prepared).transform_for_fit(prepared)
    assert result.features.is_cuda and result.target.is_cuda and result.idx.is_cuda
