from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.data_operations import sklearn_selectors
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.data_operations.common_preprocessing import PreprocessingDataOperationStrategy
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


OPERATIONS = [
    ('rfe_lin_class', LogisticRegression, TaskTypesEnum.classification),
    ('rfe_non_lin_class', DecisionTreeClassifier, TaskTypesEnum.classification),
    ('rfe_lin_reg', LinearRegression, TaskTypesEnum.regression),
    ('rfe_non_lin_reg', DecisionTreeRegressor, TaskTypesEnum.regression),
]


def data(task_type, target_shape='vector', device='cpu', width=8):
    features = torch.randn(72, width, dtype=torch.float64, generator=torch.Generator().manual_seed(914))
    target = features[:, 0] * 3 + features[:, -1].square()
    second = features[:, -1] * 7
    if task_type is TaskTypesEnum.classification:
        target = (target > target.median()).long()
        second = (second > 0).long()
    if target_shape == 'column':
        target = target[:, None]
    elif target_shape == 'multi':
        target = torch.stack((target, second), dim=1)
    return TensorData(features=features, target=target, idx=list(range(100, 172)), task=Task(task_type),
                      data_type=DataTypesEnum.tabular, numerical_idx=list(range(width)),
                      features_names=[f'x{i}' for i in range(width)],
                      idx_mapping={i: i + 10 for i in range(width)}).to(device)


@pytest.mark.parametrize('operation,estimator,task_type', OPERATIONS)
@pytest.mark.parametrize('target_shape', ['vector', 'column', 'multi'])
def test_rfe_preserves_sklearn_masks_targets_and_values(operation, estimator, task_type, target_shape, monkeypatch):
    source = data(task_type, target_shape)
    params = OperationParameters(n_features_to_select=3, step=2)
    strategy = PreprocessingDataOperationStrategy(operation, params)
    np.random.seed(42)
    fitted = strategy.fit(source)
    target = source.target.numpy()
    if estimator is LogisticRegression and target.ndim == 2:
        target = target[:, 0]
    reference = RFE(estimator(), n_features_to_select=3, step=2)
    np.random.seed(42)
    reference.fit(source.features.numpy(), target)
    np.testing.assert_array_equal(fitted.remain_features_mask.numpy(), reference.support_)
    fitted = pickle.loads(pickle.dumps(fitted))

    def unexpected_conversion(*args, **kwargs):
        pytest.fail('RFE transform must not convert features to NumPy')

    monkeypatch.setattr(sklearn_selectors, 'as_numpy', unexpected_conversion)
    result = strategy.predict_for_fit(fitted, source)
    torch.testing.assert_close(result.features, source.features[:, torch.from_numpy(reference.support_)],
                               rtol=0, atol=0)
    torch.testing.assert_close(result.target, source.target, rtol=0, atol=0)
    torch.testing.assert_close(strategy.predict(fitted, source).features, result.features, rtol=0, atol=0)
    assert result.predict is None
    assert result.idx == source.idx
    assert source.features.shape[1] == 8
    assert source.numerical_idx == list(range(8))


@pytest.mark.parametrize('params,expected_width', [({}, 4), ({'n_features_to_select': 3}, 3), ({'step': 2}, 4)])
def test_rfe_partial_parameters_keep_sklearn_defaults(params, expected_width):
    source = data(TaskTypesEnum.regression)
    strategy = PreprocessingDataOperationStrategy('rfe_lin_reg', OperationParameters(**params))
    result = strategy.predict(strategy.fit(source), source)
    assert result.features.shape == (72, expected_width)


@pytest.mark.parametrize('operation,estimator,task_type', OPERATIONS)
@pytest.mark.parametrize('vector', [False, True])
def test_single_feature_is_passthrough_without_sklearn_fit(operation, estimator, task_type, vector):
    source = data(task_type, width=1)
    if vector:
        source = replace(source, features=source.features[:, 0])
    strategy = PreprocessingDataOperationStrategy(operation)
    fitted = strategy.fit(source)
    result = strategy.predict(fitted, source)
    torch.testing.assert_close(result.features, source.features, rtol=0, atol=0)
    assert result.features.shape == source.features.shape
    assert result.features_names == source.features_names
    assert result.idx_mapping == source.idx_mapping
    assert not hasattr(fitted.operation, 'support_')


def test_refitting_on_more_columns_does_not_reuse_single_feature_passthrough():
    strategy = PreprocessingDataOperationStrategy('rfe_lin_reg')
    fitted = strategy.fit(data(TaskTypesEnum.regression, width=1))
    source = data(TaskTypesEnum.regression)
    fitted.fit(source)
    assert fitted.transform(source).features.shape[1] == 4


def test_invalid_classification_target_is_not_retried_or_hidden(monkeypatch):
    source = data(TaskTypesEnum.regression)
    implementation = sklearn_selectors.LinearClassFSImplementation()
    calls = []
    original = implementation.operation.fit

    def fit(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation.operation, 'fit', fit)
    with pytest.raises(ValueError, match='Unknown label type'):
        implementation.fit(source)
    assert len(calls) == 1


@pytest.mark.parametrize('operation,estimator,task_type', OPERATIONS)
def test_rfe_runs_in_a_tensor_pipeline_and_keeps_catalogue_task_restrictions(operation, estimator, task_type):
    source = data(task_type)
    selector = PipelineNode(operation)
    selector.parameters = {'n_features_to_select': 3}
    model = PipelineNode('logit' if task_type is TaskTypesEnum.classification else 'linear', nodes_from=[selector])
    pipeline = Pipeline(model)
    pipeline.fit(source)
    result = pipeline.predict(replace(source, target=None))
    assert result.predict.shape == (72,)
    assert torch.isfinite(result.predict).all()
    info = OperationTypesRepository('data_operation').operation_info_by_id(operation)
    assert info.supported_strategies is PreprocessingDataOperationStrategy
    assert info.task_type == ([TaskTypesEnum.classification] if task_type is TaskTypesEnum.classification
                              else [TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting])
    assert 'non-default' in info.tags


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('operation,estimator,task_type', OPERATIONS)
@pytest.mark.parametrize('fit_device,predict_device', [('cuda', 'cuda'), ('cpu', 'cuda'), ('cuda', 'cpu')])
def test_cpu_rfe_fit_and_torch_selection_follow_input_device(operation, estimator, task_type,
                                                           fit_device, predict_device, monkeypatch):
    source = data(task_type, device=fit_device)
    strategy = PreprocessingDataOperationStrategy(operation)
    fitted = strategy.fit(source)
    inference = data(task_type, device=predict_device)

    def unexpected_conversion(*args, **kwargs):
        pytest.fail('RFE transform must not transfer features to CPU')

    monkeypatch.setattr(sklearn_selectors, 'as_numpy', unexpected_conversion)
    result = strategy.predict(fitted, inference)
    assert result.features.device == inference.device
    assert result.target.device == inference.device
    torch.testing.assert_close(result.features, inference.features[:, fitted.selected_columns.to(inference.device)])
