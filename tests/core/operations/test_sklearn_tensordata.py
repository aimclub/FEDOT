from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.models.sklearn import (
    SklearnClassificationStrategy, SklearnClusteringStrategy, SklearnRegressionStrategy,
)
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def make_data(problem, multi_output=False, device='cpu'):
    generator = torch.Generator().manual_seed(41)
    features = torch.rand(40, 4, generator=generator)
    target = features[:, 0] + features[:, 1]
    if problem == 'classification':
        target = (target > 1).long()
    if multi_output:
        target = torch.stack((target, target.roll(1)), dim=1)
    data = TensorData(features=features, target=target, idx=torch.arange(len(features)),
                      task=Task(TaskTypesEnum(problem)), data_type=DataTypesEnum.tabular)
    return data.to(device)


@pytest.mark.parametrize('strategy_class', [
    SklearnClassificationStrategy, SklearnRegressionStrategy, SklearnClusteringStrategy,
])
def test_repository_uses_one_sklearn_family(strategy_class):
    repository = OperationTypesRepository('model')
    for operation in strategy_class._operations_by_types:
        info = repository.operation_info_by_id(operation)
        assert info.current_strategy(info.task_type[0]) is strategy_class
        assert 'tensor-only' in info.tags


@pytest.mark.parametrize('operation,params', [
    ('logit', {}), ('bernb', {}), ('multinb', {}), ('dt', {}),
    ('rf', {'n_estimators': 2}), ('mlp', {'hidden_layer_sizes': (8,), 'max_iter': 2}),
])
@pytest.mark.parametrize('multi_output', [False, True])
def test_classification_modes_keep_all_targets_and_metadata(operation, params, multi_output):
    data = make_data('classification', multi_output)
    # Non-consecutive labels for single-target classification must survive inference.
    if not multi_output:
        data.target = data.target * 5 + 2
    strategy = SklearnClassificationStrategy(operation, OperationParameters(**params))
    fitted = pickle.loads(pickle.dumps(strategy.fit(data)))
    test_data = replace(data, target=None)
    features = data.features.numpy()
    for mode in ['labels', 'probs', 'full_probs', 'default']:
        strategy.output_mode = mode
        output = strategy.predict_for_fit(fitted, test_data)
        expected = fitted.predict(features) if mode == 'labels' else fitted.predict_proba(features)
        if isinstance(expected, list):
            expected = np.stack(expected, axis=1)
        elif mode != 'labels' and multi_output:
            expected = np.stack((1 - expected, expected), axis=-1)
        if mode in ['probs', 'default']:
            expected = expected[..., 1]
        torch.testing.assert_close(output.predict, torch.as_tensor(expected))
        torch.testing.assert_close(output.idx, test_data.idx)
        assert output.features is test_data.features
        assert output.preparation_state is test_data.preparation_state
        assert output.target is None
        assert test_data.predict is None
    assert isinstance(fitted, MultiOutputClassifier) == (operation in ['logit', 'bernb', 'multinb'] and multi_output)


def test_fitting_another_model_does_not_change_multi_output_state():
    multi_data = make_data('classification', multi_output=True)
    strategy = SklearnClassificationStrategy('rf', OperationParameters(n_estimators=2))
    multi_model = strategy.fit(multi_data)
    single_model = strategy.fit(make_data('classification'))
    strategy.output_mode = 'full_probs'

    assert multi_model.is_multi_target
    assert not single_model.is_multi_target
    assert 'is_multi_target' not in strategy.operation_impl.__dict__
    assert strategy.predict(multi_model, multi_data).predict.shape == (40, 2, 2)


@pytest.mark.parametrize('operation', ['logit', 'rf', 'mlp'])
def test_multiclass_probabilities_keep_the_class_axis(operation):
    data = make_data('classification')
    data.target = torch.arange(40) % 3
    params = {'n_estimators': 2} if operation == 'rf' else (
        {'hidden_layer_sizes': (8,), 'max_iter': 2} if operation == 'mlp' else {})
    strategy = SklearnClassificationStrategy(operation, OperationParameters(**params))
    fitted = strategy.fit(data)
    strategy.output_mode = 'probs'

    output = strategy.predict(fitted, replace(data, target=None))

    assert output.predict.shape == (40, 3)
    torch.testing.assert_close(output.predict, torch.as_tensor(fitted.predict_proba(data.features.numpy())))


@pytest.mark.parametrize('operation,params', [
    ('linear', {}), ('rfr', {'n_estimators': 2}), ('gbr', {'n_estimators': 2}),
])
@pytest.mark.parametrize('columns', [1, 2])
def test_regression_target_shape_and_wrapper(operation, params, columns):
    data = make_data('regression', multi_output=columns == 2)
    if columns == 1:
        data.target = data.target[:, None]
    strategy = SklearnRegressionStrategy(operation, OperationParameters(**params))
    fitted = strategy.fit(data)
    output = strategy.predict(fitted, replace(data, target=None))

    assert output.predict.shape == ((40,) if columns == 1 else (40, 2))
    assert isinstance(fitted, MultiOutputRegressor) == (operation == 'gbr' and columns == 2)
    torch.testing.assert_close(output.predict, torch.as_tensor(fitted.predict(data.features.numpy())))


@pytest.mark.parametrize('operation,problem', [('logit', 'classification'), ('rfr', 'regression')])
def test_existing_sklearn_ids_fit_in_tensor_pipeline(operation, problem):
    data = make_data(problem)
    node = PipelineNode(operation)
    if operation == 'rfr':
        node.parameters = {'n_estimators': 2}
    pipeline = Pipeline(node)
    pipeline.fit(data)
    output = pipeline.predict(replace(data, target=None))

    assert isinstance(output, TensorData)
    assert torch.isfinite(output.predict).all()
    assert output.predict.shape == (40,)


def test_clustering_ignores_target_and_keeps_two_cluster_default():
    data = replace(make_data('regression'), task=Task(TaskTypesEnum.clustering), target=None)
    strategy = SklearnClusteringStrategy('kmeans')
    fitted = strategy.fit(data)
    output = strategy.predict(fitted, data)

    assert fitted.n_clusters == 2
    torch.testing.assert_close(output.predict, torch.as_tensor(fitted.predict(data.features.numpy())))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('operation,problem,strategy_class', [
    ('rf', 'classification', SklearnClassificationStrategy),
    ('ridge', 'regression', SklearnRegressionStrategy),
    ('kmeans', 'clustering', SklearnClusteringStrategy),
])
def test_cpu_estimators_return_predictions_to_cuda(operation, problem, strategy_class):
    data = make_data('regression' if problem == 'clustering' else problem, device='cuda')
    if problem == 'clustering':
        data = replace(data, task=Task(TaskTypesEnum.clustering), target=None)
    params = {'n_estimators': 2} if operation == 'rf' else {}
    strategy = strategy_class(operation, OperationParameters(**params))
    strategy.output_mode = 'full_probs'
    fitted = strategy.fit(data)
    output = strategy.predict(fitted, data)

    assert output.predict.device == data.device
    assert torch.isfinite(output.predict).all()
