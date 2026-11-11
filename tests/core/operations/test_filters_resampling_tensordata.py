from copy import deepcopy
from dataclasses import replace
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sklearn.utils import resample

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.data_operations import sklearn_filters
from fedot.core.operations.implementations.data_operations.resampling import ResampleImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.data_operations.common_preprocessing import PreprocessingDataOperationStrategy
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


FILTERS = ['ransac_lin_reg', 'ransac_non_lin_reg', 'isolation_forest_reg', 'isolation_forest_class']


def filter_data(operation, device='cpu'):
    features = torch.randn(64, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(19))
    features[:, -1] = torch.arange(64) % 2
    target = 2 * features[:, 0] - features[:, 2]
    target[::7] += 15
    classification = operation.endswith('class')
    if classification:
        target = torch.arange(64) % 3
    return TensorData(
        features=features, target=target[:, None], idx=torch.arange(64) + 100, predict=torch.zeros(64),
        task=Task(TaskTypesEnum.classification if classification else TaskTypesEnum.regression),
        data_type=DataTypesEnum.tabular, numerical_idx=[0, 1, 2], categorical_idx=[3],
        categorical_features=np.array([f'category_{i % 2}' for i in range(64)], dtype=object)[:, None],
        categorical_source_idx=[3], idx_mapping={i: i for i in range(4)}, fingerprint='source',
    ).to(device)


def filter_strategy(operation):
    options = {'random_state': 19}
    if operation.startswith('ransac'):
        options.update(min_samples=.4, residual_threshold=.5, max_trials=40)
    else:
        options.update(n_estimators=12, contamination=.15)
    return PreprocessingDataOperationStrategy(operation, OperationParameters(**options))


@pytest.mark.parametrize('operation', FILTERS)
def test_filter_training_selects_aligned_rows_and_inference_does_not_filter(operation, monkeypatch):
    source = filter_data(operation)
    original = deepcopy(source)
    strategy = filter_strategy(operation)
    fitted = strategy.fit(source)
    if operation.startswith('ransac'):
        mask = torch.as_tensor(fitted.operation.inlier_mask_)
    else:
        mask = torch.from_numpy(fitted.operation.predict(source.features.numpy()) == 1)
        if operation.endswith('class') and not fitted._is_inlier_mask_correct(source.target, source.target[mask]):
            mask = torch.ones(len(source.features), dtype=torch.bool)
    result = strategy.predict_for_fit(fitted, source)
    torch.testing.assert_close(result.features, source.features[mask], rtol=0, atol=0)
    torch.testing.assert_close(result.target, source.target[mask], rtol=0, atol=0)
    torch.testing.assert_close(result.idx, source.idx[mask])
    np.testing.assert_array_equal(result.categorical_features, source.categorical_features[mask.numpy()])
    assert result.idx_mapping == source.idx_mapping
    assert result.categorical_idx == [3]
    assert result.predict is None
    assert source == original

    def unexpected_model_call(*args, **kwargs):
        pytest.fail('Inference must not call sklearn or convert the matrix to NumPy')

    monkeypatch.setattr(sklearn_filters, 'as_numpy', unexpected_model_call)
    inference = strategy.predict(fitted, replace(source, target=None))
    assert inference.features is source.features
    assert inference.target is None
    assert inference.predict is None
    torch.testing.assert_close(inference.idx, source.idx)


@pytest.mark.parametrize('failure', [ValueError('no consensus'), np.linalg.LinAlgError('SVD did not converge'), None])
def test_ransac_retry_policy_preserves_unfiltered_fallback(failure):
    source = filter_data('ransac_lin_reg')
    implementation = sklearn_filters.LinearRegRANSACImplementation(OperationParameters(residual_threshold=.1))

    def trial(*args, **kwargs):
        if failure is not None:
            raise failure
        implementation.operation.inlier_mask_ = np.arange(64) < 1

    implementation.operation.fit = Mock(side_effect=trial)
    assert implementation.fit(source) is implementation
    result = implementation.transform_for_fit(source)
    assert implementation.operation.fit.call_count == 10
    assert implementation.operation.inlier_mask_ is None
    torch.testing.assert_close(result.features, source.features)


def test_ransac_default_none_threshold_uses_sklearn_mad_and_can_retry():
    source = filter_data('ransac_lin_reg')
    implementation = sklearn_filters.LinearRegRANSACImplementation(OperationParameters(residual_threshold=None))
    thresholds = []

    def trial(*args, **kwargs):
        thresholds.append(implementation.operation.residual_threshold)
        implementation.operation.inlier_mask_ = np.arange(64) < (1 if len(thresholds) == 1 else 50)

    implementation.operation.fit = Mock(side_effect=trial)
    implementation.fit(source)
    y = source.target.numpy()
    expected = np.median(np.abs(y - np.median(y)))
    np.testing.assert_allclose(thresholds, [expected, expected + np.abs(y - y.mean()).mean() / 10])


@pytest.mark.parametrize('retained,accepted', [([0, 1, 2], True), ([0, 1], False), ([], False),
                                            ([0, 1] + [2] * 5, True)])
def test_classification_filter_retains_classes_and_at_least_one_fifth(retained, accepted):
    source = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
    result = sklearn_filters.IsolationForestClassImplementation._is_inlier_mask_correct(
        source, torch.tensor(retained, dtype=torch.long))
    assert result is accepted


def test_classification_filter_rejects_an_empty_selection_before_constructing_tensor_data(monkeypatch):
    source = filter_data('isolation_forest_class')
    implementation = sklearn_filters.IsolationForestClassImplementation()
    monkeypatch.setattr(implementation, '_get_inlier_mask', lambda data: torch.zeros(64, dtype=torch.bool))
    result = implementation.transform_for_fit(source)
    assert result.features is source.features
    torch.testing.assert_close(result.target, source.target)


def resample_data(column=False, device='cpu', host_categories=True):
    target = torch.tensor([2] * 4 + [7] * 6)
    return TensorData(
        features=torch.arange(20, dtype=torch.float64).reshape(10, 2),
        target=target[:, None] if column else target, idx=torch.arange(10) + 100,
        task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.tabular,
        numerical_idx=[0], categorical_idx=[1] if host_categories else [],
        categorical_features=np.array([f'row_{i}' for i in range(10)], dtype=object)[:, None]
        if host_categories else None,
        categorical_source_idx=[1] if host_categories else [], idx_mapping={0: 0, 1: 1},
    ).to(device)


@pytest.mark.parametrize('balance', ['expand_minority', 'reduce_majority'])
@pytest.mark.parametrize('ratio', [None, 0, .5, 1, 1.5])
@pytest.mark.parametrize('replace_samples', [False, True])
@pytest.mark.parametrize('column', [False, True])
def test_resample_matches_sklearn_for_the_same_draws(balance, ratio, replace_samples, column, monkeypatch):
    source = resample_data(column)
    original = deepcopy(source)
    normalized_ratio = ratio if ratio is not None and 0 < ratio <= 1 else 1
    expand = balance == 'expand_minority'
    selected = np.arange(4) if expand else np.arange(4, 10)
    count = round(4 + 2 * normalized_ratio) if expand else round(6 - 2 * normalized_ratio)
    effective_replace = replace_samples or count >= len(selected)
    # Share random draws to compare the algorithms independently of NumPy/Torch RNGs.
    draws = resample(np.arange(len(selected)), replace=effective_replace, n_samples=count, random_state=19)
    expected_rows = np.concatenate((selected[draws], np.arange(4, 10))) if expand else np.concatenate(
        (np.arange(4), selected[draws]))

    def randint(high, size, **kwargs):
        assert high == len(selected) and size == (count,)
        return torch.as_tensor(draws, device=kwargs['device'])

    def randperm(size, **kwargs):
        assert size == len(selected)
        permutation = resample(np.arange(size), replace=False, n_samples=size, random_state=19)
        return torch.as_tensor(permutation, device=kwargs['device'])

    monkeypatch.setattr(torch, 'randint', randint)
    monkeypatch.setattr(torch, 'randperm', randperm)
    implementation = ResampleImplementation(OperationParameters(
        balance=balance, replace=replace_samples, balance_ratio=ratio))
    result = implementation.transform_for_fit(source)
    rows = torch.from_numpy(expected_rows)
    torch.testing.assert_close(result.features, source.features[rows], rtol=0, atol=0)
    torch.testing.assert_close(result.target, source.target[rows], rtol=0, atol=0)
    np.testing.assert_array_equal(result.categorical_features, source.categorical_features[expected_rows])
    assert implementation.replace == effective_replace
    assert implementation.balance_ratio == normalized_ratio
    assert result.target.dtype == source.target.dtype
    assert result.target.ndim == source.target.ndim
    torch.testing.assert_close(result.idx, torch.arange(len(expected_rows)))
    assert result.idx_mapping == source.idx_mapping
    assert result.predict is None
    assert source == original


@pytest.mark.parametrize('target', [[0] * 5 + [1] * 5, [0] * 10, list(range(10))])
def test_resample_skips_balanced_or_nonbinary_targets(target):
    source = replace(resample_data(), target=torch.tensor(target))
    result = ResampleImplementation().transform_for_fit(source)
    assert result.features is source.features
    torch.testing.assert_close(result.idx, source.idx)


@pytest.mark.parametrize('balance,replace_samples', [('expand_minority', True), ('reduce_majority', False)])
def test_resample_is_reproducible_and_uses_the_expected_replacement_policy(balance, replace_samples):
    source = resample_data()
    strategy = PreprocessingDataOperationStrategy('resample', OperationParameters(balance=balance, balance_ratio=1))
    fitted = pickle.loads(pickle.dumps(strategy.fit(source)))
    state = torch.random.get_rng_state().clone()
    first = strategy.predict_for_fit(fitted, source)
    second = strategy.predict_for_fit(fitted, source)
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    assert torch.equal(state, torch.random.get_rng_state())
    assert fitted.replace == replace_samples
    _, counts = torch.unique(first.target, return_counts=True)
    assert torch.equal(counts, torch.tensor([6, 6] if balance == 'expand_minority' else [4, 4]))
    sampled = first.features[:6] if balance == 'expand_minority' else first.features[4:]
    assert (len(sampled.unique(dim=0)) < len(sampled)) == replace_samples
    assert strategy.predict(fitted, source).features is source.features


@pytest.mark.parametrize('operation', FILTERS + ['resample'])
def test_filter_or_resample_runs_before_a_tensor_model(operation):
    source = resample_data() if operation == 'resample' else filter_data(operation)
    transform = PipelineNode(operation)
    if operation != 'resample':
        transform.parameters = filter_strategy(operation).params_for_fit.to_dict()
    model = 'logit' if source.task.task_type is TaskTypesEnum.classification else 'linear'
    pipeline = Pipeline(PipelineNode(model, nodes_from=[transform]))
    pipeline.fit(source)
    result = pipeline.predict(replace(source, target=None, predict=None))
    assert len(result.predict) == len(source.features)
    assert torch.isfinite(result.predict).all()
    info = OperationTypesRepository('data_operation').operation_info_by_id(operation)
    assert info.supported_strategies is PreprocessingDataOperationStrategy


@pytest.mark.parametrize('operation', ['ransac_lin_reg', 'isolation_forest_reg'])
def test_filtered_and_unfiltered_branches_align_training_rows_by_original_labels(operation):
    source = filter_data(operation)
    filtered = PipelineNode(operation)
    filtered.parameters = filter_strategy(operation).params_for_fit.to_dict()
    pipeline = Pipeline(PipelineNode('linear', nodes_from=[filtered, PipelineNode('scaling')]))
    training = pipeline.fit(source)
    expected = filtered.fitted_operation.transform_for_fit(source)
    torch.testing.assert_close(training.idx, expected.idx)
    torch.testing.assert_close(training.target, expected.target)
    assert len(pipeline.predict(source).predict) == len(source.features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('operation', FILTERS)
def test_sklearn_filters_return_aligned_cuda_data(operation):
    source = filter_data(operation, 'cuda')
    strategy = filter_strategy(operation)
    fitted = strategy.fit(source)
    result = strategy.predict_for_fit(fitted, source)
    assert result.features.is_cuda and result.target.is_cuda and result.idx.is_cuda
    rows = result.idx.cpu().numpy() - 100
    np.testing.assert_array_equal(result.categorical_features, source.categorical_features[rows])
    assert len(strategy.predict(fitted, source).features) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('balance', ['expand_minority', 'reduce_majority'])
def test_native_resample_does_not_move_tensors_to_host(balance, monkeypatch):
    source = resample_data(device='cuda', host_categories=False)
    implementation = ResampleImplementation(OperationParameters(balance=balance))

    def forbidden(*args, **kwargs):
        pytest.fail('Native resampling must not move tensors through CPU/NumPy')

    with monkeypatch.context() as patch:
        for name in ['cpu', 'numpy', 'tolist']:
            patch.setattr(torch.Tensor, name, forbidden)
        result = implementation.transform_for_fit(source)
    assert result.features.is_cuda and result.target.is_cuda and result.idx.is_cuda
    _, counts = torch.unique(result.target, return_counts=True)
    assert torch.equal(counts, torch.tensor([6, 6] if balance == 'expand_minority' else [4, 4], device='cuda'))
