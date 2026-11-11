from dataclasses import replace
import pickle
import random

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.data_operations.tabular import non_binary_columns
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.data_operations.common_preprocessing import PreprocessingDataOperationStrategy
from fedot.core.operations.strategies.sources import DataSourceStrategy
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum


def data(device='cpu'):
    features = torch.tensor([[10., 0., 0.], [20., 1., 1.], [30., 2., 0.], [40., 0., 1.]], dtype=torch.float64)
    return TensorData(features=features, target=torch.tensor([0, 0, 1, 1]), idx=['a', 'b', 'c', 'd'],
                      task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.tabular,
                      categorical_idx=[1], numerical_idx=[0, 2],
                      categorical_features=np.array(['red', 'green', 'blue', 'red'], dtype=object)[:, None],
                      categorical_source_idx=[1], idx_mapping={0: 0, 1: 1, 2: 2},
                      features_names=['age', 'color', 'flag']).to(device)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('operation,params,reference', [
    ('scaling', {}, StandardScaler()),
    ('scaling', {'with_mean': False}, StandardScaler(with_mean=False)),
    ('scaling', {'with_std': False}, StandardScaler(with_std=False)),
    ('normalization', {}, MinMaxScaler()),
    ('normalization', {'feature_range': (-2, 3), 'clip': True}, MinMaxScaler(feature_range=(-2, 3), clip=True)),
])
def test_scalers_keep_original_order_and_scale_multivalued_label_codes(operation, params, reference, dtype):
    source = data()
    source.features = source.features.to(dtype)
    original = source.features.clone()
    strategy = PreprocessingDataOperationStrategy(operation, OperationParameters(**params))
    fitted = pickle.loads(pickle.dumps(strategy.fit(source)))
    output = strategy.predict_for_fit(fitted, source)
    expected = original.numpy().copy()
    expected[:, :2] = reference.fit_transform(expected[:, :2])

    torch.testing.assert_close(output.features, torch.as_tensor(expected))
    torch.testing.assert_close(source.features, original)
    assert output.categorical_idx == [1]
    assert output.idx_mapping == source.idx_mapping
    assert output.features_names == source.features_names
    assert output.predict is None
    raw = output.get_not_encoded_data()
    np.testing.assert_array_equal(raw.categorical_features, source.categorical_features)
    torch.testing.assert_close(raw.numerical_features, output.features[:, [0, 2]])


def test_binary_policy_is_fitted_on_training_data_and_handles_nan_counts_like_numpy():
    features = torch.tensor([[0., float('nan'), 1.], [1., float('nan'), 1.], [float('nan'), 2., 1.]])
    # NaN is one distinct value in the historical NumPy rule.
    assert non_binary_columns(features).tolist() == [0]
    strategy = PreprocessingDataOperationStrategy('scaling')
    fitted = strategy.fit(data())
    inference = data()
    inference.features[0, 2] = 5  # A new value does not refit the binary-column selection.
    output = strategy.predict(fitted, inference)
    assert output.features[0, 2] == 5


@pytest.mark.parametrize('operation', ['scaling', 'normalization', 'poly_features'])
def test_integer_inputs_are_not_truncated_by_floating_transformations(operation):
    source = data()
    source.features = source.features.long()
    output = PreprocessingDataOperationStrategy(operation)
    result = output.predict(output.fit(source), source)

    assert result.features.dtype == torch.float64
    assert source.features.dtype == torch.int64
    assert torch.isfinite(result.features).all()


@pytest.mark.parametrize('degree,interaction', [(2, False), (3, True), ((2, 3), False)])
@pytest.mark.parametrize('width', [4, 12])
def test_native_polynomial_values_match_sklearn_and_preserve_sampled_columns(degree, interaction, width):
    features = torch.randn(20, width, dtype=torch.float64, generator=torch.Generator().manual_seed(83))
    source = TensorData(features=features, task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.tabular)
    strategy = PreprocessingDataOperationStrategy(
        'poly_features', OperationParameters(degree=degree, interaction_only=interaction))
    random.seed(83)
    fitted = strategy.fit(source)
    output = strategy.predict(fitted, source)
    chosen = fitted.columns_to_take
    selected = features.numpy() if chosen is None else features.numpy()[:, chosen]
    reference = PolynomialFeatures(degree=degree, interaction_only=interaction, include_bias=False)
    expected = reference.fit_transform(selected)
    if chosen is not None:
        expected = np.concatenate((features.numpy(), expected[:, len(chosen):]), axis=1)

    torch.testing.assert_close(output.features, torch.as_tensor(expected))
    assert output.numerical_idx == list(range(output.features.shape[1]))
    assert len(output.idx_mapping) == output.features.shape[1]
    assert output.fingerprint is None


def test_polynomial_features_preserve_original_category_sources_and_names():
    source = data()
    strategy = PreprocessingDataOperationStrategy('poly_features')
    output = strategy.predict(strategy.fit(source), source)

    torch.testing.assert_close(output.features[:, :3], source.features)
    assert output.categorical_idx == source.categorical_idx
    assert len(output.features_names) == output.features.shape[1]
    np.testing.assert_array_equal(output.get_not_encoded_data().categorical_features, source.categorical_features)
    assert set(output.idx_mapping[pos] for pos in range(3, output.features.shape[1])).isdisjoint({0, 1, 2})


@pytest.mark.parametrize('operation', ['scaling', 'normalization', 'poly_features'])
def test_unified_transform_runs_before_a_tensor_model(operation):
    transform = PipelineNode(operation)
    model = PipelineNode('logit', nodes_from=[transform])
    pipeline = Pipeline(model)
    source = data()
    pipeline.fit(source)
    result = pipeline.predict(replace(source, target=None))

    assert result.predict.shape == (4,)
    assert torch.isfinite(result.predict).all()


def test_encoding_nodes_are_removed_and_legacy_imputer_is_not_offered_for_tabular_tasks():
    repository = OperationTypesRepository('data_operation')
    names = {info.id for info in repository.operations}
    assert not names.intersection({'one_hot_encoding', 'label_encoding'})
    assert repository.operation_info_by_id('simple_imputation').task_type == [TaskTypesEnum.ts_forecasting]
    for name in ['scaling', 'normalization', 'poly_features', 'pca', 'truncated_svd', 'kernel_pca', 'fast_ica']:
        assert repository.operation_info_by_id(name).supported_strategies is PreprocessingDataOperationStrategy


def test_tensor_data_source_is_an_immutable_passthrough():
    source = data()
    source.predict = torch.zeros(4, dtype=source.features.dtype)
    strategy = DataSourceStrategy('data_source_table')
    result = strategy.predict(strategy.fit(source), source)

    assert result.features is source.features
    assert result.predict is None
    assert source.predict is not None
    assert result.idx == source.idx


def test_optional_imputation_can_fill_features_created_by_an_intermediate_projection():
    source = data()
    source.features[0, 0] = float('nan')
    projection = PipelineNode('pca')
    projection.parameters = {'n_components': 2}
    imputation = PipelineNode('optional_preprocessing', nodes_from=[projection])
    imputation.parameters = {'auto': False, 'imputation_method': 'mean', 'scaling_method': 'none', 'use_cache': False}
    model = PipelineNode('logit', nodes_from=[imputation])
    pipeline = Pipeline(model)
    pipeline.fit(source)
    output = pipeline.predict(replace(source, target=None))

    assert torch.isfinite(output.predict).all()
    repository = OperationTypesRepository('data_operation')
    assert repository.operation_info_by_id('optional_preprocessing').allowed_positions == ['any']


@pytest.mark.parametrize('use_cache', [False, True])
def test_optional_imputation_owns_its_output_and_hashes_an_uncached_tensor(use_cache):
    source = TensorData(features=torch.tensor([[1., float('nan')], [3., 2.], [5., 4.]]),
                        numerical_idx=[0, 1], task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.tabular)
    source.predict = torch.zeros(3)
    original = source.features.clone()
    service = OptionalTabularService(use_cache=use_cache).fit(source, {PreprocessingStepEnum.imputation: None})
    result = service.predict(source)

    torch.testing.assert_close(source.features, original, equal_nan=True)
    assert source.fingerprint is None
    assert result.fingerprint is not None
    assert result.predict is None
    assert source.predict is not None
    assert torch.isfinite(result.features).all()
    assert result is not source


@pytest.mark.parametrize('operation,params', [
    ('kernel_pca', {'n_components': 2}),
    ('fast_ica', {'n_components': 2, 'random_state': 19}),
    ('dask_pca', {'n_components': 2}),
])
def test_external_projection_uses_a_tensor_boundary_and_resets_feature_metadata(operation, params):
    if operation == 'dask_pca':
        pytest.importorskip('dask_ml')
    source = data()
    source.predict = torch.zeros(4, dtype=source.features.dtype)
    strategy = PreprocessingDataOperationStrategy(operation, OperationParameters(**params))
    fitted = strategy.fit(source)
    result = strategy.predict(fitted, source)
    expected = fitted.pca.transform(fitted.preprocess_input(source))
    expected = fitted.postprocess_input(expected)

    torch.testing.assert_close(result.features, torch.as_tensor(expected))
    assert result.predict is None
    assert result.categorical_idx == []
    assert result.categorical_features is None
    assert result.numerical_idx == [0, 1]
    assert source.categorical_idx == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('operation', ['scaling', 'normalization', 'poly_features'])
@pytest.mark.parametrize('fit_device,predict_device', [('cuda', 'cuda'), ('cpu', 'cuda'), ('cuda', 'cpu')])
def test_torch_transformations_follow_input_device(operation, fit_device, predict_device):
    source = data(fit_device)
    strategy = PreprocessingDataOperationStrategy(operation)
    fitted = strategy.fit(source)
    inference = data(predict_device)
    output = strategy.predict(fitted, inference)

    assert output.features.device == inference.device
    assert torch.isfinite(output.features).all()
    assert output.target.device == inference.device
    if operation != 'poly_features':
        torch.testing.assert_close(output.features[:, 2], inference.features[:, 2])
