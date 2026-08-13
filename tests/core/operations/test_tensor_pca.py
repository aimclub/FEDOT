import numpy as np
import pytest
import torch
from dataclasses import replace

from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.abstract_node import TensorDataOperationImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.features_reducing import (
    TensorPCAImplementation,
)
from fedot.core.operations.evaluation.tensor_transform import TensorTransformStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture
def train_td():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 5)).astype(np.float32)
    return TensorDataCreator.create(features, backend_name='cpu')


@pytest.fixture
def train_td_with_nan(train_td):
    features = train_td.features.clone()
    features[0, 1] = float('nan')
    features[3, 0] = float('nan')
    return replace(train_td, features=features, fingerprint=None)


@pytest.mark.unit
def test_pca_implementation_reduces_features(train_td):
    impl = TensorPCAImplementation(OperationParameters(n_components=2))
    fitted = impl.fit(train_td)
    assert isinstance(fitted, TensorDataOperationImplementation)

    transformed = impl.transform(train_td)
    assert transformed.features.shape == (train_td.features.shape[0], 2)
    assert transformed.numerical_idx == [0, 1]
    assert transformed.categorical_idx == []
    assert not torch.isnan(transformed.features).any()


@pytest.mark.unit
def test_pca_supports_variance_ratio(train_td):
    impl = TensorPCAImplementation(OperationParameters(n_components=0.9))
    impl.fit(train_td)
    out = impl.transform(train_td)
    assert 1 <= out.features.shape[1] <= train_td.features.shape[1]


@pytest.mark.unit
def test_pca_init_loads_default_n_components():
    impl = TensorPCAImplementation(OperationParameters())
    assert impl.params.get('n_components') == 0.7


@pytest.mark.unit
def test_pca_params_schema_rejects_invalid_n_components():
    from fedot.validation.errors import FedotValidationError
    from fedot.core.operations.evaluation.operation_implementations.schema import (
        validate_pca_params,
    )

    with pytest.raises(FedotValidationError):
        validate_pca_params({'n_components': 0})

    with pytest.raises(FedotValidationError):
        validate_pca_params({'n_components': 'auto'})


@pytest.mark.unit
def test_pca_params_schema_fills_missing_n_components():
    from fedot.core.operations.evaluation.operation_implementations.schema import (
        validate_pca_params,
    )

    assert validate_pca_params({})['n_components'] == 0.7
    assert validate_pca_params({'n_components': None})['n_components'] == 0.7


@pytest.mark.unit
def test_pca_fit_drops_nan_rows_transform_keeps_them(train_td_with_nan):
    impl = TensorPCAImplementation(OperationParameters(n_components=2))
    impl.fit(train_td_with_nan)

    assert impl.n_samples_ == train_td_with_nan.features.shape[0] - 2

    out = impl.transform(train_td_with_nan)
    assert out.features.shape == (train_td_with_nan.features.shape[0], 2)
    assert torch.isnan(out.features[0]).all()
    assert torch.isnan(out.features[3]).all()
    assert not torch.isnan(out.features[1]).any()


@pytest.mark.unit
def test_pca_fit_rejects_too_few_finite_samples(train_td):
    from fedot.validation.errors import FedotValidationError

    features = train_td.features[:3].clone()
    features[0, 0] = float('nan')
    features[1, 1] = float('nan')
    data = replace(train_td, features=features, fingerprint=None)
    impl = TensorPCAImplementation(OperationParameters(n_components=1))
    with pytest.raises(FedotValidationError, match='at least 2 finite samples'):
        impl.fit(data)


@pytest.mark.unit
def test_pipeline_node_wires_pca_tensor_strategy():
    node = PipelineNode('pca')

    assert node.name == 'pca'
    assert isinstance(node.operation, DataOperation)
    assert node.operation.operation_type == 'pca'

    node.operation._init(
        task=Task(TaskTypesEnum.classification),
        params=node.parameters,
        n_samples_data=40,
    )
    assert isinstance(node.operation._eval_strategy, TensorTransformStrategy)


@pytest.mark.unit
def test_pipeline_node_pca_fit_predict(train_td):
    node = PipelineNode('pca')
    node.parameters = {'n_components': 3}

    node.fit(train_td)
    predicted = node.predict(train_td)

    assert predicted.features.shape == (train_td.features.shape[0], 3)
    assert not torch.isnan(predicted.features).any()
