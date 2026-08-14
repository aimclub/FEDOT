import numpy as np
import pytest
import torch
from dataclasses import replace

from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.evaluation.operation_implementations.data_operations.features_reducing import (
    TruncatedSVDImplementation,
)
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_truncated_svd_params,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.validation.errors import FedotValidationError


@pytest.fixture
def train_td():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    return TensorDataCreator.create(features, backend_name='cpu')


@pytest.mark.unit
def test_truncated_svd_reduces_features(train_td):
    impl = TruncatedSVDImplementation(OperationParameters(n_components=3))
    impl.fit(train_td)
    out = impl.transform(train_td)
    assert out.features.shape == (train_td.features.shape[0], 3)
    assert impl.components_ is not None
    assert not hasattr(impl, 'singular_values_')
    assert not hasattr(impl, 'explained_variance_')


@pytest.mark.unit
def test_truncated_svd_rejects_float_and_mle():
    with pytest.raises(FedotValidationError):
        validate_truncated_svd_params({'n_components': 0.7})
    with pytest.raises(FedotValidationError):
        validate_truncated_svd_params({'n_components': 'mle'})
    with pytest.raises(FedotValidationError):
        TruncatedSVDImplementation(OperationParameters(n_components=0.7))


@pytest.mark.unit
def test_truncated_svd_default_n_components_is_auto():
    impl = TruncatedSVDImplementation(OperationParameters())
    assert impl.params.get('n_components') == 'auto'


@pytest.mark.unit
def test_truncated_svd_auto_n_components_uses_half_features_budget(train_td):
    from fedot.core.operations.evaluation.operation_implementations.tools import (
        default_components_budget,
    )

    impl = TruncatedSVDImplementation(OperationParameters(n_components='auto'))
    impl.fit(train_td)
    expected = default_components_budget(impl.n_samples_, impl.n_features_)
    assert impl.n_components_ == expected
    assert impl.params.get('n_components') == expected
    out = impl.transform(train_td)
    assert out.features.shape == (train_td.features.shape[0], expected)


@pytest.mark.unit
def test_truncated_svd_keeps_nan_rows_on_transform(train_td):
    features = train_td.features.clone()
    features[0, 0] = float('nan')
    data = replace(train_td, features=features, fingerprint=None)

    impl = TruncatedSVDImplementation(OperationParameters(n_components=2))
    impl.fit(data)
    out = impl.transform(data)
    assert out.features.shape[0] == data.features.shape[0]
    assert torch.isnan(out.features[0]).all()


@pytest.mark.unit
def test_pipeline_node_wires_truncated_svd(train_td):
    node = PipelineNode('truncated_svd')
    node.parameters = {'n_components': 2}
    node.fit(train_td)
    predicted = node.predict(train_td)
    assert predicted.features.shape == (train_td.features.shape[0], 2)
