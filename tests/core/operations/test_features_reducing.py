from dataclasses import replace

import numpy as np
import pytest
import torch

from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.abstract_node import TensorDataOperationImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.features_reducing import (
    PCAImplementation,
    TruncatedSVDImplementation,
)
from fedot.core.operations.evaluation.operation_implementations.rules import (
    SpectrumNComponentsMethod,
)
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_pca_params,
    validate_truncated_svd_params,
)
from fedot.core.operations.evaluation.operation_implementations.tools import (
    broken_stick_expectations,
    default_components_budget,
    n_components_from_broken_stick,
    n_components_from_elbow,
    resolve_spectrum_n_components,
)
from fedot.core.operations.evaluation.tensor_transform import TensorTransformStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.validation.errors import FedotValidationError


@pytest.fixture
def train_td():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    return TensorDataCreator.create(features, backend_name='cpu')


@pytest.fixture
def train_td_with_nan(train_td):
    features = train_td.features.clone()
    features[0, 1] = float('nan')
    features[3, 0] = float('nan')
    return replace(train_td, features=features, fingerprint=None)


# --- spectrum helpers ---


@pytest.mark.unit
def test_broken_stick_expectations_sum_to_one():
    n = 5
    b = broken_stick_expectations(n)
    assert b.shape == (n,)
    assert torch.allclose(b.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(b[:-1] >= b[1:])


@pytest.mark.unit
def test_broken_stick_expectations_rejects_non_positive_n():
    with pytest.raises(FedotValidationError, match='n >= 1'):
        broken_stick_expectations(0)


@pytest.mark.unit
def test_n_components_from_broken_stick_keeps_dominant_leading():
    props = torch.tensor([0.55, 0.30, 0.05, 0.05, 0.05], dtype=torch.float32)
    assert n_components_from_broken_stick(props) == 2


@pytest.mark.unit
def test_n_components_from_elbow_on_clear_knee():
    spectrum = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.5, 0.2], dtype=torch.float32)
    assert n_components_from_elbow(spectrum) == 4


@pytest.mark.unit
def test_spectrum_selectors_handle_degenerate_inputs():
    assert n_components_from_elbow(torch.tensor([1.0])) == 1
    assert n_components_from_broken_stick(torch.tensor([1.0])) == 1
    assert n_components_from_elbow(torch.zeros(4)) == 1
    assert n_components_from_broken_stick(torch.zeros(4)) == 1


@pytest.mark.unit
def test_resolve_spectrum_uses_enum_mapping():
    singular_values = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.5, 0.2], dtype=torch.float32)
    k = resolve_spectrum_n_components(
        SpectrumNComponentsMethod.ELBOW,
        singular_values=singular_values,
        max_components=6,
    )
    assert k == 4


@pytest.mark.unit
def test_resolve_spectrum_rejects_unknown_method_and_missing_input():
    with pytest.raises(FedotValidationError, match='Unsupported spectrum'):
        resolve_spectrum_n_components('nope', singular_values=torch.ones(3), max_components=3)

    with pytest.raises(FedotValidationError, match='requires singular_values or proportions'):
        resolve_spectrum_n_components('elbow', max_components=3)


# --- PCA ---


@pytest.mark.unit
def test_pca_implementation_reduces_features(train_td):
    impl = PCAImplementation(OperationParameters(n_components=2))
    fitted = impl.fit(train_td)
    assert isinstance(fitted, TensorDataOperationImplementation)

    transformed = impl.transform(train_td)
    assert transformed.features.shape == (train_td.features.shape[0], 2)
    assert transformed.numerical_idx == [0, 1]
    assert transformed.categorical_idx == []
    assert not torch.isnan(transformed.features).any()


@pytest.mark.unit
def test_pca_supports_variance_ratio(train_td):
    impl = PCAImplementation(OperationParameters(n_components=0.9))
    impl.fit(train_td)
    out = impl.transform(train_td)
    assert 1 <= out.features.shape[1] <= train_td.features.shape[1]


@pytest.mark.unit
def test_pca_init_loads_default_n_components():
    impl = PCAImplementation(OperationParameters())
    assert impl.params.get('n_components') == 'auto'


@pytest.mark.unit
def test_pca_params_schema_rejects_invalid_n_components():
    with pytest.raises(FedotValidationError):
        validate_pca_params({'n_components': 0})

    with pytest.raises(FedotValidationError):
        validate_pca_params({'n_components': 'random'})


@pytest.mark.unit
def test_pca_params_schema_fills_missing_n_components():
    assert validate_pca_params({})['n_components'] == 'auto'
    assert validate_pca_params({'n_components': None})['n_components'] == 'auto'


@pytest.mark.unit
def test_pca_auto_n_components_uses_half_features_budget(train_td):
    impl = PCAImplementation(OperationParameters(n_components='auto'))
    impl.fit(train_td)
    expected = default_components_budget(impl.n_samples_, impl.n_features_)
    assert impl.n_components_ == expected
    assert impl.params.get('n_components') == expected


@pytest.mark.unit
@pytest.mark.parametrize('method', ['elbow', 'broken_stick'])
def test_pca_spectrum_n_components_methods(train_td, method):
    impl = PCAImplementation(OperationParameters(n_components=method))
    impl.fit(train_td)
    out = impl.transform(train_td)
    assert 1 <= impl.n_components_ <= train_td.features.shape[1]
    assert out.features.shape == (train_td.features.shape[0], impl.n_components_)
    assert isinstance(impl.params.get('n_components'), int)


@pytest.mark.unit
def test_pca_fit_drops_nan_rows_transform_keeps_them(train_td_with_nan):
    impl = PCAImplementation(OperationParameters(n_components=2))
    impl.fit(train_td_with_nan)

    assert impl.n_samples_ == train_td_with_nan.features.shape[0] - 2

    out = impl.transform(train_td_with_nan)
    assert out.features.shape == (train_td_with_nan.features.shape[0], 2)
    assert torch.isnan(out.features[0]).all()
    assert torch.isnan(out.features[3]).all()
    assert not torch.isnan(out.features[1]).any()


@pytest.mark.unit
def test_pca_fit_rejects_too_few_finite_samples(train_td):
    features = train_td.features[:3].clone()
    features[0, 0] = float('nan')
    features[1, 1] = float('nan')
    data = replace(train_td, features=features, fingerprint=None)
    impl = PCAImplementation(OperationParameters(n_components=1))
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


# --- TruncatedSVD ---


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
def test_truncated_svd_rejects_mle_accepts_feature_fraction(train_td):
    with pytest.raises(FedotValidationError):
        validate_truncated_svd_params({'n_components': 'mle'})
    with pytest.raises(FedotValidationError):
        TruncatedSVDImplementation(OperationParameters(n_components='mle'))

    impl = TruncatedSVDImplementation(OperationParameters(n_components=0.5))
    impl.fit(train_td)
    expected = max(1, min(impl.n_samples_, impl.n_features_, round(0.5 * impl.n_features_)))
    assert impl.n_components_ == expected


@pytest.mark.unit
def test_truncated_svd_default_n_components_is_auto():
    impl = TruncatedSVDImplementation(OperationParameters())
    assert impl.params.get('n_components') == 'auto'


@pytest.mark.unit
def test_truncated_svd_auto_n_components_uses_half_features_budget(train_td):
    impl = TruncatedSVDImplementation(OperationParameters(n_components='auto'))
    impl.fit(train_td)
    expected = default_components_budget(impl.n_samples_, impl.n_features_)
    assert impl.n_components_ == expected
    assert impl.params.get('n_components') == expected
    out = impl.transform(train_td)
    assert out.features.shape == (train_td.features.shape[0], expected)


@pytest.mark.unit
@pytest.mark.parametrize('method', ['elbow', 'broken_stick'])
def test_truncated_svd_spectrum_n_components_methods(train_td, method):
    impl = TruncatedSVDImplementation(OperationParameters(n_components=method))
    impl.fit(train_td)
    out = impl.transform(train_td)
    assert 1 <= impl.n_components_ <= train_td.features.shape[1]
    assert out.features.shape == (train_td.features.shape[0], impl.n_components_)
    assert isinstance(impl.params.get('n_components'), int)


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
