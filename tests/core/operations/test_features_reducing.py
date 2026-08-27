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
    n_components_from_mle,
    resolve_spectrum_n_components,
)
from fedot.core.operations.evaluation.tensor_transform import TensorTransformStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import RandomStateHandler
from fedot.validation.errors import FedotValidationError


@pytest.fixture
def train_td():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    return TensorDataCreator.create(features, backend_name='cpu')


@pytest.fixture
def wide_td():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 40)).astype(np.float32)
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


@pytest.mark.unit
def test_params_accept_numpy_scalars(train_td):
    validated_pca = validate_pca_params({'n_components': np.int64(3)})
    assert validated_pca['n_components'] == 3
    assert type(validated_pca['n_components']) is int

    validated_svd = validate_truncated_svd_params({
        'n_components': np.float64(0.5),
        'n_iter': np.int64(4),
        'n_oversamples': np.int64(8),
    })
    assert validated_svd['n_components'] == 0.5
    assert type(validated_svd['n_components']) is float
    assert validated_svd['n_iter'] == 4
    assert validated_svd['n_oversamples'] == 8

    impl = TruncatedSVDImplementation(OperationParameters(
        n_components=np.int64(2),
        n_iter=np.int64(3),
        n_oversamples=np.int64(5),
    ))
    impl.fit(train_td)
    assert impl.n_components_ == 2


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
    assert impl.params.get('n_components') == 0.9
    assert isinstance(impl.n_components_, int)


@pytest.mark.unit
def test_pca_float_one_keeps_full_rank(train_td):
    impl = PCAImplementation(OperationParameters(n_components=1.0))
    impl.fit(train_td)
    expected = min(impl.n_samples_, impl.n_features_)
    assert impl.n_components_ == expected
    assert impl.transform(train_td).features.shape[1] == expected

    one = PCAImplementation(OperationParameters(n_components=1))
    one.fit(train_td)
    assert one.n_components_ == 1


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
def test_pca_params_schema_rejects_unknown_keys():
    with pytest.raises(FedotValidationError, match='n_component|Unknown keys'):
        validate_pca_params({'n_components': 2, 'n_component': 3})
    with pytest.raises(FedotValidationError, match='svd_solver|Unknown keys'):
        validate_pca_params({'n_components': 2, 'svd_solver': 'full'})
    with pytest.raises(FedotValidationError, match='svd_solver|Unknown keys'):
        PCAImplementation(OperationParameters(n_components=2, svd_solver='full'))


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
    assert impl.params.get('n_components') == 'auto'


@pytest.mark.unit
def test_pca_auto_policy_survives_refit(train_td):
    impl = PCAImplementation(OperationParameters(n_components='auto'))
    impl.fit(train_td)
    first = impl.n_components_
    impl.fit(train_td)
    assert impl.params.get('n_components') == 'auto'
    assert impl.n_components_ == first


@pytest.mark.unit
@pytest.mark.parametrize('method', ['elbow', 'broken_stick'])
def test_pca_spectrum_n_components_methods(train_td, method):
    impl = PCAImplementation(OperationParameters(n_components=method))
    impl.fit(train_td)
    out = impl.transform(train_td)
    assert 1 <= impl.n_components_ <= train_td.features.shape[1]
    assert out.features.shape == (train_td.features.shape[0], impl.n_components_)
    assert impl.params.get('n_components') == method


@pytest.mark.unit
def test_pca_mle_matches_sklearn(train_td):
    from sklearn.decomposition import PCA as SklearnPCA

    impl = PCAImplementation(OperationParameters(n_components='mle'))
    impl.fit(train_td)
    sklearn_pca = SklearnPCA(n_components='mle', svd_solver='full')
    sklearn_pca.fit(train_td.features.numpy())
    assert impl.n_components_ == sklearn_pca.n_components_


@pytest.mark.unit
def test_pca_mle_rejects_wide_tables():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(5, 10)).astype(np.float32)
    data = TensorDataCreator.create(features, backend_name='cpu')
    impl = PCAImplementation(OperationParameters(n_components='mle'))
    with pytest.raises(FedotValidationError, match='n_samples >= n_features'):
        impl.fit(data)


@pytest.mark.unit
def test_n_components_from_mle_matches_sklearn_infer_dimension():
    from sklearn.decomposition._pca import _infer_dimension

    rng = np.random.default_rng(1)
    spectrum = np.sort(rng.random(8) + 0.1)[::-1].astype(np.float64)
    n_samples = 40
    expected = int(_infer_dimension(spectrum, n_samples))
    assert n_components_from_mle(torch.tensor(spectrum), n_samples) == expected


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
    assert impl.params.get('n_components') == 0.5


@pytest.mark.unit
def test_truncated_svd_params_schema_rejects_unknown_keys():
    with pytest.raises(FedotValidationError, match='n_component|Unknown keys'):
        validate_truncated_svd_params({'n_components': 2, 'n_component': 3})
    with pytest.raises(FedotValidationError, match='Unknown keys'):
        TruncatedSVDImplementation(OperationParameters(n_components=2, svd_solver='full'))


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['pca', 'truncated_svd'])
def test_decomposition_search_space_includes_validated_n_components_modes(operation):
    choices = PipelineSearchSpace().get_parameters_dict()[operation]['n_components']['sampling-scope'][0]
    for mode in ('auto', 'elbow', 'broken_stick'):
        assert mode in choices


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
    assert impl.params.get('n_components') == 'auto'
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
    assert impl.params.get('n_components') == method


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


def _relative_reconstruction_error(features: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    reconstructed = (features @ components.T) @ components
    return torch.linalg.norm(features - reconstructed) / torch.linalg.norm(features)


@pytest.mark.unit
def test_truncated_svd_integer_k_matches_exact_svd_when_q_covers_full_rank(train_td):
    impl = TruncatedSVDImplementation(OperationParameters(n_components=3))
    impl.fit(train_td)
    _, _, vh = torch.linalg.svd(train_td.features, full_matrices=False)
    torch.testing.assert_close(impl.components_, vh[:3])


@pytest.mark.unit
def test_truncated_svd_integer_k_matches_spectrum_subspace_when_full_rank(train_td):
    spectrum = TruncatedSVDImplementation(OperationParameters(n_components='elbow'))
    spectrum.fit(train_td)
    integer = TruncatedSVDImplementation(
        OperationParameters(n_components=spectrum.n_components_),
    )
    integer.fit(train_td)
    torch.testing.assert_close(spectrum.components_, integer.components_)


@pytest.mark.unit
def test_truncated_svd_lowrank_reconstruction_tracks_exact_svd(wide_td):
    k = 3
    impl = TruncatedSVDImplementation(OperationParameters(
        n_components=k, n_oversamples=2, n_iter=5,
    ))
    impl.random_state = 0
    impl.fit(wide_td)

    _, _, vh = torch.linalg.svd(wide_td.features, full_matrices=False)
    exact_err = _relative_reconstruction_error(wide_td.features, vh[:k])
    approx_err = _relative_reconstruction_error(wide_td.features, impl.components_)
    assert approx_err <= exact_err + 5e-3


@pytest.mark.unit
def test_truncated_svd_lowrank_is_reproducible_with_random_state(wide_td):
    params = OperationParameters(n_components=3, n_oversamples=2)
    first = TruncatedSVDImplementation(params)
    second = TruncatedSVDImplementation(params)
    first.random_state = 0
    second.random_state = 0
    first.fit(wide_td)
    second.fit(wide_td)
    torch.testing.assert_close(first.components_, second.components_)


@pytest.mark.unit
def test_truncated_svd_strategy_fit_is_reproducible(train_td):
    original_seed = RandomStateHandler.MODEL_FITTING_SEED
    RandomStateHandler.MODEL_FITTING_SEED = 7
    try:
        strategy = TensorTransformStrategy(
            'truncated_svd', OperationParameters(n_components=3),
        )
        first = strategy.fit(train_td)
        second = strategy.fit(train_td)
        torch.testing.assert_close(first.components_, second.components_)
    finally:
        RandomStateHandler.MODEL_FITTING_SEED = original_seed


@pytest.mark.unit
@pytest.mark.parametrize(
    'impl_cls, params',
    [
        (PCAImplementation, OperationParameters(n_components=2)),
        (TruncatedSVDImplementation, OperationParameters(n_components=2)),
    ],
)
def test_transform_rejects_feature_width_mismatch(train_td, impl_cls, params):
    impl = impl_cls(params)
    impl.fit(train_td)
    fitted_width = impl.n_features_
    assert fitted_width is not None and fitted_width > 1
    wrong = replace(
        train_td,
        features=train_td.features[:, : fitted_width - 1],
        fingerprint=None,
    )
    with pytest.raises(
        FedotValidationError,
        match=f'expected {fitted_width} features, got {fitted_width - 1}',
    ):
        impl.transform(wrong)
