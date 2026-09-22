import numpy as np
import pytest

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.contracts import (
    ExtensionContractError,
    ExtensionManifest,
    ExternalModelSpec,
    ModelCapabilities,
    ModelHyperparamsSchema,
)
from fedot.extensions.registry import clear_extension_registry, register_extension
from fedot.extensions.runtime_contracts import ModelInput
from fedot.extensions.runtime_rules import (
    build_extension_strategy_params,
    fit_model,
    get_extension_data_types,
    get_extension_model_spec,
    get_extension_tensor_data_types,
    is_extension_operation_name,
    try_build_extension_strategy_params,
)


class _ExternalEstimator:
    def __init__(self, params=None):
        self.params = params or {}
        self.was_fitted = False

    def fit(self, features, target):
        self.was_fitted = True
        return self

    def predict(self, features):
        return np.zeros(features.shape[0])


class _NoneReturningEstimator(_ExternalEstimator):
    def fit(self, features, target):
        self.was_fitted = True
        return None


class _FittedEstimator:
    def predict(self, features):
        return np.ones(features.shape[0])


class _FunctionalEstimator:
    def fit(self, features, target):
        return _FittedEstimator()


class _InvalidFitEstimator:
    def fit(self, features, target):
        return 42

    def predict(self, features):
        return np.zeros(features.shape[0])


def _make_manifest():
    return ExtensionManifest(
        name='runtime_extension',
        version='1.0.0',
        models=(
            ExternalModelSpec(
                name='external_runtime_model',
                factory=lambda params=None: _ExternalEstimator(params),
                capabilities=ModelCapabilities(
                    tasks=(TaskTypesEnum.regression,),
                    data_types=(DataTypesEnum.table,),
                    tags=('external', 'linear'),
                ),
                hyperparams_schema=ModelHyperparamsSchema(
                    required=('alpha',),
                    optional=('beta',),
                    defaults={'beta': 0.5},
                ),
            ),
        ),
    )


def _model_spec(factory):
    return ExternalModelSpec(
        name='fit_contract_model',
        factory=factory,
        capabilities=ModelCapabilities(
            tasks=(TaskTypesEnum.regression,),
            data_types=(DataTypesEnum.table,),
        ),
    )


def _model_input():
    return ModelInput(
        features=np.ones((4, 2)),
        target=np.ones(4),
        idx=np.arange(4),
    )


def test_runtime_rules_resolve_registered_extension_model_and_build_strategy_params():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        spec = get_extension_model_spec('external_runtime_model')
        params = build_extension_strategy_params(
            'external_runtime_model', {'alpha': 1.0}, output_mode='labels')

        assert spec is not None
        assert spec.name == 'external_runtime_model'
        assert is_extension_operation_name('external_runtime_model') is True
        assert callable(params['model_fit'])
        assert callable(params['model_predict'])
        assert params['_extension_output_mode'] == 'labels'
        assert params['alpha'] == 1.0
        assert params['beta'] == 0.5
        assert get_extension_data_types(
            'external_runtime_model') == (DataTypesEnum.table,)
        assert get_extension_tensor_data_types(
            'external_runtime_model') == (DataTypesEnum.tabular,)
    finally:
        clear_extension_registry()


def test_runtime_rules_return_left_when_required_extension_params_are_missing():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        params = try_build_extension_strategy_params(
            'external_runtime_model', {'beta': 1.5})

        assert params.is_left()
        assert params.monoid[0].code == 'missing_required_hyperparams'
        assert params.monoid[0].details['required'] == ['alpha']
    finally:
        clear_extension_registry()


def test_legacy_callback_adapter_still_fits_and_predicts():
    from fedot.extensions import extension_scope

    with extension_scope(_make_manifest()):
        params = build_extension_strategy_params('external_runtime_model', {'alpha': 1.0})
        features = np.ones((4, 2))
        fitted = params['model_fit'](np.arange(4), features, np.ones(4), params)
        prediction, output_type = params['model_predict'](fitted, np.arange(4), features, params)
        assert fitted.was_fitted
        np.testing.assert_array_equal(prediction, np.zeros(4))
        assert output_type in ('table', 'tabular')


@pytest.mark.parametrize('estimator_type', [_ExternalEstimator, _NoneReturningEstimator])
def test_fit_model_retains_in_place_instance_for_self_or_none(estimator_type):
    instance = estimator_type()

    fitted = fit_model(_model_spec(lambda: instance), _model_input(), {})

    assert fitted is instance
    assert fitted.was_fitted is True


def test_fit_model_uses_new_functional_fit_result():
    original = _FunctionalEstimator()

    fitted = fit_model(_model_spec(lambda: original), _model_input(), {})

    assert fitted is not original
    assert isinstance(fitted, _FittedEstimator)
    np.testing.assert_array_equal(fitted.predict(np.ones((4, 2))), np.ones(4))


def test_fit_model_rejects_invalid_functional_fit_result():
    with pytest.raises(ExtensionContractError) as error:
        fit_model(_model_spec(_InvalidFitEstimator), _model_input(), {})

    assert error.value.code == 'invalid_fit_result'
    assert error.value.error.details == {
        'expected_methods': ['predict', 'predict_proba'],
        'result_type': 'int',
    }


def test_missing_operation_preserves_typed_failure():
    result = try_build_extension_strategy_params('unregistered_operation')
    assert result.monoid[0].code == 'operation_not_registered'
