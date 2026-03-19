import numpy as np

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.contracts import (
    ExtensionManifest,
    ExternalModelSpec,
    ModelCapabilities,
    ModelHyperparamsSchema,
)
from fedot.extensions.registry import clear_extension_registry, register_extension
from fedot.extensions.runtime_rules import (
    build_extension_strategy_params,
    get_extension_model_spec,
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


def test_runtime_rules_resolve_registered_extension_model_and_build_strategy_params():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        spec = get_extension_model_spec('external_runtime_model')
        params = build_extension_strategy_params('external_runtime_model', {'alpha': 1.0}, output_mode='labels')

        assert spec is not None
        assert spec.name == 'external_runtime_model'
        assert is_extension_operation_name('external_runtime_model') is True
        assert callable(params['model_fit'])
        assert callable(params['model_predict'])
        assert params['_extension_output_mode'] == 'labels'
        assert params['alpha'] == 1.0
        assert params['beta'] == 0.5
    finally:
        clear_extension_registry()


def test_runtime_rules_return_left_when_required_extension_params_are_missing():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        params = try_build_extension_strategy_params('external_runtime_model', {'beta': 1.5})

        assert params.is_left()
        assert params.monoid[0].code == 'missing_required_hyperparams'
        assert params.monoid[0].details['required'] == ['alpha']
    finally:
        clear_extension_registry()
