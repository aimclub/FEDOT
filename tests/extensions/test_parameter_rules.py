from fedot.core.operations.operation_parameters import OperationParameters
from fedot.extensions.contracts import ExternalModelSpec, ModelCapabilities, ModelHyperparamsSchema
from fedot.extensions.parameter_rules import (
    apply_extension_defaults,
    extract_factory_params,
    find_missing_required_params,
    normalize_extension_user_params,
    resolve_extension_params,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum



def _make_model_spec():
    return ExternalModelSpec(
        name='external_with_schema',
        factory=lambda params=None: object(),
        capabilities=ModelCapabilities(
            tasks=(TaskTypesEnum.regression,),
            data_types=(DataTypesEnum.table,),
            tags=('external',),
        ),
        hyperparams_schema=ModelHyperparamsSchema(
            required=('alpha',),
            optional=('beta',),
            defaults={'beta': 0.5},
        ),
    )



def test_extension_parameter_rules_apply_defaults_and_filter_runtime_keys():
    normalized = normalize_extension_user_params({'alpha': 1.0})
    with_defaults = apply_extension_defaults({'beta': 0.5}, normalized)
    factory_params = extract_factory_params({
        **with_defaults,
        '_extension_output_mode': 'labels',
        'model_fit': object(),
        'model_predict': object(),
    })

    assert with_defaults == {'beta': 0.5, 'alpha': 1.0}
    assert factory_params == {'beta': 0.5, 'alpha': 1.0}



def test_extension_parameter_rules_detect_missing_required_params():
    missing = find_missing_required_params(('alpha', 'gamma'), {'alpha': 1.0})
    resolution = resolve_extension_params(_make_model_spec(), {'beta': 1.5})

    assert missing == ('gamma',)
    assert resolution.is_left()
    assert resolution.monoid[0].details['required'] == ['alpha']



def test_extension_parameter_rules_return_resolved_params_when_schema_is_satisfied():
    resolution = resolve_extension_params(_make_model_spec(), {'alpha': 1.0})

    assert resolution.is_right()
    assert resolution.value == {'beta': 0.5, 'alpha': 1.0}


def test_extract_factory_params_accepts_operation_parameters():
    strategy_params = OperationParameters(
        alpha=1.0,
        beta=0.5,
        _extension_output_mode='labels',
        model_fit=object(),
        model_predict=object(),
    )

    factory_params = extract_factory_params(strategy_params)

    assert factory_params == {'alpha': 1.0, 'beta': 0.5}
