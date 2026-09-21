"""Finite-domain laws without additional development dependencies."""
from dataclasses import replace
from itertools import permutations

import pytest

from fedot.extensions import ModelHyperparamsSchema, extension_scope
from fedot.extensions.parameter_rules import resolve_extension_params
from fedot.extensions.registration_rules import plan_registration
from fedot.extensions.runtime_rules import get_extension_operation_spec
from tests.extensions.test_registration_scope import manifest


@pytest.mark.parametrize('ids', [order for size in range(1, 5) for order in permutations(range(size))])
def test_registration_permutation_preserves_all_factory_bindings(ids):
    """Independent manifests resolve to the same factories in every tested order."""
    manifests = tuple(manifest(f'property_ext_{index}', f'property_model_{index}') for index in ids)
    with extension_scope(*manifests):
        for item in manifests:
            resolved = get_extension_operation_spec(item.models[0].name)
            assert resolved is item.models[0]
    for item in manifests:
        assert get_extension_operation_spec(item.models[0].name) is None


@pytest.mark.parametrize('ids', [tuple(range(size)) for size in range(9)])
def test_additions_never_erase_a_duplicate_conflict(ids):
    """Adding independent operations cannot make a conflicting batch valid."""
    additions = tuple(manifest(f'ext_{index}', f'op_{index}') for index in ids)
    result = plan_registration((manifest(), manifest('duplicate')) + additions)
    assert result.monoid[0].code == 'operation_name_conflict'


@pytest.mark.parametrize('default', [None, 0, 42, 'value'])
@pytest.mark.parametrize('override', [None, 0, 42, 'other'])
def test_parameter_resolution_is_idempotent_and_user_values_win(default, override):
    """Applying defaults twice preserves explicit values, including None."""
    spec = replace(manifest().models[0], hyperparams_schema=ModelHyperparamsSchema(
        required=('required_value',), defaults={'required_value': default, 'other': 7}))
    params = {'required_value': override}
    resolved = resolve_extension_params(spec, params).value
    assert resolved == {'required_value': override, 'other': 7}
    assert resolve_extension_params(spec, resolved).value == resolved
    assert params == {'required_value': override}
