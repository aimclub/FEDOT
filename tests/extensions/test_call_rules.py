import inspect

import pytest

from fedot.extensions.call_rules import CallShape, invoke_factory, invoke_once, plan_call
from fedot.extensions.contracts import ExtensionContractError
from fedot.extensions.registry import smoke_test_extension
from tests.extensions.test_registration_scope import manifest


@pytest.mark.parametrize('factory', [lambda: 'ok', lambda params: params['value'],
                                     lambda *, params: params['value']])
def test_factory_selects_compatible_signature(factory):
    assert invoke_factory(factory, {'value': 'ok'}) == 'ok'


def test_positional_only_factory_is_supported():
    def factory(params, /):
        return params['value']
    assert invoke_factory(factory, {'value': 7}) == 7


def test_bind_rejects_partial_match_before_execution():
    calls = []

    def method(first, second):
        calls.append((first, second))
        return second

    result = invoke_once(method, (((1,), {}), ((1, 2), {})))
    assert result == 2
    assert calls == [(1, 2)]
    plan = plan_call(inspect.signature(method), (CallShape(1), CallShape(2)))
    assert plan.value.candidate_index == 1


def test_method_internal_type_error_is_retained_without_retry():
    calls = []
    cause = TypeError('inside user method')

    def method(*args):
        calls.append(args)
        raise cause

    with pytest.raises(ExtensionContractError) as error:
        invoke_once(method, (((1,), {}), ((1, 2), {}), ((), {})))
    assert error.value.code == 'extension_execution_failed'
    assert error.value.__cause__ is cause
    assert error.value.error.cause is cause
    assert calls == [(1,)]


def test_factory_internal_type_error_does_not_trigger_zero_argument_retry():
    calls = []
    cause = TypeError('inside factory')

    def factory(params=None):
        calls.append(params)
        raise cause

    result = smoke_test_extension(manifest(factory=factory))
    assert result.monoid[0].code == 'factory_smoke_test_failed'
    assert result.monoid[0].cause is cause
    assert calls == [{}]


def test_unsupported_signature_does_not_execute_callable():
    calls = []

    def factory(first, second):
        calls.append('called')

    with pytest.raises(ExtensionContractError) as error:
        invoke_factory(factory, {})
    assert error.value.code == 'invalid_factory_signature'
    assert calls == []
