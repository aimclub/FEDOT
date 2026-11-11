import pytest

from fedot.core.operations.operation_parameter_rules import (
    collect_changed_keys,
    merge_operation_default_params,
    resolve_setdefault_value,
)


def test_operation_parameter_rules_merge_defaults_and_track_changes():
    merged = merge_operation_default_params({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    changed_keys = collect_changed_keys(
        {'a': 1, 'b': 2}, {'a': 1, 'b': 3, 'd': 4}, ())

    assert merged == {'a': 1, 'b': 3, 'c': 4}
    assert changed_keys == ('b', 'd')


@pytest.mark.parametrize('name', ['od_wait', 'early_stopping_rounds'])
@pytest.mark.parametrize('value', [None, 17])
def test_explicit_alias_replaces_default_without_mutating_inputs(name, value):
    defaults = {'early_stopping_rounds': 100, 'depth': 4}
    passed = {name: value}
    result = merge_operation_default_params(defaults, passed, (('early_stopping_rounds', 'od_wait'),))
    assert result == {'depth': 4, name: value}
    assert defaults == {'early_stopping_rounds': 100, 'depth': 4}
    assert passed == {name: value}


def test_explicit_aliases_fail_before_library_initialization():
    with pytest.raises(ValueError, match='early_stopping_rounds, od_wait'):
        merge_operation_default_params({}, {'early_stopping_rounds': 10, 'od_wait': 20},
                                       (('early_stopping_rounds', 'od_wait'),))


def test_operation_parameter_rules_resolve_setdefault_value_explicitly():
    existing_value, should_update_existing = resolve_setdefault_value({
                                                                      'a': 1}, 'a', 2)
    missing_value, should_update_missing = resolve_setdefault_value({
                                                                    'a': 1}, 'b', 3)

    assert existing_value == 1
    assert should_update_existing is False
    assert missing_value == 3
    assert should_update_missing is True
