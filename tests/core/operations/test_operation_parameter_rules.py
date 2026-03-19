from fedot.core.operations.operation_parameter_rules import (
    collect_changed_keys,
    merge_operation_default_params,
    resolve_setdefault_value,
)


def test_operation_parameter_rules_merge_defaults_and_track_changes():
    merged = merge_operation_default_params({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    changed_keys = collect_changed_keys({'a': 1, 'b': 2}, {'a': 1, 'b': 3, 'd': 4}, ())

    assert merged == {'a': 1, 'b': 3, 'c': 4}
    assert changed_keys == ('b', 'd')


def test_operation_parameter_rules_resolve_setdefault_value_explicitly():
    existing_value, should_update_existing = resolve_setdefault_value({'a': 1}, 'a', 2)
    missing_value, should_update_missing = resolve_setdefault_value({'a': 1}, 'b', 3)

    assert existing_value == 1
    assert should_update_existing is False
    assert missing_value == 3
    assert should_update_missing is True
