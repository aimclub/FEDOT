from fedot.core.operations.operation_parameters import OperationParameters, get_default_params


def test_params_keeper_update():
    params = {'a': 1, 'b': 2, 'c': 3}
    keeper = OperationParameters(**params)
    new_params = {'a': 1, 'b': 3, 'd': 4}
    keeper.update(**new_params)
    expected_params = {'a': 1, 'b': 3, 'c': 3, 'd': 4}
    actual_params = keeper.to_dict()
    changed_params = keeper.changed_parameters.keys()
    assert actual_params == expected_params
    assert 'a' not in changed_params
    assert 'b' in changed_params
    assert 'd' in changed_params


def test_params_keeper_get():
    params = {'a': 1, 'b': 2, 'c': 3}
    keeper = OperationParameters(**params)
    a = keeper.get('a')
    b = keeper.get('b', -1)
    d = keeper.get('d', 5)
    assert a == 1
    assert b == 2
    assert d == 5


def test_params_keeper_setdefault_and_defaults_from_repository():
    keeper = OperationParameters(alpha=1.0)
    existing_value = keeper.setdefault('alpha', 2.0)
    missing_value = keeper.setdefault('beta', 3.0)
    default_params = get_default_params('ridge')
    merged_keeper = OperationParameters.from_operation_type(
        'ridge', alpha=0.75)

    assert existing_value == 1.0
    assert missing_value == 3.0
    assert keeper.get('beta') == 3.0
    assert merged_keeper.get('alpha') == 0.75
    assert set(default_params).issubset(set(merged_keeper.to_dict()))
