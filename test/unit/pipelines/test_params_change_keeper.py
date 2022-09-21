from fedot.core.operations.changing_parameters_keeper import ParametersChangeKeeper


def test_params_keeper_update():
    params = {'a': 1, 'b': 2, 'c': 3}
    keeper = ParametersChangeKeeper(parameters=params)
    keeper.update('a', 1)
    keeper.update('b', 3)
    keeper.update('d', 4)
    expected_params = {'a': 1, 'b': 3, 'c': 3, 'd': 4}
    actual_params = keeper.get_parameters()
    changed_params = keeper.changed_parameters.keys()
    assert actual_params == expected_params
    assert 'a' not in changed_params
    assert 'b' in changed_params
    assert 'd' in changed_params


def test_params_keeper_get():
    params = {'a': 1, 'b': 2, 'c': 3}
    keeper = ParametersChangeKeeper(parameters=params)
    a = keeper.get('a')
    b = keeper.get('b', -1)
    d = keeper.get('d', 5)
    assert a == 1
    assert b == 2
    assert d == 5
