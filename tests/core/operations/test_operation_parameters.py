import pytest

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

@pytest.mark.parametrize('operation', ['catboost', 'catboostreg', 'catboost/1'])
def test_catboost_legacy_alias_overrides_default_and_survives_repeated_merging(operation):
    params = OperationParameters.from_operation_type(operation, od_wait=17, od_type='IncToDec', od_pval=0.1)
    assert params.get('od_wait') == 17
    assert 'early_stopping_rounds' not in params.keys()
    assert params.get('od_type') == 'IncToDec'
    repeated = OperationParameters.from_operation_type(operation, **params.to_dict())
    assert repeated.to_dict() == params.to_dict()


@pytest.mark.parametrize('operation', ['catboost', 'catboostreg'])
def test_catboost_defaults_keep_patience_and_best_model_selection(operation):
    params = OperationParameters.from_operation_type(operation)
    assert params.get('early_stopping_rounds') == 100
    assert 'od_wait' not in params.keys()
    assert 'od_type' not in params.keys()
    assert params.get('use_eval_set') is True
    assert params.get('use_best_model') is True


@pytest.mark.parametrize('operation', ['xgboost', 'xgboostreg'])
def test_xgboost_default_matches_previous_effective_categorical_setting(operation):
    assert OperationParameters.from_operation_type(operation).get('enable_categorical') is False
    assert OperationParameters.from_operation_type(operation, enable_categorical=True).get('enable_categorical') is True
