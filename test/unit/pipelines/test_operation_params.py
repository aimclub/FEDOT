from fedot.core.operations.hyperparameters_preprocessing import HyperparametersPreprocessor
from fedot.core.operations.operation_parameters import OperationParameters


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


def test_preprocessing_rules_defined_for_both_xgboost_operations():
    """ Both halves of a model pair are expected to carry the same rules,
    as is the case for lgbm/lgbmreg and catboost/catboostreg """
    rules = HyperparametersPreprocessor.all_preprocessing_rules

    assert rules.get('xgboostreg')
    assert rules['xgboostreg'] == rules['xgboost']


def test_integer_params_are_rounded_for_xgboostreg():
    preprocessor = HyperparametersPreprocessor(operation_type='xgboostreg', n_samples_data=100)

    corrected = preprocessor.correct({'max_depth': 3.7, 'n_estimators': 80.2})

    assert corrected == {'max_depth': 4, 'n_estimators': 80}
