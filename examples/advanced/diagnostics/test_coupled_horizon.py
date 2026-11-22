import importlib.util
import os
from pathlib import Path
import sys

import pytest

from examples.advanced.diagnostics.coupled_horizon import PROFILES, apply_profile
from fedot.core.operations.operation_parameters import OperationParameters


@pytest.mark.parametrize('index,expected', enumerate(PROFILES))
def test_profiles_are_atomic_and_preserve_other_parameters(index, expected):
    params = OperationParameters(horizon_profile=index, num_leaves=31, random_state=104735,
                                 learning_rate=.7, n_estimators=11)
    apply_profile(params)
    assert (params.get('n_estimators'), params.get('learning_rate')) == expected
    assert params.get('num_leaves') == 31 and params.get('random_state') == 104735


@pytest.mark.parametrize('index', [-1, 4, True, 1.5, '1'])
def test_invalid_profiles_are_rejected(index):
    with pytest.raises(ValueError):
        apply_profile(OperationParameters(horizon_profile=index))


def test_no_profile_leaves_explicit_and_regression_parameters_unchanged():
    params = OperationParameters(n_estimators=23, learning_rate=.07)
    before = params.to_dict()
    apply_profile(params)
    assert params.to_dict() == before


def test_isolated_real_constructors_and_search_space():
    root = os.environ.get('FEDOT_HORIZON_CANDIDATE')
    if not root:
        pytest.skip('Requires an explicitly isolated horizon source')
    root = Path(root)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    model = load('_horizon_model', 'fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py')
    search = load('_horizon_search', 'fedot/core/pipelines/tuning/search_space.py')
    space = search.PipelineSearchSpace().parameters_per_operation
    assert 'learning_rate' not in space['lgbm']
    assert space['lgbm']['horizon_profile']['sampling-scope'] == [[0, 1, 2, 3]]
    assert 'horizon_profile' not in space['lgbmreg']
    for index, pair in enumerate(PROFILES):
        implementation = model.FedotLightGBMClassificationImplementation(
            OperationParameters.from_operation_type('lgbm', horizon_profile=index))
        assert (implementation.model.n_estimators, implementation.model.learning_rate) == pair
        assert 'horizon_profile' not in implementation.model_params
    regression = model.FedotLightGBMRegressionImplementation(
        OperationParameters.from_operation_type('lgbmreg', n_estimators=23, learning_rate=.07))
    assert regression.model.n_estimators == 23 and regression.model.learning_rate == .07
