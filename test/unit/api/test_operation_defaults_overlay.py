"""The experimental horizon profile must remain opt-in and worker-visible."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fedot.core.operations.operation_parameters import OperationParameters, get_default_params
from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository


@pytest.fixture
def profile(monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'examples/advanced/lgbm_400_defaults.json'
    monkeypatch.setenv('FEDOT_OPERATION_DEFAULTS_OVERLAY', str(path))
    return path


def test_overlay_off_by_default(monkeypatch):
    monkeypatch.delenv('FEDOT_OPERATION_DEFAULTS_OVERLAY', raising=False)
    assert 'n_estimators' not in get_default_params('lgbm')


def test_profile_only_changes_classifier_horizon(profile, monkeypatch):
    enabled = {name: get_default_params(name) for name in ('lgbm', 'lgbmreg', 'catboost', 'logit')}
    monkeypatch.delenv('FEDOT_OPERATION_DEFAULTS_OVERLAY')
    for name, values in enabled.items():
        expected = get_default_params(name)
        if name == 'lgbm':
            expected['n_estimators'] = 400
        assert values == expected


def test_explicit_node_parameters_win(profile):
    params = OperationParameters.from_operation_type('lgbm', n_estimators=17)
    assert params.get('n_estimators') == 17
    assert get_default_params('lgbm/custom')['n_estimators'] == 400


def test_profile_does_not_mutate_cache(profile):
    params = get_default_params('lgbm')
    params['n_estimators'] = 1
    assert get_default_params('lgbm')['n_estimators'] == 400


def test_worker_inherits_profile(profile):
    result = subprocess.check_output(
        [sys.executable, '-c', 'from fedot.core.operations.operation_parameters import get_default_params; '
         'print(get_default_params("lgbm")["n_estimators"])'], env=os.environ.copy(), text=True)
    assert result.strip().splitlines()[-1] == '400'


@pytest.mark.parametrize('content', [[], {'lgbm': 400}, None])
def test_invalid_overlay_fails_loudly(content, tmp_path, monkeypatch):
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(content))
    monkeypatch.setenv('FEDOT_OPERATION_DEFAULTS_OVERLAY', str(path))
    with pytest.raises(ValueError, match='must map'):
        DefaultOperationParamsRepository()


def test_relative_overlay_rejected(monkeypatch):
    monkeypatch.setenv('FEDOT_OPERATION_DEFAULTS_OVERLAY', 'relative.json')
    with pytest.raises(ValueError, match='absolute path'):
        DefaultOperationParamsRepository()


def test_missing_overlay_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('FEDOT_OPERATION_DEFAULTS_OVERLAY', str(tmp_path / 'missing.json'))
    with pytest.raises(FileNotFoundError):
        DefaultOperationParamsRepository()


def test_actual_lightgbm_constructor_and_explicit_rounds(profile):
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
        FedotLightGBMClassificationImplementation)
    model = FedotLightGBMClassificationImplementation(OperationParameters.from_operation_type('lgbm'))
    assert model.model.n_estimators == 400
    explicit = FedotLightGBMClassificationImplementation(
        OperationParameters.from_operation_type('lgbm', n_estimators=23))
    assert explicit.model.n_estimators == 23
