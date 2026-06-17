import datetime

import pytest

from fedot.api.api_utils.params import ApiParams
from fedot.validation.errors import FedotValidationError


def test_api_params_raises_validation_error_for_unknown_problem():
    with pytest.raises(FedotValidationError, match='Wrong type name of the given task'):
        ApiParams({}, problem='clustering')


def test_api_params_normalizes_timeout_and_generations():
    params = ApiParams({'num_of_generations': 5},
                       problem='classification', timeout=-1)

    assert params.timeout is None
    assert params['num_of_generations'] == 5

    params_with_default_generations = ApiParams(
        {}, problem='classification', timeout=1)
    assert params_with_default_generations.timeout == 1
    assert params_with_default_generations['num_of_generations'] == 10000


def test_api_params_accept_and_apply_recommendations_updates_internal_mapping(monkeypatch):
    params = ApiParams({}, problem='classification', timeout=1)
    captured = {'called': False}

    def fake_change_preset(task, data_type):
        captured['called'] = True

    monkeypatch.setattr(
        params, 'change_preset_for_label_encoded_data', fake_change_preset)

    params.accept_and_apply_recommendations(
        input_data=type(
            'Data', (), {'task': params.task, 'data_type': None})(),
        recommendations={'cv_folds': 3, 'label_encoded': {}},
    )

    assert captured['called'] is True
    assert params['cv_folds'] == 3
    assert params['label_encoded'] == {}


def test_api_params_composer_requirements_do_not_receive_cv_folds():
    params = ApiParams({'cv_folds': 3}, problem='classification', timeout=1)

    params.init_composer_requirements(datetime.timedelta(minutes=1))

    assert params['cv_folds'] == 3
    assert params.composer_requirements.cv_folds is None
