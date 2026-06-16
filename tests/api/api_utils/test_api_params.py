import datetime

from fedot.api.api_utils.params import ApiParams
from fedot.core.data.common.enums import StateEnum
from fedot.core.repository.tasks import TaskTypesEnum


def test_api_params_raises_value_error_for_unknown_problem():
    try:
        ApiParams({}, problem='clustering')
    except ValueError as error:
        assert 'Wrong type name of the given task' in str(error)
    else:
        raise AssertionError('ApiParams should reject unknown problem values')


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


def test_api_params_builds_tensor_data_config_field_on_init():
    params = ApiParams({}, problem='classification', timeout=1)

    assert params.tensor_data_config == {
        'backend_name': 'cpu',
        'use_cache': True,
    }
    assert params['tensor_data_config'] == params.tensor_data_config


def test_api_params_tensor_data_config_uses_user_values_from_fedot_kwargs():
    params = ApiParams(
        {
            'tensor_data_config': {
                'backend_name': ' GPU ',
                'use_cache': False,
                'encoding_strategy': {'kind': 'label'},
            },
            'use_preprocessing_cache': True,
        },
        problem='classification',
        timeout=1,
    )

    assert params.tensor_data_config == {
        'backend_name': 'gpu',
        'use_cache': False,
        'encoding_strategy': {'kind': 'label'},
    }


def test_prepare_tensordata_creation_uses_defaults_when_config_is_missing():
    params = ApiParams({}, problem='classification', timeout=1)

    request = params.prepare_tensordata_creation(target=[0, 1])

    assert request.backend_name == 'cpu'
    assert request.spec_kwargs['task'] is params.task
    assert request.spec_kwargs['state'] is StateEnum.FIT
    assert request.spec_kwargs['target'] == [0, 1]
    assert request.spec_kwargs['use_cache'] is True


def test_prepare_tensordata_creation_uses_tensor_data_config_and_predict_state():
    params = ApiParams(
        {
            'tensor_data_config': {
                'backend_name': 'gpu',
                'use_cache': False,
                'encoding_strategy': {'kind': 'label'},
                'optional_strategy': {'scaling': None},
            },
            'use_preprocessing_cache': True,
        },
        problem='classification',
        timeout=1,
    )

    request = params.prepare_tensordata_creation(is_predict=True)

    assert request.backend_name == 'gpu'
    assert request.spec_kwargs['state'] is StateEnum.PREDICT
    assert request.spec_kwargs['use_cache'] is False
    assert request.spec_kwargs['encoding_strategy'] == {'kind': 'label'}
    assert request.spec_kwargs['task'].task_type is TaskTypesEnum.classification
    assert 'target' not in request.spec_kwargs
    assert 'optional_strategy' not in request.spec_kwargs


def test_api_params_composer_requirements_do_not_receive_cv_folds():
    params = ApiParams({'cv_folds': 3}, problem='classification', timeout=1)

    params.init_composer_requirements(datetime.timedelta(minutes=1))

    assert params['cv_folds'] == 3
    assert params.composer_requirements.cv_folds is None
