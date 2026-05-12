from fedot.api.api_utils.params import ApiParams


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
