from fedot.core.pipelines.tuning.hyperparams import ParametersChanger


def test_mutation_lagged_param_change_correct():
    """ Checks how hyperparameters are modified. For the lagged operation an
    incremental modification must be applied, not a random
    """
    incr_operation_name = 'lagged'
    incr_current_params = {'window_size': 100}
    est_sigma = incr_current_params.get('window_size') * 0.3

    min_border = incr_current_params.get('window_size') - (5 * est_sigma)
    max_border = incr_current_params.get('window_size') + (5 * est_sigma)
    changer = ParametersChanger(incr_operation_name, incr_current_params)
    for _ in range(0, 10):
        new_params = changer.get_new_operation_params()
        new_value = new_params['window_size']
        assert min_border < new_value < max_border


def test_parameters_changer_return_correct():
    min_alpha, max_alpha = 0.01, 10.0
    changer = ParametersChanger('ridge', 'alpha')
    for _ in range(0, 10):
        new_params = changer.get_new_operation_params()
        new_alpha_value = new_params['alpha']

        assert min_alpha <= new_alpha_value <= max_alpha


def test_parameters_changer_unknown_operation_correct():
    changer = ParametersChanger('nonexistent_operation', 'nonexistent_param')
    empty_output = changer.get_new_operation_params()
    assert empty_output is None
