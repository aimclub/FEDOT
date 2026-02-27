import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.api.utils.checkers_collections import ApiConfigCheck


def get_corrupted_input_data(fill_value):
    array = np.random.rand(10, 10)
    for i in range(10):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        array[x, y] = fill_value
    target = np.random.rand(10)
    return array, target


@pytest.mark.parametrize("input_data", (
    get_corrupted_input_data(np.nan),
    get_corrupted_input_data(np.inf)
), ids=['nan_filled', 'inf_filled'])
def test_data_check(input_data):
    features, target = input_data
    data_check = DataCheck(input_data=(features, target),
                           task='classification')
    clean_data = data_check.check_input_data()
    assert clean_data is not None


def test_ApiConfigCheck():
    DEFAULT_SUBCONFIG = {
        'use_automl': True,
        'optimisation_strategy': {
            'optimisation_strategy': {
                'mutation_agent': 'random',
                'mutation_strategy': 'growth_mutation_strategy'},
            'optimisation_agent': 'Industrial'}}

    TASK_MAPPING = {
        'classification': {
            'task': 'classification',
            'use_automl': True,
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}},
        'regression': {
            'task': 'regression',
                    'use_automl': True,
                    'optimisation_strategy': {
                        'optimisation_strategy': {
                            'mutation_agent': 'random',
                            'mutation_strategy': 'growth_mutation_strategy'},
                        'optimisation_agent': 'Industrial'}},
        'ts_forecasting': {
            'task': 'ts_forecasting',
            'use_automl': True,
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}}}

    DEFAULT_AUTOML_CONFIG = {'task': None, **DEFAULT_SUBCONFIG}

    DEFAULT_COMPUTE_CONFIG = {'backend': 'cpu',
                              'distributed': dict(processes=False,
                                                  n_workers=2,
                                                  threads_per_worker=2,
                                                  memory_limit=0.3
                                                  ),
                              'output_folder': './results',
                              'use_cache': None,
                              'automl_folder': {'optimisation_history': './results/opt_hist',
                                                'composition_results': './results/comp_res'}}

    AUTOML_LEARNING_STRATEGY = {'timeout': 0.1,
                                'logging_level': 50}

    LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                       'optimisation_loss': {'quality_loss': 'f1'}}
    INDUSTRIAL_CONFIG = {'problem': 'classification'}

    API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
                  'automl_config': DEFAULT_AUTOML_CONFIG,
                  'learning_config': LEARNING_CONFIG,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    exp_config = {'task': 'classification',
                  'logging_level': 40,
                  'timeout': 140,
                  'mutation_strategy': 'SOME_STRATEGY',
                  'quality_loss': 'CockLord'}

    check = ApiConfigCheck()
    UPD_API_CONFIG = check.update_config_with_kwargs(config_to_update=API_CONFIG,
                                                     **exp_config)

    check.compare_configs(API_CONFIG, UPD_API_CONFIG)
    model = FedotIndustrial(**UPD_API_CONFIG)

    assert model is not None
