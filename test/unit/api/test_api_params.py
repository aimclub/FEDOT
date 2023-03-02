from typing import Optional

import pytest
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum

from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum

default_int_value = 2
fedot_params = dict(parallelization_mode='populational',
                    show_progress=True,
                    max_depth=default_int_value,
                    max_arity=default_int_value,
                    pop_size=default_int_value,
                    num_of_generations=default_int_value,
                    keep_n_best=default_int_value,
                    available_operations=['lagged', 'ridge'],
                    metric=RegressionMetricsEnum.SMAPE,
                    validation_blocks=default_int_value,
                    cv_folds=default_int_value,
                    genetic_scheme=GeneticSchemeTypesEnum.steady_state,
                    early_stopping_iterations=default_int_value,
                    early_stopping_timeout=default_int_value,
                    optimizer=EvoGraphOptimizer,
                    optimizer_external_params=dict(),
                    collect_intermediate_metric=False,
                    max_pipeline_fit_time=default_int_value,
                    initial_assumption=PipelineBuilder().add_node('lagged').add_node('ridge').build(),
                    preset=AUTO_PRESET_NAME,
                    use_pipelines_cache=True,
                    use_preprocessing_cache=True,
                    use_input_preprocessing=True,
                    cache_folder='cache',
                    keep_history=True,
                    history_dir='history',
                    with_tuning=False)

params_with_missings = dict(parallelization_mode='populational',
                            num_of_generations=default_int_value,
                            available_operations=['lagged', 'ridge'],
                            metric=RegressionMetricsEnum.SMAPE,
                            validation_blocks=default_int_value,
                            cv_folds=default_int_value,
                            initial_assumption=PipelineBuilder().add_node('lagged').add_node('ridge').build(),
                            preset=AUTO_PRESET_NAME,
                            use_pipelines_cache=True,
                            use_preprocessing_cache=True,
                            history_dir='history',
                            with_tuning=False)

correct_composer_keys = ['max_arity', 'max_depth', 'num_of_generations',
                         'early_stopping_iterations', 'early_stopping_timeout',
                         'max_graph_fit_time', 'parallelization_mode',
                         'static_individual_metadata', 'show_progress',
                         'collect_intermediate_metric', 'keep_n_best',
                         'keep_history', 'history_dir', 'cv_folds', 'validation_blocks']

correct_gp_algorithm_keys = ['mutation_types', 'genetic_scheme_type', 'pop_size']


def get_api_params_repository(task_type: Optional[TaskTypesEnum] = None):
    task_type = task_type or TaskTypesEnum.ts_forecasting
    params_repository = ApiParamsRepository(task_type)
    return params_repository


@pytest.mark.parametrize('input_params', [fedot_params, params_with_missings])
def test_correctly_sets_default_params(input_params):
    params_repository = get_api_params_repository()
    output_params = params_repository.check_and_set_default_params(input_params)
    default_params = params_repository.default_params_for_task(params_repository.task_type)
    for k in input_params.keys():
        assert k in default_params
        if input_params[k] is None:
            assert output_params[k] == default_params[k]
        else:
            assert output_params[k] == input_params[k]


@pytest.mark.parametrize('input_params, case, correct_keys',
                         [(fedot_params, 'composer', correct_composer_keys),
                          (params_with_missings, 'composer', correct_composer_keys),
                          (fedot_params, 'gp_algo', correct_gp_algorithm_keys),
                          (params_with_missings, 'gp_algo', correct_gp_algorithm_keys)])
def test_filter_params_correctly(input_params, case, correct_keys):
    params_repository = get_api_params_repository()
    input_params = params_repository.check_and_set_default_params(input_params)
    if case == 'composer':
        output_params = params_repository.get_params_for_composer_requirements(input_params)
    elif case == 'gp_algo':
        output_params = params_repository.get_params_for_gp_algorithm_params(input_params)
    assert all(key in correct_keys for key in output_params.keys())
    assert not any(key in correct_keys and key not in output_params.keys() for key in input_params)
