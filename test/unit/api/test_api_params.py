import dataclasses
from typing import Optional

import pytest
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum

from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum


fedot_params_full = dict(parallelization_mode='populational',
                         show_progress=True,
                         max_depth=4,
                         max_arity=3,
                         pop_size=15,
                         num_of_generations=5,
                         keep_n_best=2,
                         available_operations=['lagged', 'ridge'],
                         metric=RegressionMetricsEnum.SMAPE,
                         validation_blocks=2,
                         cv_folds=None,
                         genetic_scheme=GeneticSchemeTypesEnum.steady_state,
                         early_stopping_iterations=2,
                         early_stopping_timeout=None,
                         optimizer=EvoGraphOptimizer,
                         optimizer_external_params=dict(),
                         collect_intermediate_metric=False,
                         max_pipeline_fit_time=7,
                         initial_assumption=PipelineBuilder().add_node('lagged').add_node('ridge').build(),
                         preset=AUTO_PRESET_NAME,
                         use_pipelines_cache=True,
                         use_preprocessing_cache=True,
                         use_input_preprocessing=True,
                         cache_dir='cache',
                         keep_history=True,
                         history_dir='history',
                         with_tuning=False)

params_with_missings = dict(parallelization_mode='populational',
                            num_of_generations=10,
                            available_operations=['lagged', 'ridge'],
                            metric=RegressionMetricsEnum.SMAPE,
                            validation_blocks=3,
                            cv_folds=None,
                            initial_assumption=PipelineBuilder().add_node('lagged').add_node('ridge').build(),
                            preset=AUTO_PRESET_NAME,
                            use_pipelines_cache=True,
                            use_preprocessing_cache=True,
                            history_dir='history',
                            with_tuning=False)

correct_composer_attributes = {field.name for field in dataclasses.fields(PipelineComposerRequirements)}

correct_gp_algorithm_attributes = {field.name for field in dataclasses.fields(GPAlgorithmParameters)}


def get_api_params_repository(task_type: Optional[TaskTypesEnum] = None):
    task_type = task_type or TaskTypesEnum.ts_forecasting
    params_repository = ApiParamsRepository(task_type)
    return params_repository


@pytest.mark.parametrize('input_params', [fedot_params_full, params_with_missings])
def test_correctly_sets_default_params(input_params):
    params_repository = get_api_params_repository()
    output_params = params_repository.check_and_set_default_params(input_params)
    default_params = params_repository.default_params_for_task(params_repository.task_type)
    assert output_params.keys() <= default_params.keys()
    for k, v in default_params.items():
        if k not in input_params and v is not None:
            assert output_params[k] == default_params[k]
        elif k in input_params:
            assert output_params[k] == input_params[k]


@pytest.mark.parametrize('input_params', [fedot_params_full, params_with_missings])
@pytest.mark.parametrize('case, correct_keys', [('composer', correct_composer_attributes),
                                                ('gp_algo', correct_gp_algorithm_attributes)])
def test_filter_params_correctly(input_params, case, correct_keys):
    params_repository = get_api_params_repository()
    input_params = params_repository.check_and_set_default_params(input_params)
    if case == 'composer':
        output_params = params_repository.get_params_for_composer_requirements(input_params)
    elif case == 'gp_algo':
        output_params = params_repository.get_params_for_gp_algorithm_params(input_params)
    assert output_params.keys() <= correct_keys
    # check all correct parameter in input params are in output params
    assert (input_params.keys() & correct_keys) <= output_params.keys()
