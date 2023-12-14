import datetime
from typing import Sequence

from fedot.core.optimisers.stages_builder.optimizer_stages_builder import OptimizerStagesBuilder
from golem.core.optimisers.common_optimizer.common_optimizer import CommonOptimizer
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum

from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, add_resample_mutation
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import default_fedot_data_dir
from golem.core.optimisers.optimizer import GraphOptimizer, AlgorithmParameters


class ApiParamsRepository:
    """Repository storing possible Api parameters and their default values. Also returns parameters required
    for data classes (``PipelineComposerRequirements``, ``GPAlgorithmParameters``, ``GraphGenerationParams``)
    used while model composition.
     """

    COMPOSER_REQUIREMENTS_KEYS = {'max_arity', 'max_depth', 'num_of_generations',
                                  'early_stopping_iterations', 'early_stopping_timeout',
                                  'parallelization_mode', 'use_input_preprocessing',
                                  'show_progress', 'collect_intermediate_metric', 'keep_n_best',
                                  'keep_history', 'history_dir', 'cv_folds'}

    STATIC_INDIVIDUAL_METADATA_KEYS = {'use_input_preprocessing'}

    def __init__(self, task_type: TaskTypesEnum):
        self.task_type = task_type
        self.default_params = ApiParamsRepository.default_params_for_task(self.task_type)

    @staticmethod
    def default_params_for_task(task_type: TaskTypesEnum) -> dict:
        """ Returns a dict with default parameters"""
        if task_type in [TaskTypesEnum.classification, TaskTypesEnum.regression]:
            cv_folds = 5

        elif task_type == TaskTypesEnum.ts_forecasting:
            cv_folds = 3

        # Dict with allowed keyword attributes for Api and their default values. If None - default value set
        # in dataclasses ``PipelineComposerRequirements``, ``GPAlgorithmParameters``, ``GraphGenerationParams``
        # will be used.
        default_param_values_dict = dict(
            parallelization_mode='populational',
            show_progress=True,
            max_depth=6,
            max_arity=3,
            pop_size=20,
            num_of_generations=None,
            keep_n_best=1,
            available_operations=None,
            metric=None,
            cv_folds=cv_folds,
            genetic_scheme=None,
            early_stopping_iterations=None,
            early_stopping_timeout=10,
            optimizer=None,
            collect_intermediate_metric=False,
            max_pipeline_fit_time=None,
            initial_assumption=None,
            preset=AUTO_PRESET_NAME,
            use_pipelines_cache=True,
            use_preprocessing_cache=True,
            use_input_preprocessing=True,
            use_auto_preprocessing=False,
            use_meta_rules=False,
            cache_dir=default_fedot_data_dir(),
            keep_history=True,
            history_dir=default_fedot_data_dir(),
            with_tuning=True
        )
        return default_param_values_dict

    def check_and_set_default_params(self, params: dict) -> dict:
        """ Sets default values for parameters which were not set by the user
        and raises KeyError for invalid parameter keys"""
        allowed_keys = self.default_params.keys()
        invalid_keys = params.keys() - allowed_keys
        if invalid_keys:
            raise KeyError(f"Invalid key parameters {invalid_keys}")
        else:
            missing_params = self.default_params.keys() - params.keys()
            for k in missing_params:
                if (v := self.default_params[k]) is not None:
                    params[k] = v
        return params

    @staticmethod
    def get_params_for_composer_requirements(params: dict) -> dict:
        """ Returns dict with parameters suitable for ``PipelineComposerParameters``"""
        composer_requirements_params = {k: v for k, v in params.items()
                                        if k in ApiParamsRepository.COMPOSER_REQUIREMENTS_KEYS}

        max_pipeline_fit_time = params.get('max_pipeline_fit_time')
        if max_pipeline_fit_time:
            composer_requirements_params['max_graph_fit_time'] = datetime.timedelta(minutes=max_pipeline_fit_time)

        composer_requirements_params = ApiParamsRepository.set_static_individual_metadata(composer_requirements_params)

        return composer_requirements_params

    @staticmethod
    def set_static_individual_metadata(composer_requirements_params: dict) -> dict:
        """ Returns dict with representing ``static_individual_metadata`` for ``PipelineComposerParameters``"""
        static_individual_metadata = {k: v for k, v in composer_requirements_params.items()
                                      if k in ApiParamsRepository.STATIC_INDIVIDUAL_METADATA_KEYS}
        for k in ApiParamsRepository.STATIC_INDIVIDUAL_METADATA_KEYS:
            composer_requirements_params.pop(k)

        composer_requirements_params['static_individual_metadata'] = static_individual_metadata
        return composer_requirements_params

    def get_params_for_algorithm_params(self,
                                        multi_objective: bool,
                                        params: dict) -> (GraphOptimizer, AlgorithmParameters):
        # define parameters
        genetic_scheme_type = (GeneticSchemeTypesEnum.steady_state
                               if params.get('genetic_scheme') == 'steady_state'
                               else GeneticSchemeTypesEnum.parameter_free)
        mutation_types = ApiParamsRepository._get_default_mutations(self.task_type, params)
        algorithm_params = GPAlgorithmParameters(multi_objective=multi_objective,
                                                 pop_size=params.get('pop_size'),
                                                 genetic_scheme_type=genetic_scheme_type,
                                                 mutation_types=mutation_types)

        # define optimizer
        if self.task_type is TaskTypesEnum.ts_forecasting:
            algorithm = CommonOptimizer
            algorithm_params.stages = OptimizerStagesBuilder(self.task_type).build()
        else:
            algorithm = EvoGraphOptimizer

        return algorithm, algorithm_params

    @staticmethod
    def _get_default_mutations(task_type: TaskTypesEnum, params) -> Sequence[MutationTypesEnum]:
        mutations = [parameter_change_mutation,
                     MutationTypesEnum.single_change,
                     MutationTypesEnum.single_drop,
                     MutationTypesEnum.single_add,
                     MutationTypesEnum.single_edge]

        # TODO remove workaround after boosting mutation fix
        #      Boosting mutation does not work due to problem with __eq__ with it copy.
        #      ``partial`` refactor to ``def`` does not work
        #      Also boosting mutation does not work by it own.
        if task_type == TaskTypesEnum.ts_forecasting:
            # mutations.append(partial(boosting_mutation, params=params))
            pass
        else:
            mutations.append(add_resample_mutation)

        return mutations
