import datetime
from typing import Sequence


from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum

from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, boosting_mutation
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.repository.tasks import TaskTypesEnum


class ApiParamsRepository:
    """Repository storing possible Api parameters and their default values. Also returns parameters required
    for different data classes used while model composition.
     """

    COMPOSER_REQUIREMENTS_KEYS = ['max_arity', 'max_depth', 'num_of_generations',
                                  'early_stopping_iterations', 'early_stopping_timeout',
                                  'max_pipeline_fit_time', 'parallelization_mode',
                                  'use_input_preprocessing', 'show_progress',
                                  'collect_intermediate_metric', 'keep_n_best',
                                  'keep_history', 'history_dir', 'cv_folds', 'validation_blocks']

    METADATA_KEYS = ['use_input_preprocessing']

    def __init__(self, task_type: TaskTypesEnum):
        self.task_type = task_type
        self.default_params = ApiParamsRepository.default_params_for_task(self.task_type)

    @staticmethod
    def default_params_for_task(task_type: TaskTypesEnum) -> dict:
        """ Returns a dict with default parameters"""
        if task_type in [TaskTypesEnum.classification, TaskTypesEnum.regression]:
            cv_folds = 5
            validation_blocks = None

        elif task_type == TaskTypesEnum.ts_forecasting:
            cv_folds = 3
            validation_blocks = 2

        default_params_dict = dict(
            parallelization_mode='populational',
            show_progress=True,
            max_depth=6,
            max_arity=3,
            pop_size=20,
            num_of_generations=None,
            keep_n_best=1,
            available_operations=None,
            metric=None,
            validation_blocks=validation_blocks,
            cv_folds=cv_folds,
            genetic_scheme=None,
            early_stopping_iterations=None,
            early_stopping_timeout=10,
            optimizer=None,
            optimizer_external_params=None,
            collect_intermediate_metric=False,
            max_pipeline_fit_time=None,
            initial_assumption=None,
            preset=AUTO_PRESET_NAME,
            use_pipelines_cache=True,
            use_preprocessing_cache=True,
            use_input_preprocessing=True,
            cache_folder=None,
            keep_history=True,
            history_dir=None,
            with_tuning=False
        )
        return default_params_dict

    def check_and_set_default_params(self, params: dict) -> dict:
        """ Sets default values for parameters which were not set by the user
        and raises KeyError for invalid parameter keys"""
        allowed_keys = set(self.default_params.keys())
        invalid_keys = set(params.keys()) - allowed_keys
        if invalid_keys:
            raise KeyError(f"Invalid key parameters {invalid_keys}")

        else:
            for k, v in self.default_params.items():
                if k not in params and v is not None:
                    params[k] = v
        return params

    @staticmethod
    def get_params_for_composer_requirements(params: dict) -> dict:

        composer_requirements_params = {k: v for k, v in params.items()
                                        if k in ApiParamsRepository.COMPOSER_REQUIREMENTS_KEYS}

        max_pipeline_fit_time = composer_requirements_params.get('max_pipeline_fit_time')
        if max_pipeline_fit_time:
            composer_requirements_params['max_graph_fit_time'] = datetime.timedelta(minutes=max_pipeline_fit_time)
        composer_requirements_params.pop('max_pipeline_fit_time', None)

        composer_requirements_params = ApiParamsRepository.set_static_individual_metadata(composer_requirements_params)

        return composer_requirements_params

    @staticmethod
    def set_static_individual_metadata(composer_requirements_params: dict) -> dict:
        static_individual_metadata = {k: v for k, v in composer_requirements_params.items()
                                      if k in ApiParamsRepository.METADATA_KEYS}
        for k in ApiParamsRepository.METADATA_KEYS:
            composer_requirements_params.pop(k)

        composer_requirements_params['static_individual_metadata'] = static_individual_metadata
        return composer_requirements_params

    def get_params_for_gp_algorithm_params(self, params: dict) -> dict:
        gp_algorithm_params = {'pop_size': params.get('pop_size'),
                               'genetic_scheme_type': GeneticSchemeTypesEnum.parameter_free}
        if params.get('genetic_scheme') == 'steady_state':
            gp_algorithm_params['genetic_scheme_type'] = GeneticSchemeTypesEnum.steady_state

        gp_algorithm_params['mutation_types'] = ApiParamsRepository._get_default_mutations(self.task_type)
        return gp_algorithm_params

    @staticmethod
    def _get_default_mutations(task_type: TaskTypesEnum) -> Sequence[MutationTypesEnum]:
        mutations = [parameter_change_mutation,
                     MutationTypesEnum.single_change,
                     MutationTypesEnum.single_drop,
                     MutationTypesEnum.single_add,
                     MutationTypesEnum.single_edge]

        # TODO remove workaround after boosting mutation fix
        if task_type == TaskTypesEnum.ts_forecasting:
            mutations.append(boosting_mutation)

        return mutations
