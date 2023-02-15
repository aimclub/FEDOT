import datetime
from typing import Any, Dict, List, Optional, Union, Sequence

from golem.core.log import LoggerAdapter
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum

from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, boosting_mutation
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.tasks import Task, TaskParams, TaskTypesEnum, TsForecastingParams
from fedot.core.utilities.random import RandomStateHandler


class ApiParams:

    def __init__(self, input_params: Dict[str, Any], task: Task, n_jobs: int,
                 timeout: float):
        self.log = default_log(self)
        self.task = task
        self.n_jobs = n_jobs
        self.timeout = timeout
        self._parameters = self._set_default_params(input_params)
        self._check_timeout_vs_generations()

        self.composer_requirements = None
        self.graph_generation_params = None
        self.optimizer_params = None

    def get(self, key: str, default_value=None) -> Any:
        return self._parameters.get(key, default_value)

    def update(self, **params):
        self._parameters.update(**params)

    def to_dict(self):
        return self._parameters

    def update_available_operations_by_preset(self, data: InputData):
        preset = self._parameters.get('preset')
        if preset != AUTO_PRESET_NAME:
            preset_operations = OperationsPreset(task=self.task, preset_name=preset)
            self._parameters = preset_operations.composer_params_based_on_preset(self._parameters,
                                                                                 data.data_type)

    def accept_and_apply_recommendations(self, input_data: Union[InputData, MultiModalData], recommendations: Dict):
        """
        Accepts recommendations for api params from DataAnalyser

        :param input_data - data for preprocessing
        :param recommendations - dict with recommendations
        """
        # TODO fix multimodality
        if isinstance(input_data, MultiModalData):
            self._parameters['cv_folds'] = None  # there are no support for multimodal data now
            for data_source_name, values in input_data.items():
                self.accept_and_apply_recommendations(input_data[data_source_name],
                                                      recommendations[data_source_name])
        else:
            if 'label_encoded' in recommendations:
                self.log.info("Change preset due to label encoding")
                self.change_preset_for_label_encoded_data(input_data.task, input_data.data_type)

    def change_preset_for_label_encoded_data(self, task: Task, data_type: DataTypesEnum):
        """ Change preset on tree like preset, if data had been label encoded """
        if 'preset' in self._parameters:
            preset_name = ''.join((self._parameters['preset'], '*tree'))
        else:
            preset_name = '*tree'
        preset_operations = OperationsPreset(task=task, preset_name=preset_name)

        if self._parameters.get('available_operations') is not None:
            del self._parameters['available_operations']
        self._parameters = preset_operations.composer_params_based_on_preset(self._parameters, data_type)

    def _set_default_params(self, composer_tuner_params: dict) -> dict:
        """ Sets default values for parameters which were not set by the user """
        default_params_dict = self._get_default_params()

        for k, v in composer_tuner_params.items():
            if k in default_params_dict:
                default_params_dict[k] = v
            else:
                raise KeyError(f"Invalid key parameter {k}")
        return default_params_dict

    def _get_default_params(self) -> dict:
        """ Returns a dict with default parameters"""
        if self.task.task_type in [TaskTypesEnum.classification, TaskTypesEnum.regression]:
            cv_folds = 5
            validation_blocks = None

        elif self.task.task_type == TaskTypesEnum.ts_forecasting:
            cv_folds = 3
            validation_blocks = 2

        # If early_stopping_iterations is not specified,
        # than estimate it as in time-based manner as: timeout / 3.
        # The minimal number of generations is 5.
        early_stopping_iterations = None
        if self.timeout:
            depending_on_timeout = int(self.timeout / 3)
            early_stopping_iterations = max(depending_on_timeout, 5)

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
            early_stopping_iterations=early_stopping_iterations,
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

    def _check_timeout_vs_generations(self):
        num_of_generations = self._parameters['num_of_generations']
        if self.timeout in [-1, None]:
            self.timeout = None
            if num_of_generations is None:
                raise ValueError('"num_of_generations" should be specified if infinite "timeout" is given')
        elif self.timeout > 0:
            if num_of_generations is None:
                self._parameters['num_of_generations'] = 10000
        else:
            raise ValueError(f'invalid "timeout" value: timeout={self.timeout}')

    def init_composer_requirements(self, datetime_composing: Optional[datetime.timedelta]) \
            -> PipelineComposerRequirements:

        preset = self._parameters['preset']

        # define available operations
        if not self._parameters.get('available_operations'):
            available_operations = OperationsPreset(self.task, preset).filter_operations_by_preset()
            self._parameters['available_operations'] = available_operations

        primary_operations, secondary_operations = \
            PipelineOperationRepository.divide_operations(self._parameters.get('available_operations'), self.task)

        max_pipeline_fit_time = self._parameters['max_pipeline_fit_time']
        if max_pipeline_fit_time:
            max_pipeline_fit_time = datetime.timedelta(minutes=max_pipeline_fit_time)

        self.composer_requirements = PipelineComposerRequirements(
            primary=primary_operations,
            secondary=secondary_operations,
            max_arity=self._parameters['max_arity'],
            max_depth=self._parameters['max_depth'],

            num_of_generations=self._parameters['num_of_generations'],
            timeout=datetime_composing,
            early_stopping_iterations=self._parameters['early_stopping_iterations'],
            early_stopping_timeout=self._parameters['early_stopping_timeout'],
            max_pipeline_fit_time=max_pipeline_fit_time,
            n_jobs=self.n_jobs,
            parallelization_mode=self._parameters['parallelization_mode'],
            static_individual_metadata={
                k: v for k, v in self._parameters.items()
                if k in ['use_input_preprocessing']
            },
            show_progress=self._parameters['show_progress'],
            collect_intermediate_metric=self._parameters['collect_intermediate_metric'],
            keep_n_best=self._parameters['keep_n_best'],

            keep_history=True,
            history_dir=self._parameters['history_dir'],

            cv_folds=self._parameters['cv_folds'],
            validation_blocks=self._parameters['validation_blocks'],
        )
        return self.composer_requirements

    def init_optimizer_params(self, multi_objective: bool) -> GPGraphOptimizerParameters:

        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
        if self._parameters['genetic_scheme'] == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        self.optimizer_params = GPGraphOptimizerParameters(
            multi_objective=multi_objective,
            pop_size=self._parameters['pop_size'],
            genetic_scheme_type=genetic_scheme_type,
            mutation_types=ApiParams._get_default_mutations(self.task.task_type)
        )
        return self.optimizer_params

    def init_graph_generation_params(self, requirements: PipelineComposerRequirements):
        preset = self._parameters['preset']
        available_operations = self._parameters['available_operations']
        advisor = PipelineChangeAdvisor(self.task)
        graph_model_repo = PipelineOperationRepository() \
            .from_available_operations(task=self.task, preset=preset,
                                       available_operations=available_operations)
        node_factory = PipelineOptNodeFactory(requirements=requirements, advisor=advisor,
                                              graph_model_repository=graph_model_repo) \
            if requirements else None
        self.graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                                             rules_for_constraint=rules_by_task(self.task.task_type),
                                                             advisor=advisor,
                                                             node_factory=node_factory)
        return self.graph_generation_params

    @staticmethod
    def _get_default_mutations(task_type: TaskTypesEnum) -> Sequence[MutationTypesEnum]:
        mutations = [parameter_change_mutation,
                     MutationTypesEnum.single_change,
                     MutationTypesEnum.single_drop,
                     MutationTypesEnum.single_add]

        # TODO remove workaround after boosting mutation fix
        if task_type == TaskTypesEnum.ts_forecasting:
            mutations.append(boosting_mutation)
        # TODO remove workaround after validation fix
        if task_type is not TaskTypesEnum.ts_forecasting:
            mutations.append(MutationTypesEnum.single_edge)

        return mutations
