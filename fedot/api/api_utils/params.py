import datetime
from typing import Any, Dict, Optional, Union, Sequence

from golem.core.log import LoggerAdapter, default_log
from golem.core.optimisers.genetic.evaluation import determine_n_jobs
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, boosting_mutation
from fedot.core.constants import AUTO_PRESET_NAME, DEFAULT_FORECAST_LENGTH
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TaskParams, TsForecastingParams


class ApiParams:

    def __init__(self, input_params: Dict[str, Any], problem: str, task_params: Optional[TaskParams] = None,
                 n_jobs: int = -1, timeout: float = 5):
        self.log: LoggerAdapter = default_log(self)
        self.task: Task = self._get_task_with_params(problem, task_params)
        self.n_jobs: int = determine_n_jobs(n_jobs)
        self.timeout = timeout

        self._params_repository = ApiParamsRepository(self.task.task_type)
        self._parameters: dict = self._params_repository.check_and_set_default_params(input_params)
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

    def _get_task_with_params(self, problem: str, task_params: Optional[TaskParams] = None):
        if problem == 'ts_forecasting' and task_params is None:
            self.log.warning(f'The value of the forecast depth was set to {DEFAULT_FORECAST_LENGTH}.')
            task_params = TsForecastingParams(forecast_length=DEFAULT_FORECAST_LENGTH)

        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        try:
            return task_dict[problem]
        except ValueError as exc:
            ValueError('Wrong type name of the given task')

    def _check_timeout_vs_generations(self):
        num_of_generations = self._parameters.get('num_of_generations')
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

        composer_requirements_params = self._params_repository.get_params_for_composer_requirements(self._parameters)
        self.composer_requirements = PipelineComposerRequirements(
            primary=primary_operations,
            secondary=secondary_operations,
            timeout=datetime_composing,
            n_jobs=self.n_jobs, **composer_requirements_params
        )
        return self.composer_requirements

    def init_optimizer_params(self, multi_objective: bool) -> GPAlgorithmParameters:

        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
        if self._parameters.get('genetic_scheme') == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        self.optimizer_params = GPAlgorithmParameters(
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
