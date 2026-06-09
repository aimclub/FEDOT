import datetime
from collections import UserDict
from copy import deepcopy, copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from golem.core.log import LoggerAdapter, default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.utilities.utilities import determine_n_jobs

from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.api_params_rules import (
    build_label_encoded_preset_name,
    merge_param_recommendations,
    normalize_timeout_and_generations,
    resolve_task,
    should_update_available_operations,
)
from fedot.api.api_utils.presets import OperationsPreset
from fedot.api.api_utils.tensor_data_config import resolve_tensor_data_config
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.input_data.data import InputData
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TaskParams
from fedot.api.api_utils.api_data_rules import TensorDataCreationRequest
from fedot.core.data.tensor_data.tensor_data import TensorData


class ApiParams(UserDict):

    def __init__(self, input_params: Dict[str, Any], problem: str, task_params: Optional[TaskParams] = None,
                 n_jobs: int = -1, timeout: float = 5, seed=None):
        self.log: LoggerAdapter = default_log(self)
        task_resolution = resolve_task(problem, task_params)
        if task_resolution.warning_message:
            self.log.warning(task_resolution.warning_message)
        self.task: Task = task_resolution.task
        self.n_jobs: int = determine_n_jobs(n_jobs)
        self.timeout = timeout

        self._params_repository = ApiParamsRepository(self.task.task_type)
        parameters: dict = self._params_repository.check_and_set_default_params(
            input_params)
        parameters['seed'] = seed
        super().__init__(parameters)
        self._check_timeout_vs_generations()

        self.tensor_data_config: Dict[str, Any] = resolve_tensor_data_config(
            self.get('tensor_data_config'),
            use_preprocessing_cache=self.get('use_preprocessing_cache', True),
        )

        self.composer_requirements = None
        self.graph_generation_params = None
        self.optimizer_params = None

    def prepare_tensordata_creation(
        self,
        *,
        target=None,
        is_predict: bool = False,
    ) -> TensorDataCreationRequest:
        config = dict(self.tensor_data_config)
        backend_name = config.pop('backend_name', 'cpu')

        spec_kwargs = {
            **config,
            'task': self.task,
            'state': StateEnum.PREDICT if is_predict else StateEnum.FIT,
        }
        if target is not None:
            spec_kwargs['target'] = target

        return TensorDataCreationRequest(backend_name=backend_name, spec_kwargs=spec_kwargs)

    def update_available_operations_by_preset(self, data: Union[InputData, TensorData]):
        """ Updates available_operations by preset and data type"""
        preset = self.get('preset')
        if should_update_available_operations(preset):
            preset_operations = OperationsPreset(
                task=self.task, preset_name=preset)
            self.data = preset_operations.composer_params_based_on_preset(
                self.data, data.data_type)

    def accept_and_apply_recommendations(self, input_data: Union[InputData, MultiModalData], recommendations: Dict):
        """
        Accepts recommendations for api params from DataAnalyser

        Args:
            input_data: data for preprocessing
            recommendations: dict with recommendations
        """

        if isinstance(input_data, MultiModalData):
            for data_source_name, values in input_data.items():
                self.accept_and_apply_recommendations(input_data[data_source_name],
                                                      recommendations[data_source_name])
        else:
            if 'label_encoded' in recommendations:
                self.log.info("Change preset due to label encoding")
                self.change_preset_for_label_encoded_data(
                    input_data.task, input_data.data_type)

            # update api params with recommendations obtained using meta rules
            self.data = merge_param_recommendations(self.data, recommendations)

    def change_preset_for_label_encoded_data(self, task: Task, data_type: DataTypesEnum):
        """ Change preset on tree like preset, if data had been label encoded """
        preset_name = build_label_encoded_preset_name(self.get('preset'))
        preset_operations = OperationsPreset(
            task=task, preset_name=preset_name)

        self.pop('available_operations', None)
        self.data = preset_operations.composer_params_based_on_preset(
            self.data, data_type)

    def _get_task_with_params(self, problem: str, task_params: Optional[TaskParams] = None) -> Task:
        """ Creates Task from problem name and task_params"""
        task_resolution = resolve_task(problem, task_params)
        if task_resolution.warning_message:
            self.log.warning(task_resolution.warning_message)
        return task_resolution.task

    def _check_timeout_vs_generations(self):
        timeout_resolution = normalize_timeout_and_generations(
            self.timeout, self.get('num_of_generations'))
        self.timeout = timeout_resolution.timeout
        self['num_of_generations'] = timeout_resolution.num_of_generations

    def init_params_for_composing(self, datetime_composing: Optional[datetime.timedelta], multi_objective: bool):
        """ Method to initialize ``PipelineComposerRequirements``, ``GPAlgorithmParameters``,
        ``GraphGenerationParams``"""
        self.init_composer_requirements(datetime_composing)
        self.init_optimizer_params(multi_objective=multi_objective)
        self.init_graph_generation_params(
            requirements=self.composer_requirements)

    def init_composer_requirements(self, datetime_composing: Optional[datetime.timedelta]) \
            -> PipelineComposerRequirements:
        """ Method to initialize ``PipelineComposerRequirements``"""
        preset = self['preset']

        # define available operations
        if not self.get('available_operations'):
            available_operations = OperationsPreset(
                self.task, preset).filter_operations_by_preset()
            self['available_operations'] = available_operations

        primary_operations, secondary_operations = \
            PipelineOperationRepository.divide_operations(
                self.get('available_operations'), self.task)

        composer_requirements_params = self._params_repository.get_params_for_composer_requirements(
            self.data)
        self.composer_requirements = PipelineComposerRequirements(primary=primary_operations,
                                                                  secondary=secondary_operations,
                                                                  timeout=datetime_composing,
                                                                  n_jobs=self.n_jobs, **composer_requirements_params)
        return self.composer_requirements

    def init_optimizer_params(self, multi_objective: bool) -> GPAlgorithmParameters:
        """Method to initialize ``GPAlgorithmParameters``"""
        gp_algorithm_parameters = self._params_repository.get_params_for_gp_algorithm_params(
            self.data)

        # workaround for "{TypeError}__init__() got an unexpected keyword argument 'seed'"
        seed = gp_algorithm_parameters['seed']
        del gp_algorithm_parameters['seed']

        self.optimizer_params = GPAlgorithmParameters(
            multi_objective=multi_objective, **gp_algorithm_parameters
        )
        self.optimizer_params.seed = seed
        return self.optimizer_params

    def init_graph_generation_params(self, requirements: PipelineComposerRequirements) -> GraphGenerationParams:
        """Method to initialize ``GraphGenerationParameters``"""
        preset = self['preset']
        available_operations = self['available_operations']
        advisor = PipelineChangeAdvisor(self.task)
        graph_model_repo = (PipelineOperationRepository()
                            .from_available_operations(task=self.task, preset=preset,
                                                       available_operations=available_operations))
        node_factory = (PipelineOptNodeFactory(requirements=requirements, advisor=advisor,
                                               graph_model_repository=graph_model_repo) if requirements else None)
        self.graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                                             rules_for_constraint=rules_by_task(
                                                                 self.task.task_type),
                                                             advisor=advisor,
                                                             node_factory=node_factory)
        return self.graph_generation_params

    def to_dict(self, deep=False) -> Dict:
        cp = deepcopy if deep else copy
        return cp(self.data)
