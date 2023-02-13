import random
import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

import numpy as np

from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, boosting_mutation
from fedot.core.constants import AUTO_PRESET_NAME, DEFAULT_FORECAST_LENGTH
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.log import LoggerAdapter
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
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

    def __init__(self, input_params: Dict[str, Any]):
        self.log: LoggerAdapter = None
        self.task: Task = None
        self.task_params: TaskParams = None
        self.metric_name: Union[str, List[str]] = None
        self._all_parameters = self._initialize_params(input_params)

    def get(self, key: str, default_value=None) -> Any:
        return self._all_parameters.get(key, default_value)

    def update(self, **params):
        self._all_parameters.update(**params)

    def to_dict(self):
        return self._all_parameters

    def _initialize_params(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Merge input_params dictionary with several parameters for AutoML algorithm """
        params = self.get_initial_params(input_params)

        # Final check for correctness for timeout and generations
        params = check_timeout_vs_generations(params)

        return params

    def update_available_operations_by_preset(self, data: InputData):
        preset = self._all_parameters.get('preset')
        if preset != AUTO_PRESET_NAME:
            preset_operations = OperationsPreset(task=self.task, preset_name=preset)
            self._all_parameters = preset_operations.composer_params_based_on_preset(self._all_parameters,
                                                                                     data.data_type)

    def get_initial_params(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        params = self._parse_input_params(input_params)

        param_dict = {
            'task': self.task,
            'logger': self.log
        }
        params = {**params, **param_dict}
        return params

    def accept_and_apply_recommendations(self, input_data: Union[InputData, MultiModalData], recommendations: Dict):
        """
        Accepts recommendations for api params from DataAnalyser

        :param input_data - data for preprocessing
        :param recommendations - dict with recommendations
        """
        # TODO fix multimodality
        if isinstance(input_data, MultiModalData):
            self._all_parameters['cv_folds'] = None  # there are no support for multimodal data now
            for data_source_name, values in input_data.items():
                self.accept_and_apply_recommendations(input_data[data_source_name],
                                                      recommendations[data_source_name])
        else:
            if 'label_encoded' in recommendations:
                self.log.info("Change preset due to label encoding")
                self.change_preset_for_label_encoded_data(input_data.task, input_data.data_type)

    def change_preset_for_label_encoded_data(self, task: Task, data_type: DataTypesEnum):
        """ Change preset on tree like preset, if data had been label encoded """
        if 'preset' in self._all_parameters:
            preset_name = ''.join((self._all_parameters['preset'], '*tree'))
        else:
            preset_name = '*tree'
        preset_operations = OperationsPreset(task=task, preset_name=preset_name)

        if self._all_parameters.get('available_operations') is not None:
            del self._all_parameters['available_operations']
        self._all_parameters = preset_operations.composer_params_based_on_preset(self._all_parameters, data_type)

    def _parse_input_params(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Parses input params into different class fields """

        # reset logging level for Singleton
        Log().reset_logging_level(input_params['logging_level'])
        self.log = default_log(prefix='FEDOT logger')

        composer_tuner_params = self.set_default_params(input_params['composer_tuner_params'], input_params['problem'])

        simple_keys = ['problem', 'n_jobs', 'parallelization_mode', 'timeout']
        params = {k: input_params[k] for k in simple_keys}

        params = {**composer_tuner_params, **params}

        # If early_stopping_iterations is not specified,
        # than estimate it as in time-based manner as: timeout / 3.
        # The minimal number of generations is 5.
        if params['early_stopping_iterations'] is None:
            if params['timeout']:
                depending_on_timeout = int(params['timeout'] / 3)
                params['early_stopping_iterations'] = max(depending_on_timeout, 5)

        specified_seed = input_params['seed']
        if specified_seed is not None:
            np.random.seed(specified_seed)
            random.seed(specified_seed)
            RandomStateHandler.MODEL_FITTING_SEED = specified_seed

        if params['problem'] == 'ts_forecasting' and input_params['task_params'] is None:
            self.log.warning(f'The value of the forecast depth was set to {DEFAULT_FORECAST_LENGTH}.')
            self.task_params = TsForecastingParams(forecast_length=DEFAULT_FORECAST_LENGTH)

        if params['problem'] == 'clustering':
            raise ValueError('This type of task is not supported in API now')

        self.task = self.get_task(params['problem'], self.task_params)
        self.metric_name = self.get_default_metric(params['problem'])
        params.pop('problem')
        return params

    @staticmethod
    def set_default_params(composer_tuner_params: dict, problem: str):
        """ Sets default values for parameters which were not set by the user """
        default_params_dict = ApiParams.get_default_params(problem)

        for k, v in composer_tuner_params.items():
            if k in default_params_dict:
                default_params_dict[k] = v
            else:
                raise KeyError(f"Invalid key parameter {k}")
        return default_params_dict

    @staticmethod
    def get_default_params(problem: str) -> dict:
        """ Returns a dict with default parameters"""
        if problem in ['classification', 'regression']:
            cv_folds = 5
            validation_blocks = None

        elif problem == 'ts_forecasting':
            cv_folds = 3
            validation_blocks = 2

        default_params_dict = dict(train_data=None,
                                   task=Task,
                                   n_jobs=1,
                                   parallelization_mode='populational',
                                   show_progress=True,
                                   logger=None,
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
                                   history_folder=None,
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
                                   'use_input_preprocessing': True,
                                   cache_folder=None,
                                   keep_history=True,
                                   history_dir=None,
                                   with_tuning=False
                                   )
        return default_params_dict

    @staticmethod
    def get_default_metric(problem: str) -> Union[str, List[str]]:
        default_test_metric_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return default_test_metric_dict[problem]

    @staticmethod
    def get_task(problem: str, task_params: Optional[TaskParams] = None):
        """ Return task by the given ML problem name and the parameters """
        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'clustering': Task(TaskTypesEnum.clustering, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)}
        try:
            return task_dict[problem]
        except ValueError as exc:
            ValueError('Wrong type name of the given task')

    def init_composer_requirements(self, datetime_composing: Optional[datetime.timedelta]) \
            -> PipelineComposerRequirements:

        # api_params, composer_params, _ = _divide_parameters(self._all_parameters)
        preset = self._all_parameters['preset']

        task = self._all_parameters['task']

        # define available operations
        if not self._all_parameters.get('available_operations'):
            available_operations = OperationsPreset(task, preset).filter_operations_by_preset()
            self._all_parameters['available_operations'] = available_operations

        primary_operations, secondary_operations = \
            PipelineOperationRepository.divide_operations(self._all_parameters.get('available_operations'), task)

        max_pipeline_fit_time = self._all_parameters['max_pipeline_fit_time']
        if max_pipeline_fit_time:
            max_pipeline_fit_time = datetime.timedelta(minutes=max_pipeline_fit_time)

        composer_requirements = PipelineComposerRequirements(
            primary=primary_operations,
            secondary=secondary_operations,
            max_arity=self._all_parameters['max_arity'],
            max_depth=self._all_parameters['max_depth'],

            num_of_generations=self._all_parameters['num_of_generations'],
            timeout=datetime_composing,
            early_stopping_iterations=self._all_parameters['early_stopping_iterations'],
            early_stopping_timeout=self._all_parameters['early_stopping_timeout'],
            max_pipeline_fit_time=max_pipeline_fit_time,
            n_jobs=self._all_parameters['n_jobs'],
            parallelization_mode=self._all_parameters['parallelization_mode'],
            static_individual_metadata={
                k: v for k, v in self._all_parameters.items()
                if k in ['use_input_preprocessing']
            },
            show_progress=self._all_parameters['show_progress'],
            collect_intermediate_metric=self._all_parameters['collect_intermediate_metric'],
            keep_n_best=self._all_parameters['keep_n_best'],

            keep_history=True,
            history_dir=self._all_parameters['history_folder'],

            cv_folds=self._all_parameters['cv_folds'],
            validation_blocks=self._all_parameters['validation_blocks'],
        )
        return composer_requirements

    def init_optimizer_parameters(self, multi_objective: bool) -> GPGraphOptimizerParameters:
        # _, composer_params, _ = _divide_parameters(self._all_parameters)

        task = self._all_parameters.get('task')

        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
        if self._all_parameters['genetic_scheme'] == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        optimizer_params = GPGraphOptimizerParameters(
            multi_objective=multi_objective,
            pop_size=self._all_parameters['pop_size'],
            genetic_scheme_type=genetic_scheme_type,
            mutation_types=ApiParams._get_default_mutations(task.task_type)
        )
        return optimizer_params

    def init_graph_generation_params(self, requirements: PipelineComposerRequirements):
        task = self._all_parameters['task']
        preset = self._all_parameters['preset']
        available_operations = self._all_parameters['available_operations']
        advisor = PipelineChangeAdvisor(task)
        graph_model_repo = PipelineOperationRepository() \
            .from_available_operations(task=task, preset=preset,
                                       available_operations=available_operations)
        node_factory = PipelineOptNodeFactory(requirements=requirements, advisor=advisor,
                                              graph_model_repository=graph_model_repo) \
            if requirements else None
        graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                                        rules_for_constraint=rules_by_task(task.task_type),
                                                        advisor=advisor,
                                                        node_factory=node_factory)
        return graph_generation_params

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


def check_timeout_vs_generations(params) -> Dict[str, Any]:
    timeout = params['timeout']
    num_of_generations = params['num_of_generations']
    if timeout in [-1, None]:
        params['timeout'] = None
        if num_of_generations is None:
            raise ValueError('"num_of_generations" should be specified if infinite "timeout" is given')
    elif timeout > 0:
        if num_of_generations is None:
            params['num_of_generations'] = 10000
    else:
        raise ValueError(f'invalid "timeout" value: timeout={timeout}')
    return params
