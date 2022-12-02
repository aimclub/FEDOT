import datetime
import gc
import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.presets import OperationsPreset
from fedot.api.time import ApiTime
from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.evaluation import determine_n_jobs
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.quality_metrics_repository import MetricsRepository, MetricType, MetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.utilities.define_metric_by_task import MetricByTask


class ApiComposer:

    def __init__(self, problem: str):
        self.log = default_log(self)
        self.metrics = ApiMetrics(problem)
        self.pipelines_cache: Optional[OperationsCache] = None
        self.preprocessing_cache: Optional[PreprocessingCache] = None
        self.preset_name = None
        self.timer = None
        # status flag indicating that composer step was applied
        self.was_optimised = False
        # status flag indicating that tuner step was applied
        self.was_tuned = False

    def obtain_metric(self, task: Task, metric: Union[str, MetricsEnum, Callable, Sequence]) -> Sequence[MetricType]:
        """Chooses metric to use for quality assessment of pipeline during composition"""
        if metric is None:
            metric = MetricByTask.get_default_quality_metrics(task.task_type)

        metric_ids = []
        for specific_metric in ensure_wrapped_in_sequence(metric):
            if isinstance(specific_metric, Callable):
                metric = specific_metric
            else:
                metric = None
                if isinstance(specific_metric, str):
                    # Composer metric was defined by name (str)
                    metric = self.metrics.get_metrics_mapping(metric_name=specific_metric)
                elif isinstance(specific_metric, MetricsEnum):
                    metric = specific_metric
            if metric is None:
                raise ValueError(f'Incorrect metric {specific_metric}')
            metric_ids.append(metric)
        return metric_ids

    def obtain_model(self, **common_dict):
        # Prepare parameters
        api_params_dict, composer_params_dict, tuner_params_dict = _divide_parameters(common_dict)
        # Start composing - pipeline structure search
        return self.compose_fedot_model(api_params_dict, composer_params_dict, tuner_params_dict)

    def init_cache(self, use_pipelines_cache: bool, use_preprocessing_cache: bool,
                   cache_folder: Optional[Union[str, os.PathLike]] = None):
        if use_pipelines_cache:
            self.pipelines_cache = OperationsCache(cache_folder)
            #  in case of previously generated singleton cache
            self.pipelines_cache.reset()
        if use_preprocessing_cache:
            self.preprocessing_cache = PreprocessingCache(cache_folder)
            #  in case of previously generated singleton cache
            self.preprocessing_cache.reset()

    @staticmethod
    def _init_composer_requirements(api_params: dict,
                                    composer_params: dict,
                                    datetime_composing: Optional[datetime.timedelta],
                                    preset: str) -> PipelineComposerRequirements:

        task = api_params['task']

        # define available operations
        available_operations = composer_params.get('available_operations',
                                                   OperationsPreset(task, preset).filter_operations_by_preset())
        primary_operations, secondary_operations = \
            PipelineOperationRepository.divide_operations(available_operations, task)

        max_pipeline_fit_time = composer_params['max_pipeline_fit_time']
        if max_pipeline_fit_time:
            max_pipeline_fit_time = datetime.timedelta(minutes=max_pipeline_fit_time)

        composer_requirements = PipelineComposerRequirements(
            primary=primary_operations,
            secondary=secondary_operations,
            max_arity=composer_params['max_arity'],
            max_depth=composer_params['max_depth'],

            num_of_generations=composer_params['num_of_generations'],
            timeout=datetime_composing,
            early_stopping_iterations=composer_params.get('early_stopping_iterations', None),
            early_stopping_timeout=composer_params.get('early_stopping_timeout', None),
            max_pipeline_fit_time=max_pipeline_fit_time,
            n_jobs=api_params['n_jobs'],
            show_progress=api_params['show_progress'],
            collect_intermediate_metric=composer_params['collect_intermediate_metric'],
            keep_n_best=composer_params['keep_n_best'],

            keep_history=True,
            history_dir=composer_params.get('history_folder'),

            cv_folds=composer_params['cv_folds'],
            validation_blocks=composer_params['validation_blocks'],
        )
        return composer_requirements

    @staticmethod
    def _init_optimizer_parameters(composer_params: dict,
                                   multi_objective: bool,
                                   task_type: TaskTypesEnum) -> GPGraphOptimizerParameters:
        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
        if composer_params['genetic_scheme'] == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        optimizer_params = GPGraphOptimizerParameters(
            multi_objective=multi_objective,
            pop_size=composer_params['pop_size'],
            genetic_scheme_type=genetic_scheme_type,
            mutation_types=ApiComposer._get_default_mutations(task_type)
        )
        return optimizer_params

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

    @staticmethod
    def _init_graph_generation_params(task: Task, preset: str, available_operations: List[str],
                                      requirements: PipelineComposerRequirements):
        advisor = PipelineChangeAdvisor(task)
        graph_model_repo = PipelineOperationRepository()\
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

    def compose_fedot_model(self, api_params: dict, composer_params: dict, tuning_params: dict) \
            -> Tuple[Pipeline, Sequence[Pipeline], OptHistory]:
        """ Function for composing FEDOT pipeline model """
        task: Task = api_params['task']
        train_data = api_params['train_data']
        timeout = api_params['timeout']
        with_tuning = tuning_params['with_tuning']
        available_operations = composer_params['available_operations']
        preset = composer_params['preset']

        self.timer = ApiTime(time_for_automl=timeout, with_tuning=with_tuning)

        # Work with initial assumptions
        assumption_handler = AssumptionsHandler(train_data)

        initial_assumption = assumption_handler.propose_assumptions(composer_params['initial_assumption'],
                                                                    available_operations)
        with self.timer.launch_assumption_fit():
            fitted_assumption = \
                assumption_handler.fit_assumption_and_check_correctness(initial_assumption[0],
                                                                        pipelines_cache=self.pipelines_cache,
                                                                        preprocessing_cache=self.preprocessing_cache)

        self.log.message(
            f'Initial pipeline was fitted in {round(self.timer.assumption_fit_spend_time.total_seconds(), 1)} sec.')

        n_jobs = determine_n_jobs(api_params['n_jobs'])
        self.preset_name = assumption_handler.propose_preset(preset, self.timer, n_jobs=n_jobs)

        composer_requirements = ApiComposer._init_composer_requirements(api_params, composer_params,
                                                                        self.timer.timedelta_composing,
                                                                        self.preset_name)

        metric_functions = self.obtain_metric(task, composer_params['metric'])
        graph_generation_params = \
            self._init_graph_generation_params(task=task,
                                               preset=preset,
                                               available_operations=composer_params.get('available_operations'),
                                               requirements=composer_requirements)
        self.log.message(f"AutoML configured."
                         f" Parameters tuning: {with_tuning}."
                         f" Time limit: {timeout} min."
                         f" Set of candidate models: {available_operations}.")

        best_pipeline, best_pipeline_candidates, gp_composer = self.compose_pipeline(task, train_data,
                                                                                     fitted_assumption,
                                                                                     metric_functions,
                                                                                     composer_requirements,
                                                                                     composer_params,
                                                                                     graph_generation_params)
        if with_tuning:
            best_pipeline = self.tune_final_pipeline(task, train_data,
                                                     metric_functions[0],
                                                     composer_requirements,
                                                     best_pipeline)
        if gp_composer.history:
            adapter = gp_composer.optimizer.graph_generation_params.adapter
            gp_composer.history.tuning_result = adapter.adapt(best_pipeline)
        # enforce memory cleaning
        gc.collect()

        self.log.message('Model generation finished')
        return best_pipeline, best_pipeline_candidates, gp_composer.history

    def compose_pipeline(self, task: Task,
                         train_data: InputData,
                         fitted_assumption: Pipeline,
                         metric_functions: Sequence[MetricsEnum],
                         composer_requirements: PipelineComposerRequirements,
                         composer_params: dict,
                         graph_generation_params: GraphGenerationParams,
                         ) -> Tuple[Pipeline, List[Pipeline], GPComposer]:

        multi_objective = len(metric_functions) > 1
        optimizer_params = ApiComposer._init_optimizer_parameters(composer_params,
                                                                  multi_objective=multi_objective,
                                                                  task_type=task.task_type)

        gp_composer: GPComposer = ComposerBuilder(task=task) \
            .with_requirements(composer_requirements) \
            .with_initial_pipelines(fitted_assumption) \
            .with_optimizer(composer_params.get('optimizer')) \
            .with_optimizer_params(parameters=optimizer_params,
                                   external_parameters=composer_params.get('optimizer_external_params')) \
            .with_metrics(metric_functions) \
            .with_cache(self.pipelines_cache, self.preprocessing_cache) \
            .with_graph_generation_param(graph_generation_params=graph_generation_params) \
            .build()

        n_jobs = determine_n_jobs(composer_requirements.n_jobs)
        if self.timer.have_time_for_composing(composer_params['pop_size'], n_jobs):
            # Launch pipeline structure composition
            with self.timer.launch_composing():
                self.log.message('Pipeline composition started.')
                self.was_optimised = False
                best_pipelines = gp_composer.compose_pipeline(data=train_data)
                best_pipeline_candidates = gp_composer.best_models
                self.was_optimised = True
        else:
            # Use initial pipeline as final solution
            self.log.message(f'Timeout is too small for composing and is skipped '
                             f'because fit_time is {self.timer.assumption_fit_spend_time.total_seconds()} sec.')
            best_pipelines = fitted_assumption
            best_pipeline_candidates = [fitted_assumption]

        for pipeline in best_pipeline_candidates:
            pipeline.log = self.log
        best_pipeline = best_pipelines[0] if isinstance(best_pipelines, Sequence) else best_pipelines
        return best_pipeline, best_pipeline_candidates, gp_composer

    def tune_final_pipeline(self, task: Task,
                            train_data: InputData,
                            metric_function: Optional[MetricType],
                            composer_requirements: PipelineComposerRequirements,
                            pipeline_gp_composed: Pipeline,
                            ) -> Pipeline:
        """ Launch tuning procedure for obtained pipeline by composer """
        timeout_for_tuning = abs(self.timer.determine_resources_for_tuning()) / 60
        tuner = TunerBuilder(task) \
            .with_tuner(PipelineTuner) \
            .with_metric(metric_function) \
            .with_iterations(DEFAULT_TUNING_ITERATIONS_NUMBER) \
            .with_timeout(datetime.timedelta(minutes=timeout_for_tuning)) \
            .with_eval_time_constraint(composer_requirements.max_pipeline_fit_time) \
            .with_requirements(composer_requirements) \
            .build(train_data)

        if self.timer.have_time_for_tuning():
            # Tune all nodes in the pipeline
            with self.timer.launch_tuning():
                self.was_tuned = False
                self.log.message(f'Hyperparameters tuning started with {round(timeout_for_tuning)} min. timeout')
                tuned_pipeline = tuner.tune(pipeline_gp_composed)
                self.was_tuned = True
                self.log.message('Hyperparameters tuning finished')
        else:
            self.log.message(f'Time for pipeline composing was {str(self.timer.composing_spend_time)}.\n'
                             f'The remaining {max(0, round(timeout_for_tuning, 1))} seconds are not enough '
                             f'to tune the hyperparameters.')
            self.log.message('Composed pipeline returned without tuning.')
            tuned_pipeline = pipeline_gp_composed

        return tuned_pipeline


def _divide_parameters(common_dict: dict) -> List[dict]:
    """ Divide common dictionary into dictionary with parameters for API, Composer and Tuner

    :param common_dict: dictionary with parameters for all AutoML modules
    """
    api_params_dict = dict(train_data=None, task=Task, timeout=5, n_jobs=1, show_progress=True, logger=None)

    composer_params_dict = dict(max_depth=None, max_arity=None, pop_size=None, num_of_generations=None,
                                keep_n_best=None, available_operations=None, metric=None,
                                validation_blocks=None, cv_folds=None, genetic_scheme=None, history_folder=None,
                                early_stopping_iterations=None, early_stopping_timeout=None, optimizer=None,
                                optimizer_external_params=None, collect_intermediate_metric=False,
                                max_pipeline_fit_time=None, initial_assumption=None, preset='auto',
                                use_pipelines_cache=True, use_preprocessing_cache=True, cache_folder=None,
                                keep_history=True, history_dir=None,)

    tuner_params_dict = dict(with_tuning=False)

    dict_list = [api_params_dict, composer_params_dict, tuner_params_dict]
    for k, v in common_dict.items():
        is_unknown_key = True
        for i, dct in enumerate(dict_list):
            if k in dict_list[i]:
                dict_list[i][k] = v
                is_unknown_key = False
        if is_unknown_key:
            raise KeyError(f"Invalid key parameter {k}")

    return dict_list
