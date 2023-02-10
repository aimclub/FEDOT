import datetime
import gc
import os
from itertools import chain
from typing import Callable, List, Optional, Sequence, Tuple, Union

from golem.core.log import default_log
from golem.core.optimisers.genetic.evaluation import determine_n_jobs
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.params import ApiParams
from fedot.api.time import ApiTime
from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.evaluation import determine_n_jobs
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.quality_metrics_repository import MetricType, MetricsEnum
from fedot.core.repository.tasks import Task
from fedot.utilities.define_metric_by_task import MetricByTask


class ApiComposer:

    def __init__(self, problem: str, api_params: ApiParams):
        self.log = default_log(self)
        self.params = api_params
        self.metrics = ApiMetrics(problem)
        self.pipelines_cache: Optional[OperationsCache] = None
        self.preprocessing_cache: Optional[PreprocessingCache] = None
        self.timer = None
        self.metric_names = None
        # status flag indicating that composer step was applied
        self.was_optimised = False
        # status flag indicating that tuner step was applied
        self.was_tuned = False
        self.tuner_requirements = None

    def obtain_metric(self, task: Task) -> Sequence[MetricType]:
        """Chooses metric to use for quality assessment of pipeline during composition"""
        metric = self.params.get('metric')
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

    def init_cache(self, use_pipelines_cache: bool = True,
                   use_input_preprocessing: bool = True, use_preprocessing_cache: bool = True,
                   cache_folder: Optional[Union[str, os.PathLike]] = None):
        if use_pipelines_cache:
            self.pipelines_cache = OperationsCache(cache_folder)
            #  in case of previously generated singleton cache
            self.pipelines_cache.reset()
        if use_input_preprocessing and use_preprocessing_cache:
            self.preprocessing_cache = PreprocessingCache(cache_folder)
            #  in case of previously generated singleton cache
            self.preprocessing_cache.reset()

    # def set_tuner_requirements(self, **common_dict):
    #     api_params, composer_params, _ = _divide_parameters(common_dict)
    #
    #     self.tuner_requirements = PipelineComposerRequirements(
    #         n_jobs=api_params['n_jobs'],
    #         cv_folds=composer_params['cv_folds'],
    #         validation_blocks=composer_params['validation_blocks'],
    #     )

    def obtain_model(self) -> Tuple[Pipeline, Sequence[Pipeline], OptHistory]:
        """ Function for composing FEDOT pipeline model """
        # api_params, composer_params, tuning_params = _divide_parameters(self.params._all_parameters)
        task: Task = self.params.get('task')
        train_data = self.params.get('train_data')
        timeout = self.params.get('timeout')
        with_tuning = self.params.get('with_tuning')
        available_operations = self.params.get('available_operations')
        preset = self.params.get('preset')

        self.timer = ApiTime(time_for_automl=timeout, with_tuning=with_tuning)

        # Work with initial assumptions
        assumption_handler = AssumptionsHandler(train_data)

        initial_assumption = assumption_handler.propose_assumptions(self.params.get('initial_assumption'),
                                                                    available_operations,
                                                                    use_input_preprocessing=self.params.get(
                                                                        'use_input_preprocessing'))

        n_jobs = determine_n_jobs(self.params.get('n_jobs'))

        with self.timer.launch_assumption_fit():
            fitted_assumption = \
                assumption_handler.fit_assumption_and_check_correctness(initial_assumption[0],
                                                                        pipelines_cache=self.pipelines_cache,
                                                                        preprocessing_cache=self.preprocessing_cache,
                                                                        eval_n_jobs=n_jobs)

        self.log.message(
            f'Initial pipeline was fitted in {round(self.timer.assumption_fit_spend_time.total_seconds(), 1)} sec.')

        self.params.update(preset=assumption_handler.propose_preset(preset, self.timer, n_jobs=n_jobs))

        composer_requirements = self.params.init_composer_requirements(self.timer.timedelta_composing)

        available_operations = list(chain(composer_requirements.primary,
                                          composer_requirements.secondary))

        self.tuner_requirements = composer_requirements

        metric_functions = self.obtain_metric(task)
        graph_generation_params = \
            self.params.init_graph_generation_params(requirements=composer_requirements)
        self.log.message(f"AutoML configured."
                         f" Parameters tuning: {with_tuning}."
                         f" Time limit: {timeout} min."
                         f" Set of candidate models: {available_operations}.")

        best_pipeline, best_pipeline_candidates, gp_composer = self.compose_pipeline(task, train_data,
                                                                                     fitted_assumption,
                                                                                     metric_functions,
                                                                                     composer_requirements,
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
                         graph_generation_params: GraphGenerationParams,
                         ) -> Tuple[Pipeline, List[Pipeline], GPComposer]:

        multi_objective = len(metric_functions) > 1
        optimizer_params = self.params.init_optimizer_parameters(multi_objective=multi_objective)

        gp_composer: GPComposer = ComposerBuilder(task=task) \
            .with_requirements(composer_requirements) \
            .with_initial_pipelines(fitted_assumption) \
            .with_optimizer(self.params.get('optimizer')) \
            .with_optimizer_params(parameters=optimizer_params,
                                   external_parameters=self.params.get('optimizer_external_params')) \
            .with_metrics(metric_functions) \
            .with_cache(self.pipelines_cache, self.preprocessing_cache) \
            .with_graph_generation_param(graph_generation_params=graph_generation_params) \
            .build()

        self.metric_names = gp_composer.optimizer.objective.metric_names

        n_jobs = determine_n_jobs(composer_requirements.n_jobs)

        if self.timer.have_time_for_composing(self.params.get('pop_size'), n_jobs):
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
            .with_tuner(SimultaneousTuner) \
            .with_metric(metric_function) \
            .with_iterations(DEFAULT_TUNING_ITERATIONS_NUMBER) \
            .with_timeout(datetime.timedelta(minutes=timeout_for_tuning)) \
            .with_eval_time_constraint(composer_requirements.max_graph_fit_time) \
            .with_requirements(composer_requirements) \
            .build(train_data)

        if self.timer.have_time_for_tuning():
            # Tune all nodes in the pipeline
            with self.timer.launch_tuning():
                self.was_tuned = False
                self.log.message(f'Hyperparameters tuning started with {round(timeout_for_tuning)} min. timeout')
                tuned_pipeline = tuner.tune(pipeline_gp_composed)
                self.log.message('Hyperparameters tuning finished')
        else:
            self.log.message(f'Time for pipeline composing was {str(self.timer.composing_spend_time)}.\n'
                             f'The remaining {max(0, round(timeout_for_tuning, 1))} seconds are not enough '
                             f'to tune the hyperparameters.')
            self.log.message('Composed pipeline returned without tuning.')
            tuned_pipeline = pipeline_gp_composed
        self.was_tuned = tuner.was_tuned
        return tuned_pipeline
