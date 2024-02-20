import datetime
import gc
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.tuner_interface import BaseTuner

from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.api.api_utils.params import ApiParams
from fedot.api.time import ApiTime
from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import MetricIDType


class ApiComposer:

    def __init__(self, api_params: ApiParams, metrics: Union[MetricIDType, Sequence[MetricIDType]]):
        self.log = default_log(self)
        self.params = api_params
        self.metrics = metrics
        self.pipelines_cache: Optional[OperationsCache] = None
        self.preprocessing_cache: Optional[PreprocessingCache] = None
        self.timer = None
        # status flag indicating that composer step was applied
        self.was_optimised = False
        # status flag indicating that tuner step was applied`
        self.was_tuned = False
        self.init_cache()

    def init_cache(self):
        use_pipelines_cache = self.params.get('use_pipelines_cache')
        use_preprocessing_cache = self.params.get('use_preprocessing_cache')
        use_input_preprocessing = self.params.get('use_input_preprocessing')
        cache_dir = self.params.get('cache_dir')
        if use_pipelines_cache:
            self.pipelines_cache = OperationsCache(cache_dir)
            #  in case of previously generated singleton cache
            self.pipelines_cache.reset()
        if use_input_preprocessing and use_preprocessing_cache:
            self.preprocessing_cache = PreprocessingCache(cache_dir)
            #  in case of previously generated singleton cache
            self.preprocessing_cache.reset()

    def obtain_model(self, train_data: InputData) -> Tuple[Pipeline, Sequence[Pipeline], OptHistory]:
        """ Function for composing FEDOT pipeline model """
        timeout: float = self.params.timeout
        with_tuning = self.params.get('with_tuning')

        self.timer = ApiTime(time_for_automl=timeout, with_tuning=with_tuning)

        initial_assumption, fitted_assumption = self.propose_and_fit_initial_assumption(train_data)

        multi_objective = len(self.metrics) > 1
        self.params.init_params_for_composing(self.timer.timedelta_composing, multi_objective)

        self.log.message(f"AutoML configured."
                         f" Parameters tuning: {with_tuning}."
                         f" Time limit: {timeout} min."
                         f" Set of candidate models: {self.params.get('available_operations')}.")

        best_pipeline, best_pipeline_candidates, gp_composer = self.compose_pipeline(
            train_data,
            initial_assumption,
            fitted_assumption
        )
        if with_tuning:
            best_pipeline = self.tune_final_pipeline(train_data, best_pipeline, gp_composer.history)

        gc.collect()

        self.log.message('Model generation finished')
        return best_pipeline, best_pipeline_candidates, gp_composer.history

    def propose_and_fit_initial_assumption(self, train_data: InputData) -> Tuple[Sequence[Pipeline], Pipeline]:
        """ Method for obtaining and fitting initial assumption"""
        available_operations = self.params.get('available_operations')

        preset = self.params.get('preset')

        assumption_handler = AssumptionsHandler(train_data)

        initial_assumption = assumption_handler.propose_assumptions(self.params.get('initial_assumption'),
                                                                    available_operations,
                                                                    use_input_preprocessing=self.params.get(
                                                                        'use_input_preprocessing'))

        with self.timer.launch_assumption_fit():
            fitted_assumption = \
                assumption_handler.fit_assumption_and_check_correctness(deepcopy(initial_assumption[0]),
                                                                        pipelines_cache=self.pipelines_cache,
                                                                        preprocessing_cache=self.preprocessing_cache,
                                                                        eval_n_jobs=self.params.n_jobs)

        self.log.message(
            f'Initial pipeline was fitted in {round(self.timer.assumption_fit_spend_time.total_seconds(), 1)} sec.')

        self.params.update(preset=assumption_handler.propose_preset(preset, self.timer, n_jobs=self.params.n_jobs))

        return initial_assumption, fitted_assumption

    def compose_pipeline(self, train_data: InputData, initial_assumption: Sequence[Pipeline],
                         fitted_assumption: Pipeline) -> Tuple[Pipeline, List[Pipeline], GPComposer]:

        gp_composer: GPComposer = (ComposerBuilder(task=self.params.task)
                                   .with_requirements(self.params.composer_requirements)
                                   .with_initial_pipelines(initial_assumption)
                                   .with_optimizer(self.params.get('optimizer'))
                                   .with_optimizer_params(parameters=self.params.optimizer_params)
                                   .with_metrics(self.metrics)
                                   .with_cache(self.pipelines_cache, self.preprocessing_cache)
                                   .with_graph_generation_param(self.params.graph_generation_params)
                                   .build())

        if self.timer.have_time_for_composing(self.params.get('pop_size'), self.params.n_jobs):
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

    def tune_final_pipeline(self, train_data: InputData, pipeline_gp_composed: Pipeline,
                            history: Optional[OptHistory]) -> Tuple[BaseTuner, Pipeline]:
        """ Launch tuning procedure for obtained pipeline by composer """
        timeout_for_tuning = abs(self.timer.determine_resources_for_tuning()) / 60
        tuner = (TunerBuilder(self.params.task)
                 .with_tuner(SimultaneousTuner)
                 .with_metric(self.metrics[0])
                 .with_iterations(DEFAULT_TUNING_ITERATIONS_NUMBER)
                 .with_timeout(datetime.timedelta(minutes=timeout_for_tuning))
                 .with_eval_time_constraint(self.params.composer_requirements.max_graph_fit_time)
                 .with_requirements(self.params.composer_requirements)
                 .with_history(history)
                 .build(train_data))

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
