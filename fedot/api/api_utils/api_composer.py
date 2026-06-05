import datetime
import gc
import time
from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Union

from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot.api.api_utils.api_composer_rules import build_cache_init_plan, build_tuner_plan
from fedot.api.api_utils.api_run_planner import FinalFitAction, build_composer_execution_plan, plan_final_fit
from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.predefined_model import PredefinedModel
from fedot.api.time import ApiTime
from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.input_data.data import InputData, InputDataList
from fedot.core.optimisers.objective.data_source_context import (
    ComposerDataSourceContext,
    build_external_holdout_composer_data_source_context,
    build_internal_composer_data_source_context,
)
from fedot.core.pipelines.ensembling.config import ChunkedEnsembleConfig
from fedot.core.pipelines.ensembling.pipeline_ensemble import PipelineEnsemble
from fedot.core.pipelines.ensembling.utils import (
    calculate_validation_metrics,
    ensure_all_classes_in_chunk,
)
from fedot.core.pipelines.ensembling.routing import SamplingRoutingContext
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import MetricIDType, metric_name
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.composer_timer import fedot_composer_timer


class ApiComposer:

    def __init__(self, api_params: ApiParams, metrics: Union[MetricIDType, Sequence[MetricIDType]]):
        self.log = default_log(self)
        self.params = api_params
        self.metrics = metrics
        self.operations_cache: Optional[OperationsCache] = None
        self.preprocessing_cache: Optional[PreprocessingCache] = None
        self.predictions_cache: Optional[PredictionsCache] = None
        self.timer = None
        # status flag indicating that composer step was applied
        self.was_optimised = False
        # status flag indicating that tuner step was applied`
        self.was_tuned = False
        self.init_cache()

    def init_cache(self):
        cache_plan = build_cache_init_plan(
            use_operations_cache=self.params.get('use_operations_cache'),
            use_preprocessing_cache=self.params.get('use_preprocessing_cache'),
            use_predictions_cache=self.params.get('use_predictions_cache'),
            use_input_preprocessing=self.params.get('use_input_preprocessing'),
            cache_dir=self.params.get('cache_dir'),
            use_stats=self.params.get('use_stats'),
        )

        if cache_plan.use_operations_cache:
            self.operations_cache = OperationsCache(
                cache_dir=cache_plan.cache_dir, use_stats=cache_plan.use_stats)
            self.operations_cache.reset()
        if cache_plan.use_preprocessing_cache:
            self.preprocessing_cache = PreprocessingCache(
                cache_dir=cache_plan.cache_dir, use_stats=cache_plan.use_stats)
            self.preprocessing_cache.reset()
        if cache_plan.use_predictions_cache:
            self.predictions_cache = PredictionsCache(
                cache_dir=cache_plan.cache_dir, use_stats=cache_plan.use_stats)
            self.predictions_cache.reset()

    def obtain_model(self, train_data: InputData) -> Tuple[Pipeline, Sequence[Pipeline], OptHistory]:
        return self._obtain_model(
            train_data=train_data,
            context_builder=build_internal_composer_data_source_context,
        )

    def obtain_model_with_external_validation(self,
                                              train_data: InputData,
                                              validation_data: InputData) -> Tuple[
                                                  Pipeline, Sequence[Pipeline], OptHistory
                                              ]:
        return self._obtain_model(
            train_data=train_data,
            context_builder=lambda data, _: build_external_holdout_composer_data_source_context(
                train_data=data,
                validation_data=validation_data,
            ),
        )

    def _obtain_model(self,
                      train_data: InputData,
                      context_builder: Callable[
                          [InputData, Optional[int]],
                          ComposerDataSourceContext,
                      ]) -> Tuple[Pipeline, Sequence[Pipeline], OptHistory]:
        """ Function for composing FEDOT pipeline model """

        with fedot_composer_timer.launch_composing():
            timeout: float = self.params.timeout
            with_tuning = self.params.get('with_tuning')

            self.timer = ApiTime(time_for_automl=timeout,
                                 with_tuning=with_tuning)

            initial_assumption, fitted_assumption = self.propose_and_fit_initial_assumption(
                train_data)

            multi_objective = len(self.metrics) > 1
            self.params.init_params_for_composing(self.timer.timedelta_composing, multi_objective)
            data_source_context = context_builder(train_data, self.params.get('cv_folds'))

            self.log.message(f"AutoML configured."
                             f" Parameters tuning: {with_tuning}."
                             f" Time limit: {timeout} min."
                             f" Set of candidate models: {self.params.get('available_operations')}.")

            best_pipeline, best_pipeline_candidates, gp_composer = self.compose_pipeline(
                train_data,
                initial_assumption,
                fitted_assumption,
                data_source_context,
            )

        timeout_for_tuning = abs(
            self.timer.determine_resources_for_tuning()) / 60
        execution_plan = build_composer_execution_plan(
            with_tuning=with_tuning,
            have_time_for_composing=self.was_optimised,
            have_time_for_tuning=self.timer.have_time_for_tuning(),
            tuning_timeout_minutes=timeout_for_tuning,
        )

        if execution_plan.should_tune:
            with fedot_composer_timer.launch_tuning('composing'):
                best_pipeline = self.tune_final_pipeline(
                    train_data, best_pipeline, execution_plan)
        elif with_tuning:
            self.log.message(
                f'Time for pipeline composing was {str(self.timer.composing_spend_time)}.\n'
                f'The remaining {max(0, round(execution_plan.tuning_timeout_minutes, 1))} seconds are not enough '
                f'to tune the hyperparameters.')
            self.log.message('Composed pipeline returned without tuning.')
            self.was_tuned = False

        if gp_composer.history:
            adapter = self.params.graph_generation_params.adapter
            gp_composer.history.tuning_result = adapter.adapt(best_pipeline)
        gc.collect()

        self.log.message('Model generation finished')
        return best_pipeline, best_pipeline_candidates, gp_composer.history

    def obtain_ensemble_model(self,
                              train_data_list: InputDataList,
                              predefined_model: Optional[Union[str, Pipeline]] = None,
                              validation_data: Optional[InputData] = None,
                              class_representatives: Optional[dict] = None,
                              api_preprocessor=None,
                              chunked_ensemble_config: Optional[ChunkedEnsembleConfig] = None,
                              routing_context: Optional[SamplingRoutingContext] = None) -> \
            Tuple[PipelineEnsemble, Sequence[Sequence[Pipeline]], List[OptHistory]]:
        if not train_data_list:
            raise ValueError('InputDataList for ensemble model must not be empty.')
        if predefined_model is None and validation_data is None:
            raise ValueError('Chunked ensemble composition requires common validation data.')

        pipelines: List[Pipeline] = []
        pipeline_infos = []
        best_models: List[Sequence[Pipeline]] = []
        histories: List[OptHistory] = []
        chunk_failures = []
        initial_timeout = self.params.timeout
        started_at = time.perf_counter()

        task_type = train_data_list[0].task.task_type
        validation_metric = metric_name(self.metrics[0])
        chunked_ensemble_config = chunked_ensemble_config or ChunkedEnsembleConfig()

        for chunk_idx, chunk_data in enumerate(train_data_list):
            current_chunk = chunk_data
            if class_representatives and task_type == TaskTypesEnum.classification:
                current_chunk = ensure_all_classes_in_chunk(chunk_data, class_representatives)

            if initial_timeout is not None:
                elapsed_minutes = (time.perf_counter() - started_at) / 60.0
                remaining_minutes = max(0.0, initial_timeout - elapsed_minutes)
                remaining_chunks = max(1, len(train_data_list) - len(pipelines))
                self.params.timeout = remaining_minutes / remaining_chunks

            try:
                if predefined_model is not None:
                    chunk_predefined_model = (
                        deepcopy(predefined_model)
                        if isinstance(predefined_model, Pipeline)
                        else predefined_model
                    )
                    pipeline = PredefinedModel(
                        chunk_predefined_model,
                        current_chunk,
                        self.log,
                        use_input_preprocessing=self.params.get('use_input_preprocessing'),
                        api_preprocessor=deepcopy(api_preprocessor) if api_preprocessor is not None else None,
                    ).fit()
                    best_pipeline_candidates = [pipeline]
                    history = None
                else:
                    pipeline, best_pipeline_candidates, history = self.obtain_model_with_external_validation(
                        train_data=current_chunk,
                        validation_data=validation_data,
                    )
            except Exception as ex:
                error_message = (
                    'Failed to fit predefined model'
                    if predefined_model is not None
                    else 'Failed to build chunk pipeline'
                )
                self.log.message(f'{error_message} #{chunk_idx}: {ex}')
                chunk_failures.append({
                    'chunk_idx': chunk_idx,
                    'status': 'failed',
                    'reason': 'exception',
                    'message': str(ex),
                })
                continue
            if pipeline is None:
                self.log.message(f'No models were found for chunk #{chunk_idx}. Chunk is skipped.')
                chunk_failures.append({
                    'chunk_idx': chunk_idx,
                    'status': 'failed',
                    'reason': 'no_models_found',
                    'message': 'No models were found for chunk.',
                })
                continue
            self._fit_chunk_pipeline_for_ensemble(pipeline, current_chunk, history)
            if not pipeline.is_fitted:
                self.log.message(f'Chunk pipeline #{chunk_idx} was not fitted after final fit stage. Chunk is skipped.')
                chunk_failures.append({
                    'chunk_idx': chunk_idx,
                    'status': 'failed',
                    'reason': 'pipeline_not_fitted',
                    'message': 'Chunk pipeline was not fitted after final fit stage.',
                })
                continue

            pipelines.append(pipeline)
            best_models.append(best_pipeline_candidates)
            if history is not None:
                histories.append(history)

            model_metrics = {}
            val_predictions = None
            val_probabilities = None

            if validation_data is not None:
                try:
                    model_output_mode = 'labels' if task_type == TaskTypesEnum.classification else 'default'
                    model_output = pipeline.predict(validation_data, output_mode=model_output_mode)
                    val_predictions = model_output.predict
                    if task_type == TaskTypesEnum.classification:
                        val_probabilities = pipeline.predict(validation_data, output_mode='probs').predict
                    model_metrics = calculate_validation_metrics(
                        y_true=validation_data.target,
                        y_labels=val_predictions,
                        y_proba=val_probabilities,
                        task_type=task_type,
                    )
                    self.log.message(f'Chunk #{chunk_idx} model metrics: {model_metrics}')
                except Exception as ex:
                    self.log.message(f'Validation metrics are unavailable for chunk #{chunk_idx}: {ex}')
                    model_metrics = {}
                    val_predictions = None
                    val_probabilities = None

            pipeline_infos.append(
                {
                    'name': self._resolve_chunk_name(chunk_idx, routing_context),
                    'source_chunk_idx': chunk_idx,
                    'pipeline': pipeline,
                    'data_size': int(len(current_chunk.idx)),
                    'metrics': model_metrics,
                    'val_predictions': val_predictions,
                    'val_probabilities': val_probabilities,
                }
            )

        self.params.timeout = initial_timeout
        if len(pipelines) < chunked_ensemble_config.min_successful_chunks:
            raise ValueError(
                f'Chunked ensemble requires at least {chunked_ensemble_config.min_successful_chunks} '
                f'successful chunks, '
                f'but got {len(pipelines)}. Failure report: {chunk_failures}'
            )

        ensemble = PipelineEnsemble(
            pipelines=pipelines,
            validation_metric=validation_metric,
            ensemble_method=chunked_ensemble_config.ensemble_method.value,
            pipeline_infos=pipeline_infos,
            routing_context=routing_context,
            ensemble_params=chunked_ensemble_config.ensemble_params,
            batch_size=chunked_ensemble_config.batch_size,
        )

        return ensemble, best_models, histories

    @staticmethod
    def _resolve_chunk_name(chunk_idx: int,
                            routing_context: Optional[SamplingRoutingContext]) -> str:
        if routing_context is not None and chunk_idx < len(routing_context.partition_names):
            return str(routing_context.partition_names[chunk_idx])
        return f'chunk_{chunk_idx}'

    def _fit_chunk_pipeline_for_ensemble(self,
                                         pipeline: Pipeline,
                                         chunk_data: InputData,
                                         history: Optional[OptHistory]) -> Pipeline:
        final_fit_plan = plan_final_fit(
            history=history,
            pipeline_is_fitted=pipeline.is_fitted,
            is_pipeline_ensemble=False,
        )
        if final_fit_plan.action is FinalFitAction.fit_pipeline_on_full_data:
            pipeline.fit(chunk_data, n_jobs=self.params.n_jobs)
        return pipeline

    def propose_and_fit_initial_assumption(self, train_data: InputData) -> Tuple[Sequence[Pipeline], Pipeline]:
        """ Method for obtaining and fitting initial assumption"""
        available_operations = self.params.get('available_operations')

        preset = self.params.get('preset')

        assumption_handler = AssumptionsHandler(train_data)

        initial_assumption = assumption_handler.propose_assumptions(self.params.get('initial_assumption'),
                                                                    available_operations,
                                                                    use_input_preprocessing=self.params.get(
                                                                        'use_input_preprocessing'))

        with self.timer.launch_assumption_fit(n_folds=self.params.data['cv_folds']):
            fitted_assumption = \
                assumption_handler.fit_assumption_and_check_correctness(deepcopy(initial_assumption[0]),
                                                                        operations_cache=self.operations_cache,
                                                                        preprocessing_cache=self.preprocessing_cache,
                                                                        eval_n_jobs=self.params.n_jobs)

        self.log.message(
            f'Initial pipeline was fitted in '
            f'{round(self.timer.assumption_fit_spend_time_single_fold.total_seconds(), 1)} sec.')

        self.log.message(
            f'Taking into account n_folds={self.params.data["cv_folds"]}, estimated fit time for initial assumption '
            f'is {round(self.timer.assumption_fit_spend_time.total_seconds(), 1)} sec.')

        self.params.update(preset=assumption_handler.propose_preset(
            preset, self.timer, n_jobs=self.params.n_jobs))

        return initial_assumption, fitted_assumption

    def compose_pipeline(self,
                         train_data: InputData,
                         initial_assumption: Sequence[Pipeline],
                         fitted_assumption: Pipeline,
                         data_source_context: ComposerDataSourceContext) -> Tuple[Pipeline, List[Pipeline], GPComposer]:

        gp_composer: GPComposer = (ComposerBuilder(task=self.params.task)
                                   .with_requirements(self.params.composer_requirements)
                                   .with_initial_pipelines(initial_assumption)
                                   .with_optimizer(self.params.get('optimizer'))
                                   .with_optimizer_params(parameters=self.params.optimizer_params)
                                   .with_metrics(self.metrics)
                                   .with_cache(self.operations_cache, self.preprocessing_cache, self.predictions_cache)
                                   .with_graph_generation_param(self.params.graph_generation_params)
                                   .build())

        have_time_for_composing = self.timer.have_time_for_composing(
            self.params.get('pop_size'), self.params.n_jobs)
        execution_plan = build_composer_execution_plan(
            with_tuning=self.params.get('with_tuning'),
            have_time_for_composing=have_time_for_composing,
            have_time_for_tuning=False,
            tuning_timeout_minutes=0,
        )

        if execution_plan.should_compose:
            with self.timer.launch_composing():
                self.log.message('Pipeline composition started.')
                self.was_optimised = False
                best_pipelines = gp_composer.compose_pipeline(
                    data=train_data,
                    data_source_context=data_source_context,
                )
                best_pipeline_candidates = gp_composer.best_models
                self.was_optimised = True
        else:
            self.log.message(f'Timeout is too small for composing and is skipped '
                             f'because fit_time is {self.timer.assumption_fit_spend_time.total_seconds()} sec.')
            best_pipelines = fitted_assumption
            best_pipeline_candidates = [fitted_assumption]
            self.was_optimised = False

        for pipeline in best_pipeline_candidates:
            pipeline.log = self.log
        best_pipeline = best_pipelines[0] if isinstance(
            best_pipelines, Sequence) else best_pipelines
        return best_pipeline, best_pipeline_candidates, gp_composer

    def tune_final_pipeline(self, train_data: InputData,
                            pipeline_gp_composed: Pipeline,
                            execution_plan=None) -> Pipeline:
        """ Launch tuning procedure for obtained pipeline by composer """
        timeout_for_tuning = execution_plan.tuning_timeout_minutes if execution_plan else abs(
            self.timer.determine_resources_for_tuning()) / 60
        tuner_plan = build_tuner_plan(
            metrics=self.metrics,
            timeout_minutes=timeout_for_tuning,
            iterations=DEFAULT_TUNING_ITERATIONS_NUMBER,
        )
        tuner = (TunerBuilder(self.params.task)
                 .with_tuner(SimultaneousTuner)
                 .with_metric(tuner_plan.metric)
                 .with_iterations(tuner_plan.iterations)
                 .with_timeout(datetime.timedelta(minutes=tuner_plan.timeout_minutes))
                 .with_eval_time_constraint(self.params.composer_requirements.max_graph_fit_time)
                 .with_cv_folds(self.params.get('cv_folds'))
                 .with_n_jobs(self.params.n_jobs)
                 .build(train_data))

        with self.timer.launch_tuning():
            self.was_tuned = False
            self.log.message(
                f'Hyperparameters tuning started with {round(tuner_plan.timeout_minutes)} min. timeout')
            tuned_pipeline = tuner.tune(pipeline_gp_composed)
            self.log.message('Hyperparameters tuning finished')
        self.was_tuned = tuner.was_tuned
        return tuned_pipeline
