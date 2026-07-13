import logging
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from golem.core.dag.graph_utils import graph_structure
from golem.core.log import Log, default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.utilities.data_structures import ensure_wrapped_in_sequence
from golem.visualisation.opt_viz_extra import visualise_pareto

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_run_planner import (
    FinalFitAction,
    ChunkedEnsemblePlan,
    SamplingStagePlan,
    plan_chunked_ensemble,
    plan_final_fit,
    plan_sampling_stage,
)
from fedot.api.api_utils.api_service_rules import (
    build_explain_plan,
    build_forecast_plan,
    build_metrics_plan,
    build_metrics_validation_plan,
    build_predict_plan,
    build_predict_proba_plan,
    build_tune_execution_plan,
    resolve_forecast_horizon,
    resolve_predict_proba_mode,
)
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.data_definition import FeaturesType, TargetType
from fedot.api.api_utils.input_analyser import InputAnalyser
from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.predefined_model import PredefinedModel
from fedot.api.sampling_stage.config import SamplingChunkingConfig
from fedot.api.sampling_stage.executor import SamplingStageExecutor
from fedot.core.constants import DEFAULT_API_TIMEOUT_MINUTES, DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.input_data.data import InputData, InputDataList, OutputData, PathType
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.data.visualisation import plot_biplot, plot_forecast, plot_roc_auc
from fedot.core.optimisers.objective import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.ensembling.config import ChunkedEnsembleConfig
from fedot.core.pipelines.ensembling.pipeline_ensemble import PipelineEnsemble
from fedot.core.pipelines.ensembling.routing import SamplingRoutingContext
from fedot.core.pipelines.ensembling.utils import prepare_chunked_ensemble_validation
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import convert_forecast_to_output, out_of_sample_ts_forecast
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import MetricCallable
from fedot.core.repository.tasks import TaskParams, TaskTypesEnum
from fedot.core.utils import set_random_seed
from fedot.explainability.explainer_template import Explainer
from fedot.explainability.explainers import explain_pipeline
from fedot.preprocessing.base_preprocessing import BasePreprocessor
from fedot.remote.remote_evaluator import RemoteEvaluator
from fedot.utilities.composer_timer import fedot_composer_timer
from fedot.utilities.define_metric_by_task import MetricByTask
from fedot.utilities.memory import MemoryAnalytics
from fedot.utilities.project_import_export import export_project_to_zip, import_project_from_zip
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.service.tensor_optional_runtime import (
    get_optional_runtime_spec_for_tensor_data,
)


NOT_FITTED_ERR_MSG = 'Model not fitted yet'


@dataclass(frozen=True)
class FitDataContext:
    """Planning artefacts prepared before fitting starts and reused across fit stages."""

    recommendations_for_data: Optional[dict]
    sampling_stage_plan: SamplingStagePlan
    chunked_ensemble_plan: ChunkedEnsemblePlan
    ensemble_validation_data: Optional[InputData]
    class_representatives: Optional[dict]


class Fedot:
    """ The main class for FEDOT AutoML API.

    Alternatively, may be initialized using the class :class:`~fedot.api.builder.FedotBuilder`,
    where all the optional AutoML parameters are documented and separated by meaning.

    Args:
        problem: name of the modelling problem to solve.
            .. details:: Possible options:

                - ``classification`` -> for classification task
                - ``regression`` -> for regression task
                - ``ts_forecasting`` -> for time series forecasting task

        timeout: time for model design (in minutes): ``None`` or ``-1`` means infinite time.
        task_params: additional parameters of the task.
        seed: value for a fixed random seed.
        logging_level: logging levels are the same as in
            `built-in logging library <https://docs.python.org/3/library/logging.html>`_.

            .. details:: Possible options:

                - ``50`` -> critical
                - ``40`` -> error
                - ``30`` -> warning
                - ``20`` -> info
                - ``10`` -> debug
                - ``0`` -> nonset

        safe_mode: if set ``True`` it will cut large datasets to prevent memory overflow and use label encoder
            instead of OneHot encoder if summary cardinality of categorical features is high.
            Default value is ``False``.

        n_jobs: num of ``n_jobs`` for parallelization (set to ``-1`` to use all cpu's). Defaults to ``-1``.

        composer_tuner_params: Additional optional parameters. See their documentation at the methods of
            :class:`~fedot.api.builder.FedotBuilder`.

            ``tensor_data_config`` is a dictionary of options for
            :class:`~fedot.core.data.tensor_data.tensor_data_creator.TensorDataCreator`
            (for example ``backend_name``, ``use_cache``, ``encoding_strategy``,
            ``custom_strategy``, ``data_type``, ``ts_orientation``). It is validated
            during initialization and stored on :attr:`~fedot.api.api_utils.params.ApiParams.tensor_data_config`.
    """

    def __init__(self,
                 problem: str,
                 timeout: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES,
                 task_params: TaskParams = None,
                 seed: Optional[int] = None,
                 logging_level: int = logging.ERROR,
                 safe_mode: bool = False,
                 n_jobs: int = -1,
                 use_cache: bool = True,
                 **composer_tuner_params
                 ):

        set_random_seed(seed)
        self.log = self._init_logger(logging_level)
        self.use_cache = use_cache

        # Attributes for dealing with metrics, data sources and hyperparameters
        self.params = ApiParams(composer_tuner_params, problem, task_params, n_jobs, timeout, seed)

        default_metrics = MetricByTask.get_default_quality_metrics(self.params.task.task_type)
        passed_metrics = self.params.get('metric')
        self.metrics = ensure_wrapped_in_sequence(passed_metrics) if passed_metrics else default_metrics

        self.api_composer = ApiComposer(self.params, self.metrics)

        # Initialize data processors for data preprocessing and preliminary data analysis
        self.data_processor = ApiDataProcessor(task=self.params.task,
                                               use_input_preprocessing=self.params.get('use_input_preprocessing'))
        self.data_analyser = InputAnalyser(safe_mode=safe_mode)

        self.target: Optional[TargetType] = None
        self.prediction: Optional[OutputData] = None
        self._is_in_sample_prediction = True
        self.train_data: Optional[Union[InputData, InputDataList]] = None
        self.test_data: Optional[InputData] = None

        # Outputs
        self.current_pipeline: Optional[Union[Pipeline, PipelineEnsemble]] = None
        self.best_models: Sequence[Union[Pipeline, Sequence[Pipeline]]] = ()
        self.history: Optional[Union[OptHistory, Sequence[OptHistory]]] = None
        self.sampling_stage_metadata: Optional[dict] = None
        self.sampling_routing_context: Optional[SamplingRoutingContext] = None

        fedot_composer_timer.reset_timer()

    def _init_remote_if_necessary(self, train_data: Union[InputData, InputDataList]):
        remote = RemoteEvaluator()
        if remote.is_enabled and remote.remote_task_params is not None:
            task = self.params.task
            if task.task_type is TaskTypesEnum.ts_forecasting:
                task_str = (f'Task(TaskTypesEnum.ts_forecasting, '
                            f'TsForecastingParams(forecast_length={task.task_params.forecast_length}))')
            else:
                task_str = f'Task({str(task.task_type)})'
            remote.remote_task_params.task_type = task_str
            remote.remote_task_params.is_multi_modal = isinstance(train_data, MultiModalData)

    # TODO @artemlunev: should be refactored for TD
    def _apply_sampling_stage(self,
                              fit_context: FitDataContext,
                              train_data: Union[InputData, InputDataList]) -> Tuple[
                                  FitDataContext, Union[InputData, InputDataList]]:
        if fit_context.sampling_stage_plan.skip_metadata is not None:
            self.sampling_stage_metadata = fit_context.sampling_stage_plan.skip_metadata
            self.log.message('Composition for AtomizedModel currently unavailable')
            return fit_context, train_data

        if not fit_context.sampling_stage_plan.should_run_sampling_stage:
            return fit_context, train_data

        sampling_config = self.params.get('sampling_config')
        if sampling_config is None:
            return fit_context, train_data

        if not isinstance(train_data, InputData):
            raise ValueError('Sampling stage supports only InputData in V1.')

        self.log.message('Sampling stage started')
        executor = SamplingStageExecutor(
            sampling_config=sampling_config,
            task_type=self.params.task.task_type,
            total_timeout_minutes=self.params.timeout,
            log=self.log,
        )
        self._log_applied_config(
            config=executor.config,
            label='sampling',
        )
        stage_result = executor.execute(train_data)
        self.sampling_stage_metadata = stage_result.metadata
        # Routed/gated ensemble modes reuse the strategy predictor fitted by sampling_zoo.
        self.sampling_routing_context = stage_result.routing_context

        if self.params.timeout is not None:
            self.params.timeout = stage_result.updated_timeout_minutes

        self.log.message(
            f'Sampling stage finished. Rows: {stage_result.metadata["rows_before"]} -> '
            f'{stage_result.metadata["rows_after"]}. '
            f'Updated timeout: {self.params.timeout} min.'
        )
        return fit_context, stage_result.train_data

    def _log_applied_config(self, config: Any, label: str):
        config_payload = {
            field.name: (
                getattr(config, field.name).value
                if isinstance(getattr(config, field.name), Enum)
                else getattr(config, field.name)
            )
            for field in fields(config)
        }
        self.log.info(f'Applied {label} config: {config_payload}')

    def _finalize_fit_model_if_required(self,
                                        fit_context: FitDataContext,
                                        predefined_model: Union[str, Pipeline, None]):
        if predefined_model is not None and not isinstance(self.current_pipeline, PipelineEnsemble):
            return

        full_train_not_preprocessed = deepcopy(self.train_data)
        with fedot_composer_timer.launch_train_inference():
            final_fit_plan = plan_final_fit(
                history=self.history,
                pipeline_is_fitted=self.current_pipeline.is_fitted,
                is_pipeline_ensemble=isinstance(self.current_pipeline, PipelineEnsemble),
            )
            if final_fit_plan.action is FinalFitAction.fit_pipeline_on_full_data:
                self._fit_final_pipeline(
                    fit_context.recommendations_for_data,
                    full_train_not_preprocessed,
                )
                self.log.message('Final pipeline was fitted')
            elif final_fit_plan.action is FinalFitAction.finalize_ensemble:
                self._finalize_pipeline_ensemble(validation_data=fit_context.ensemble_validation_data)
                self.log.message('Pipeline ensemble was finalized')
            else:
                self.log.message('Already fitted initial pipeline is used')

    def _fit_final_pipeline(self,
                            recommendations: Optional[dict],
                            full_train_not_preprocessed: Union[InputData, MultiModalData]):
        if recommendations is not None:
            # if data was cut we need to refit pipeline on full data
            self.data_processor.accept_and_apply_recommendations(full_train_not_preprocessed,
                                                                 {k: v for k, v in recommendations.items()
                                                                  if k != 'cut'})
        self.current_pipeline.fit(
            full_train_not_preprocessed,
            n_jobs=self.params.n_jobs
        )

    def _finalize_pipeline_ensemble(self, validation_data: Optional[InputData] = None):
        self.current_pipeline.finalize(validation_data=validation_data)

    def _merge_current_pipeline_preprocessors(self):
        if isinstance(self.current_pipeline, PipelineEnsemble):
            for pipeline in self.current_pipeline.pipelines:
                pipeline.preprocessor = BasePreprocessor.merge_preprocessors(
                    api_preprocessor=deepcopy(self.data_processor.preprocessor),
                    pipeline_preprocessor=pipeline.preprocessor,
                    use_auto_preprocessing=self.params.get('use_auto_preprocessing')
                )
            return

        self.current_pipeline.preprocessor = BasePreprocessor.merge_preprocessors(
            api_preprocessor=self.data_processor.preprocessor,
            pipeline_preprocessor=self.current_pipeline.preprocessor,
            use_auto_preprocessing=self.params.get('use_auto_preprocessing')
        )


    def fit_transform_tensor_optional(
        self,
        tensor_data: TensorData
    ) -> TensorData:
    # TODO romankuklo: if no imputation in user strategy and no auto preprocessing, 
    # we should show it for pipeline models, but all preprocesing only through services.

        use_auto_preprocessing = self.params.get('use_auto_preprocessing')
        user_optional_strategy = self.params.tensor_data_config.get('optional_strategy')

        if not use_auto_preprocessing and user_optional_strategy is None:
            return tensor_data

        runtime_spec = get_optional_runtime_spec_for_tensor_data(tensor_data)

        if use_auto_preprocessing:
            optional_strategy = runtime_spec.default_steps
        else:
            optional_strategy = user_optional_strategy

        service = runtime_spec.service_cls(use_cache=self.use_cache)

        return service.fit_transform(tensor_data, optional_strategy)


    def _prepare_fit_context(self) -> FitDataContext:

        with fedot_composer_timer.launch_data_definition('fit'):

            self.target = self.train_data.target
            self.params.update_available_operations_by_preset(self.train_data)

            recommendations_for_data = None
            if self.params.get('use_input_preprocessing'):
                _, recommendations_for_params = self.data_analyser.give_recommendations(
                    input_data=self.train_data,
                    input_params=self.params,
                )
                self.params.accept_and_apply_recommendations(
                    input_data=self.train_data,
                    recommendations=recommendations_for_params,
                )

            self._init_remote_if_necessary(self.train_data)

            with fedot_composer_timer.launch_preprocessing():
                self.train_data = self.fit_transform_tensor_optional(self.train_data)

            # TODO romankuklo: add sampling stage and chunked ensemble for TD

            sampling_stage_plan = None
            chunked_ensemble_plan = ChunkedEnsemblePlan(
                should_use_chunked_ensemble=False,
                config=None,
                train_split_ratio=1.0,
                should_select_class_representatives=False,
                validation_split_seed=123,
            )
            ensemble_validation_data = None
            class_representatives = None

            self.data_analyser.warn_if_large_tensor_without_sampling(
                self.train_data,
                sampling_config_present=sampling_stage_plan,
            )
        return FitDataContext(
            recommendations_for_data=recommendations_for_data,
            sampling_stage_plan=sampling_stage_plan,
            chunked_ensemble_plan=chunked_ensemble_plan,
            ensemble_validation_data=ensemble_validation_data,
            class_representatives=class_representatives,
        )

    def _obtain_pipeline(self,
                         fit_context: FitDataContext,
                         predefined_model: Union[str, Pipeline, None]):
        # TODO: add other strategies here
        with fedot_composer_timer.launch_fitting():
            if predefined_model is not None:
                predefined = PredefinedModel(
                    predefined_model,
                    self.train_data,
                    self.log,
                    use_input_preprocessing=False,
                    api_preprocessor=None,
                )
                self.current_pipeline = predefined.fit()
                self.best_models = ()
                self.history = None
            else:
                self.current_pipeline, self.best_models, self.history = self.api_composer.obtain_model(
                    self.train_data,
                )

            if self.current_pipeline is None:
                raise ValueError('No models were found')

    def fit(self,
            tensor_data: TensorData,
            predefined_model: Union[str, Pipeline] = None) -> Pipeline:

        MemoryAnalytics.start()

        self.train_data = tensor_data
        initial_timeout = self.params.timeout

        try:
            fit_context = self._prepare_fit_context()
            # TODO romankuklo: apply sampling stage and chunked ensemble
            # fit_context, train_data = self._apply_sampling_stage(fit_context, train_data)
            # self.train_data = train_data
            self._obtain_pipeline(
                fit_context=fit_context,
                predefined_model=predefined_model,
            )
            self._finalize_fit_model_if_required(
                fit_context=fit_context,
                predefined_model=predefined_model,
            )

            if isinstance(self.current_pipeline, Pipeline):
                self.log.message(f'Final pipeline: {graph_structure(self.current_pipeline)}')
            elif isinstance(self.current_pipeline, PipelineEnsemble):
                self.log.message(f'Final pipeline ensemble: {len(self.current_pipeline.pipelines)} pipelines')

            return self.current_pipeline
        finally:
            self.params.timeout = initial_timeout
            MemoryAnalytics.finish()

    def tune(self,
             tensor_data: Optional[TensorData] = None,
             metric_name: Optional[Union[str, MetricCallable]] = None,
             iterations: int = DEFAULT_TUNING_ITERATIONS_NUMBER,
             timeout: Optional[float] = None,
             cv_folds: Optional[int] = None,
             n_jobs: Optional[int] = None,
             show_progress: bool = False) -> Pipeline:
        """Method for hyperparameters tuning of current pipeline

        Args:
            tensor_data: data for tuning pipeline in one of the supported formats.
            metric_name: name of metric for quality tuning.
            iterations: numbers of tuning iterations.
            timeout: time for tuning (in minutes). If ``None`` or ``-1`` means tuning until max iteration reach.
            cv_folds: number of folds on data for cross-validation.
            n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's).
            show_progress: shows progress of tuning if ``True``.

        Returns:
            :class:`Pipeline` object.
        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)
        if isinstance(self.current_pipeline, PipelineEnsemble):
            self.log.warning('Tuning for pipeline ensembles is not supported yet. Existing ensemble is returned.')
            return self.current_pipeline
        
        # TODO romankuklo: add optional preprocessing

        with fedot_composer_timer.launch_tuning('post'):
            tune_plan = build_tune_execution_plan(
                tensor_data=tensor_data,
                train_data=self.train_data,
                requested_cv_folds=cv_folds,
                default_cv_folds=self.params.get('cv_folds'),
                requested_n_jobs=n_jobs,
                default_n_jobs=self.params.n_jobs,
                requested_metric=metric_name,
                default_metric=self.metrics[0],
            )

            if tensor_data is None:
                raise ValueError('Tensor data is required for tuning')

            pipeline_tuner = (TunerBuilder(self.params.task)
                              .with_tuner(SimultaneousTuner)
                              .with_cv_folds(tune_plan.cv_folds)
                              .with_n_jobs(tune_plan.n_jobs)
                              .with_metric(tune_plan.metric)
                              .with_iterations(iterations)
                              .with_timeout(timeout)
                              .build(tune_plan.tensor_data))

            self.current_pipeline = pipeline_tuner.tune(self.current_pipeline, show_progress=show_progress)
            self.api_composer.was_tuned = pipeline_tuner.was_tuned

            # Tuner returns a not fitted pipeline, and it is required to fit on train dataset
            self.current_pipeline.fit(self.train_data)

        return self.current_pipeline
    
    def predict(
        self,
        tensor_data: TensorData,
        in_sample: bool = True,
        validation_blocks: Optional[int] = None,
        path_to_save: Optional[PathType] = None
    ) -> TensorData:
        """Runs prediction on a prepared :class:`TensorData` instance.

        Args:
            tensor_data: test data already converted to ``TensorData`` (e.g. via
                :class:`~fedot.core.data.tensor_data.tensor_data_creator.TensorDataCreator`).
            output_mode: prediction format for classification models.
            path_to_save: if specified, path to save prediction to.
        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        self.test_data = tensor_data
        # TODO @romankuklo: add optional preprocessing
        with fedot_composer_timer.launch_predicting():
            self.prediction = self.data_processor.define_predictions(
                current_pipeline=self.current_pipeline,
                test_data=tensor_data,
                in_sample=in_sample,
                validation_blocks=validation_blocks,
            )

        if path_to_save is not None:
            self.save_predict(self.prediction, path_to_save)
        return self.prediction

    # TODO @romankuklo: refactor for TensorData
    def predict_proba(self,
                      features: FeaturesType,
                      probs_for_all_classes: bool = False,
                      path_to_save: Optional[PathType] = None) -> np.ndarray:
        """Predicts the probability of new target using already fitted classification model

        Args:
            features: an array with features of test data.
            probs_for_all_classes: if ``True`` - return probability for each class even for binary classification.
            path_to_save: if specified, path to save prediction to.

        Returns:
            An array with prediction values.
        """

        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        with fedot_composer_timer.launch_predicting():
            if self.params.task.task_type == TaskTypesEnum.classification:
                self.test_data = self.data_processor.define_data(target=self.target,
                                                                 features=features, is_predict=True)

                mode = resolve_predict_proba_mode(probs_for_all_classes)

                self.prediction = self.current_pipeline.predict(self.test_data, output_mode=mode)

                if path_to_save is not None:
                    self.save_predict(self.prediction, path_to_save)
            else:
                raise ValueError('Probabilities of predictions are available only for classification')

        return self.prediction.predict

    # TODO @romankuklo: refactor for TensorData
    def forecast(self,
                            tensor_data: TensorData,
                            horizon: Optional[int] = None,
                            path_to_save: Optional[PathType] = None) -> np.ndarray:
        self._check_forecast_applicable()

        forecast_plan = build_forecast_plan(
            requested_horizon=horizon,
            forecast_length=self.train_data.task.task_params.forecast_length,
        )
        self.test_data = self.data_processor.to_input_data(tensor_data)
        if forecast_plan.clear_target:
            self.test_data.target = None
        predict = out_of_sample_ts_forecast(
            self.current_pipeline, self.test_data, forecast_plan.horizon)
        self.prediction = convert_forecast_to_output(self.test_data, predict)
        self._is_in_sample_prediction = False
        if path_to_save is not None:
            self.save_predict(self.prediction, path_to_save)
        return self.prediction.predict

    def _check_forecast_applicable(self):
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.params.task.task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError(
                'Forecasting can be used only for the time series')

    def get_metrics(self,
                               tensor_data,
                               target: Union[np.ndarray, pd.Series] = None,
                               metric_names: Union[str, List[str]] = None,
                               rounding_order: int = 3) -> dict:
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        metrics_plan = build_metrics_plan()
        self.test_data = self.data_processor.to_input_data(tensor_data)
        self.prediction = self.current_pipeline.predict(
            tensor_data,
            output_mode=metrics_plan.output_mode,
        )
        self._is_in_sample_prediction = False
        return self.get_metrics(target=target, metric_names=metric_names, rounding_order=rounding_order)

    # TODO @romankuklo: refactor for TensorData
    def explain(self, tensor_data,
                           method: str = 'surrogate_dt', visualization: bool = True, **kwargs) -> Explainer:
        explain_plan = build_explain_plan(
            method=method, visualization=visualization)
        data = self.data_processor.to_input_data(tensor_data)
        return explain_pipeline(
            pipeline=self.current_pipeline,
            data=data,
            method=explain_plan.method,
            visualization=explain_plan.visualization,
            **kwargs,
        )

    def load(self, path):
        """Loads saved graph from disk

        Args:
            path: path to ``json`` file with model.
        """
        self.current_pipeline = Pipeline(
            use_input_preprocessing=self.params.get('use_input_preprocessing'))
        self.current_pipeline.load(path)
        self.data_processor.preprocessor = self.current_pipeline.preprocessor

    def plot_pareto(self):
        metric_names = [str(metric) for metric in self.metrics]
        # archive_history stores archives of the best models.
        # Each archive is sorted from the best to the worst model,
        # so the best_candidates is sorted too.
        best_candidates = self.history.archive_history[-1]
        visualise_pareto(front=best_candidates,
                         objectives_names=metric_names,
                         show=True)

    def plot_prediction(self, in_sample: Optional[bool] = None, target: Optional[Any] = None):
        """Plots prediction obtained from a graph.

        Args:
            in_sample: if current prediction is in_sample (for time-series forecasting), plots predictions as future
                values.
            target: user-specified name of target variable for :class:`MultiModalData`.
        """
        task = self.params.task

        if self.prediction is not None:
            if task.task_type == TaskTypesEnum.ts_forecasting:
                in_sample = in_sample or self._is_in_sample_prediction
                plot_forecast(self.test_data, self.prediction,
                              in_sample, target)
            elif task.task_type == TaskTypesEnum.regression:
                plot_biplot(self.prediction)
            elif task.task_type == TaskTypesEnum.classification:
                self.predict_proba(self.test_data)
                plot_roc_auc(self.test_data, self.prediction)
            else:
                self.log.error('Not supported yet')
                raise NotImplementedError(
                    f"For task {task} plot prediction is not supported")
        else:
            self.log.error('No prediction to visualize')
            raise ValueError('Prediction from model is empty')
    
    def get_metrics(self,
                    tensor_data: TensorData,
                    metric_names: Union[str, List[str]] = None,
                    in_sample: Optional[bool] = None,
                    validation_blocks: Optional[int] = None,
                    rounding_order: int = 3) -> dict:
        """Gets quality metrics for a fitted graph

        Args:
            target: an array with target values of test data. If ``None``, target specified for fit is used.
            metric_names: names of required metrics.
            in_sample: used for time series forecasting.
                If True prediction will be obtained as ``.predict(..., in_sample=True)``.
            validation_blocks: number of validation blocks for time series in-sample forecast.
            rounding_order: number of decimal places for metrics

        Returns:
            Values of quality metrics.
        """
        metrics_plan = build_metrics_validation_plan(
            is_pipeline_fitted=self.current_pipeline is not None,
            metric_names=metric_names,
            default_metrics=self.metrics,
            requested_in_sample=in_sample,
            default_in_sample=self._is_in_sample_prediction,
            validation_blocks=validation_blocks,
            rounding_order=rounding_order,
        )

        if tensor_data.target is not None:
            if self.test_data is None:
                self.test_data = tensor_data
                self.test_data.target = tensor_data.target[:len(self.prediction.predict)]
            else:
                self.test_data.target = tensor_data.target[:len(self.prediction.predict)]

        objective = MetricsObjective(metrics_plan.metrics)
        obj_eval = PipelineObjectiveEvaluateWithTensorData(objective=objective,
                                                          data_producer=lambda: (
                                                              yield self.train_data, self.test_data),
                                                          validation_blocks=metrics_plan.validation_blocks,
                                                          eval_n_jobs=self.params.n_jobs,
                                                          do_unfit=False)

        metrics = obj_eval.evaluate(self.current_pipeline).values
        metrics = {metric_name: round(abs(metric), metrics_plan.rounding_order) for (metric_name, metric) in
                   zip(metrics_plan.metric_names, metrics)}

        return metrics

    def save_predict(self, predicted_data: OutputData, path_to_save: PathType):
        """ Saves pipeline forecasts in csv file """
        saved_to_path = predicted_data.save_predict(path_to_save)
        self.log.message(f'Predictions saved to {saved_to_path}')

    def export_as_project(self, project_path='fedot_project.zip'):
        export_project_to_zip(zip_name=project_path, opt_history=self.history,
                              pipeline=self.current_pipeline,
                              train_data=self.train_data, test_data=self.test_data)

    def import_as_project(self, project_path='fedot_project.zip'):
        self.current_pipeline, self.train_data, self.test_data, self.history = \
            import_project_from_zip(zip_path=project_path)
        # TODO workaround to init internal fields of API and data
        self.train_data = self.data_processor.define_data(
            features=self.train_data, is_predict=False)
        self.test_data = self.data_processor.define_data(
            features=self.test_data, is_predict=True)
        self.predict(self.test_data)

    def explain(self, features: FeaturesType = None,
                method: str = 'surrogate_dt', visualization: bool = True, **kwargs) -> Explainer:
        """Creates explanation for :attr:`~Fedot.current_pipeline` according to the selected ``method``.

        An :class:`Explainer` instance will return.

        Args:
            features: samples to be explained. If ``None``, ``train_data`` from last fit will be used.
            method: explanation method, defaults to ``surrogate_dt``
            visualization: print and plot the explanation simultaneously, defaults to ``True``.
        Notes:
            An explanation can be retrieved later by executing :meth:`Explainer.visualize`.
        """
        pipeline = self.current_pipeline
        if features is None:
            data = self.train_data
        else:
            data = self.data_processor.define_data(features=features,
                                                   is_predict=False)
        explainer = explain_pipeline(pipeline=pipeline, data=data, method=method,
                                     visualization=visualization, **kwargs)

        return explainer

    def return_report(self) -> pd.DataFrame:
        """ Function returns a report on time consumption.

            The following steps are presented in this report:
            - 'Data Definition (fit)': Time spent on data definition in fit().
            - 'Data Preprocessing': Total time spent on preprocessing data, includes fitting and predicting stages.
            - 'Fitting (summary)': Total time spent on Composing, Tuning and Training Inference.
            - 'Composing': Time spent on searching for the best pipeline.
            - 'Train Inference': Time spent on training the pipeline found during composing.
            - 'Tuning (composing)': Time spent on hyperparameters tuning in the whole fitting, if with_tune is True.
            - 'Tuning (after)': Time spent on .tune() (hyperparameters tuning) after composing.
            - 'Data Definition (predict)': Time spent on data definition in predict().
            - 'Predicting': Time spent on predicting (inference).
        """
        report = fedot_composer_timer.report

        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        report = pd.DataFrame(data=report.values(), index=report.keys())
        return report.iloc[:, 0].dt.components.iloc[:, :-2]

    @staticmethod
    def _init_logger(logging_level: int):
        # reset logging level for Singleton
        Log().reset_logging_level(logging_level)
        return default_log(prefix='FEDOT logger')
