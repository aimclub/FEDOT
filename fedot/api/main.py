import logging
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd
from golem.core.dag.graph_utils import graph_structure
from golem.core.log import default_log, Log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.visualisation.opt_viz_extra import visualise_pareto

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.data_definition import FeaturesType, TargetType
from fedot.api.api_utils.input_analyser import InputAnalyser
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.predefined_model import PredefinedModel
from fedot.core.constants import DEFAULT_API_TIMEOUT_MINUTES, DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.visualisation import plot_biplot, plot_forecast, plot_roc_auc
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import convert_forecast_to_output, out_of_sample_ts_forecast
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.tasks import TaskParams, TaskTypesEnum
from fedot.core.utils import set_random_seed
from fedot.explainability.explainer_template import Explainer
from fedot.explainability.explainers import explain_pipeline
from fedot.preprocessing.base_preprocessing import BasePreprocessor
from fedot.remote.remote_evaluator import RemoteEvaluator
from fedot.utilities.memory import MemoryAnalytics
from fedot.utilities.project_import_export import export_project_to_zip, import_project_from_zip

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


class Fedot:
    """Main class for FEDOT API.

    Facade for
    :class:`ApiParams`, :class:`ApiDataProcessor`, :class:`ApiComposer`, :class:`ApiMetrics`,
    :class:`~fedot.api.api_utils.assumptions.assumptions_handler.AssumptionsHandler`.

    Args:
        problem: name of the modelling problem to solve.
            .. details:: Possible options:

                - ``classification`` -> for classification task
                - ``regression`` -> for regression task
                - ``ts_forecasting`` -> for time series forecasting task

        timeout: time for model design (in minutes): ``None`` or ``-1`` means infinite time.
        task_params: additional parameters of the task.
        seed: value for a fixed random seed.
        logging_level: logging levels are the same as in `logging <https://docs.python.org/3/library/logging.html>`_.

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
        parallelization_mode (str): type of evaluation for groups of individuals (``'populational'`` or
            ``'sequential'``). Default value is ``'populational'``.

        initial_assumption (Union[Pipeline, List[Pipeline]]): initial assumption(s) for composer.
            Can be either a single :class:`Pipeline` or sequence of ones. Default values are task-specific and
            selected by the method :meth:`~fedot.api.api_utils.assumptions.task_assumptions.TaskAssumptions.for_task`.

        available_operations (List[str])): list of model names to use. Pick the names according to the `repositories \
            <https://github.com/aimclub/FEDOT/tree/master/fedot/core/repository/data>`_.

        metric (Union[str, Callable, fedot.core.repository.quality_metrics_repository.MetricsEnum, List[str, \
            Callable, fedot.core.repository.quality_metrics_repository.MetricsEnum]):
            metric for quality calculation during composing, also is used for tuning if ``with_tuning=True``.

            .. details:: Default value depends on a given task:

                - ``roc_auc`` -> for classification
                - ``rmse`` -> for regression & time series forecasting

            .. details:: Available metrics are listed in the following enumerations:

                - classification -> \
                    :class:`~fedot.core.repository.quality_metrics_repository.ClassificationMetricsEnum`
                - regression & time series forcasting -> \
                    :class:`~fedot.core.repository.quality_metrics_repository.RegressionMetricsEnum`
                - pipeline complexity (task-independent)-> \
                    :class:`~fedot.core.repository.quality_metrics_repository.ComplexityMetricsEnum`

        collect_intermediate_metric (bool): save metrics for intermediate (non-root) nodes in composed
            :class:`Pipeline`.

        cv_folds (int): number of folds for cross-validation.

            .. details:: Default value depends on the given ``problem``:

                - ``5`` -> for classification and regression tasks
                - ``3`` -> for time series forecasting task

        validation_blocks (int): number of validation blocks for time series forecasting. Default value is ``2``.

        show_progress (bool): indicates whether to show progress using tqdm/tuner or not. Defaults to ``True``.

        num_of_generations (int): number of evolutionary generations for composer. Defaults to ``None`` - no limit.
        early_stopping_iterations (int): composer will stop after `n` generation without improving.
        early_stopping_timeout (int): stagnation timeout in minutes: composer will stop after `n` minutes
            without improving. Defaults to ``10``.
        max_pipeline_fit_time (int): time constraint for operation fitting (in minutes).
            Defaults to ``None`` - no limit. Once the limit is reached, the candidate pipeline will be dropped.

        max_depth (int): max depth of the pipeline. Defaults to ``6``.
        max_arity (int): max arity of the pipeline nodes. Defaults to ``3``.
        pop_size (int): size of population (generation) during composing. Defaults to ``20``.
        keep_n_best (int): number of the best individuals in generation that survive during the evolution.
             Defaults to ``1``.
        genetic_scheme (str): name of the genetic scheme. Defaults to ``steady_state``.

        with_tuning (bool): flag for tuning hyperparameters of the final evolved :class:`Pipeline`.
            Defaults to ``True``.

        preset (str): name of the preset for model building (e.g. ``'best_quality'``, ``'fast_train'``, ``'gpu'``).
            Default value is ``'auto'``.

            .. details:: Possible options:

                - ``'best_quality'`` -> All models that are available for this data type and task are used
                - ``'fast_train'`` -> Models that learn quickly. This includes preprocessing operations
                  (data operations) that only reduce the dimensionality of the data, but cannot increase it.
                  For example, there are no polynomial features and one-hot encoding operations
                - ``'stable'`` -> The most reliable preset in which the most stable operations are included
                - ``'auto'`` -> Automatically determine which preset should be used
                - ``'gpu'`` -> Models that use GPU resources for computation
                - ``'ts'`` -> A special preset with models for time series forecasting task
                - ``'automl'`` -> A special preset with only AutoML libraries such as TPOT and H2O as operations

        use_input_preprocessing (bool): indicates whether to do preprocessing of further given data.
            Defaults to ``True``.
        use_meta_rules (bool): indicates whether to change set params according to FEDOT meta rules.
        use_pipelines_cache (bool): indicates whether to use pipeline structures caching. Defaults to ``True``.
        use_preprocessing_cache (bool): bool indicating whether to use optional preprocessors caching.
            Defaults to ``True``.
        cache_dir (str): path to the directory containing cache files (if any cache is enabled).
            By default, creates a folder named "FEDOT" in temporary system files of the OS.
        history_dir (str): relative or absolute path of the folder for composing history. Has the same default value
            as ``cache_dir``. A relative path is relative to the default value.

        optimizer (Type[golem.core.optimisers.optimizer.GraphOptimizer]): inherit from
            :class:`~golem.core.optimisers.optimizer.GraphOptimizer` to specify a custom optimizer.
            Default optimizer is :class:`~golem.core.optimisers.genetic.gp_optimizer.EvoGraphOptimizer`.
            See the `example \
<https://github.com/aimclub/FEDOT/blob/master/examples/advanced/fedot_based_solutions/external_optimizer.py>`_
    """

    def __init__(self,
                 problem: str,
                 timeout: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES,
                 task_params: TaskParams = None,
                 seed: Optional[int] = None, logging_level: int = logging.ERROR,
                 safe_mode: bool = False,
                 n_jobs: int = -1,
                 **composer_tuner_params
                 ):

        set_random_seed(seed)
        self.log = self._init_logger(logging_level)

        # Classes for dealing with metrics, data sources and hyperparameters
        self.params = ApiParams(composer_tuner_params, problem, task_params, n_jobs, timeout)
        self.metrics = ApiMetrics(self.params.task, self.params.get('metric'))

        self.api_composer = ApiComposer(self.params, self.metrics)

        # Initialize data processors for data preprocessing and preliminary data analysis
        self.data_processor = ApiDataProcessor(task=self.params.task,
                                               use_input_preprocessing=self.params.get('use_input_preprocessing'))
        self.data_analyser = InputAnalyser(safe_mode=safe_mode)

        self.target: Optional[TargetType] = None
        self.prediction: Optional[OutputData] = None
        self._is_in_sample_prediction = True
        self.train_data: Optional[InputData] = None
        self.test_data: Optional[InputData] = None

        # Outputs
        self.current_pipeline: Optional[Pipeline] = None
        self.best_models: Sequence[Pipeline] = ()
        self.history: Optional[OptHistory] = None

    def fit(self,
            features: FeaturesType,
            target: TargetType = 'target',
            predefined_model: Union[str, Pipeline] = None) -> Pipeline:
        """Composes and fits a new pipeline, or fits a predefined one.

        Args:
            features: train data feature values in one of the supported features formats.
            target: train data target values in one of the supported target formats.
            predefined_model: the name of a single model or a :class:`Pipeline` instance, or ``"auto"``.
                With any value specified, the method does not perform composing and tuning.
                In case of ``"auto"``, the method generates a single initial assumption and then fits
                the created pipeline.

        Returns:
            :class:`Pipeline` object.

        """

        MemoryAnalytics.start()

        self.target = target

        self.train_data = self.data_processor.define_data(features=features, target=target, is_predict=False)
        self.params.update_available_operations_by_preset(self.train_data)

        if self.params.get('use_input_preprocessing'):
            # Launch data analyser - it gives recommendations for data preprocessing
            recommendations_for_data, recommendations_for_params = \
                self.data_analyser.give_recommendations(input_data=self.train_data,
                                                        input_params=self.params)
            self.data_processor.accept_and_apply_recommendations(input_data=self.train_data,
                                                                 recommendations=recommendations_for_data)
            self.params.accept_and_apply_recommendations(input_data=self.train_data,
                                                         recommendations=recommendations_for_params)
        else:
            recommendations_for_data = None

        self._init_remote_if_necessary()

        if predefined_model is not None:
            # Fit predefined model and return it without composing
            self.current_pipeline = PredefinedModel(predefined_model, self.train_data, self.log,
                                                    use_input_preprocessing=self.params.get(
                                                        'use_input_preprocessing')).fit()
        else:
            self.current_pipeline, self.best_models, self.history = self.api_composer.obtain_model(self.train_data)

            if self.current_pipeline is None:
                raise ValueError('No models were found')

            full_train_not_preprocessed = deepcopy(self.train_data)
            # Final fit for obtained pipeline on full dataset
            if self.history and not self.history.is_empty() or not self.current_pipeline.is_fitted:
                self._train_pipeline_on_full_dataset(recommendations_for_data, full_train_not_preprocessed)
                self.log.message('Final pipeline was fitted')
            else:
                self.log.message('Already fitted initial pipeline is used')

        # Store data encoder in the pipeline if it is required
        self.current_pipeline.preprocessor = BasePreprocessor.merge_preprocessors(
            self.data_processor.preprocessor, self.current_pipeline.preprocessor)

        self.log.message(f'Final pipeline: {graph_structure(self.current_pipeline)}')

        MemoryAnalytics.finish()

        return self.current_pipeline

    def tune(self,
             input_data: Optional[InputData] = None,
             metric_name: Optional[Union[str, Callable]] = None,
             iterations: int = DEFAULT_TUNING_ITERATIONS_NUMBER,
             timeout: Optional[float] = None,
             cv_folds: Optional[int] = None,
             validation_blocks: Optional[int] = None,
             n_jobs: Optional[int] = None,
             show_progress: bool = False) -> Pipeline:
        """Method for hyperparameters tuning of current pipeline

        Args:
            input_data: data for tuning pipeline.
            metric_name: name of metric for quality tuning.
            iterations: numbers of tuning iterations.
            timeout: time for tuning (in minutes). If ``None`` or ``-1`` means tuning until max iteration reach.
            cv_folds: number of folds on data for cross-validation.
            validation_blocks: number of validation blocks (used for time-series forecasting problem).
            n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's).
            show_progress: shows progress of tuning if ``True``.

        Returns:
            :class:`Pipeline` object.

        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        input_data = input_data or self.train_data
        cv_folds = cv_folds or self.params.get('cv_folds')
        validation_blocks = validation_blocks or self.params.get('validation_blocks')
        n_jobs = n_jobs or self.params.n_jobs

        metric = self.metrics.obtain_metrics(metric_name)[0] if metric_name else self.metrics.metric_functions[0]

        pipeline_tuner = (TunerBuilder(self.params.task)
                          .with_tuner(SimultaneousTuner)
                          .with_cv_folds(cv_folds)
                          .with_validation_blocks(validation_blocks)
                          .with_n_jobs(n_jobs)
                          .with_metric(metric)
                          .with_iterations(iterations)
                          .with_timeout(timeout)
                          .build(input_data))

        self.current_pipeline = pipeline_tuner.tune(self.current_pipeline, show_progress)
        self.api_composer.was_tuned = pipeline_tuner.was_tuned

        # Tuner returns a not fitted pipeline, and it is required to fit on train dataset
        self.current_pipeline.fit(self.train_data)
        return self.current_pipeline

    def predict(self,
                features: FeaturesType,
                save_predictions: bool = False,
                in_sample: bool = True,
                validation_blocks: Optional[int] = None) -> np.ndarray:
        """Predicts new target using already fitted model.

        For time-series performs forecast with depth ``forecast_length`` if ``in_sample=False``.
        If ``in_sample=True`` performs in-sample forecast using features as sample.

        Args:
            features: the array with features of test data.
            save_predictions: if ``True`` - save predictions as csv-file in working directory.
            in_sample: used while time-series prediction. If ``in_sample=True`` performs in-sample forecast using
                features with number if iterations specified in ``validation_blocks``.
            validation_blocks: number of validation blocks for in-sample forecast.
                If ``validation_blocks = None`` uses number of validation blocks set during model initialization
                (default is ``2``).


        Returns:
            The array with prediction values.
        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        self.test_data = self.data_processor.define_data(target=self.target, features=features, is_predict=True)
        self._is_in_sample_prediction = in_sample
        validation_blocks = validation_blocks or self.params.get('validation_blocks')

        self.prediction = self.data_processor.define_predictions(current_pipeline=self.current_pipeline,
                                                                 test_data=self.test_data,
                                                                 in_sample=self._is_in_sample_prediction,
                                                                 validation_blocks=validation_blocks)

        if save_predictions:
            self.save_predict(self.prediction)

        return self.prediction.predict

    def predict_proba(self,
                      features: FeaturesType,
                      save_predictions: bool = False,
                      probs_for_all_classes: bool = False) -> np.ndarray:
        """Predicts the probability of new target using already fitted classification model

        Args:
            features: the array with features of test data.
            save_predictions: if ``True`` - save predictions as ``.csv`` file in working directory.
            probs_for_all_classes: if ``True`` - return probability for each class even for binary classification.

        Returns:
            The array with prediction values.
        """

        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.params.task.task_type == TaskTypesEnum.classification:
            self.test_data = self.data_processor.define_data(target=self.target,
                                                             features=features, is_predict=True)

            mode = 'full_probs' if probs_for_all_classes else 'probs'

            self.prediction = self.current_pipeline.predict(self.test_data, output_mode=mode)

            if save_predictions:
                self.save_predict(self.prediction)
        else:
            raise ValueError('Probabilities of predictions are available only for classification')

        return self.prediction.predict

    def forecast(self,
                 pre_history: Optional[Union[str, Tuple[np.ndarray, np.ndarray], InputData, dict]] = None,
                 horizon: Optional[int] = None,
                 save_predictions: bool = False) -> np.ndarray:
        """Forecasts the new values of time series. If horizon is bigger than forecast length of fitted model -
        out-of-sample forecast is applied (not supported for multi-modal data).

        Args:
            pre_history: the array with features for pre-history of the forecast.
            horizon: amount of steps to forecast.
            save_predictions: if ``True`` save predictions as csv-file in working directory.

        Returns:
            The array with prediction values.
        """
        self._check_forecast_applicable()

        forecast_length = self.train_data.task.task_params.forecast_length
        horizon = horizon or forecast_length
        if pre_history is None:
            pre_history = self.train_data
            pre_history.target = None
        self.test_data = self.data_processor.define_data(target=self.target,
                                                         features=pre_history,
                                                         is_predict=True)
        predict = out_of_sample_ts_forecast(self.current_pipeline, self.test_data, horizon)
        self.prediction = convert_forecast_to_output(self.test_data, predict)
        self._is_in_sample_prediction = False
        if save_predictions:
            self.save_predict(self.prediction)
        return self.prediction.predict

    def _check_forecast_applicable(self):
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.params.task.task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

    def load(self, path):
        """Loads saved graph from disk

        Args:
            path: path to ``json`` file with model.
        """
        self.current_pipeline = Pipeline(use_input_preprocessing=self.params.get('use_input_preprocessing'))
        self.current_pipeline.load(path)
        self.data_processor.preprocessor = self.current_pipeline.preprocessor

    def plot_pareto(self):
        metric_names = self.metrics.metric_names
        # archive_history stores archives of the best models.
        # Each archive is sorted from the best to the worst model,
        # so the best_candidates is sorted too.
        best_candidates = self.history.archive_history[-1]
        visualise_pareto(front=best_candidates,
                         objectives_names=metric_names,
                         show=True)

    def plot_prediction(self, in_sample: Optional[bool] = None, target: Optional[Any] = None):
        """Plots the prediction obtained from graph

        Args:
            in_sample: if current prediction is in_sample (for time-series forecasting), plots predictions as future
                values.
            target: user-specified name of target variable for :class:`MultiModalData`.

        """
        task = self.params.task

        if self.prediction is not None:
            if task.task_type == TaskTypesEnum.ts_forecasting:
                in_sample = in_sample or self._is_in_sample_prediction
                plot_forecast(self.test_data, self.prediction, in_sample, target)
            elif task.task_type == TaskTypesEnum.regression:
                plot_biplot(self.prediction)
            elif task.task_type == TaskTypesEnum.classification:
                self.predict_proba(self.test_data)
                plot_roc_auc(self.test_data, self.prediction)
            else:
                self.log.error('Not supported yet')
                raise NotImplementedError(f"For task {task} plot prediction is not supported")
        else:
            self.log.error('No prediction to visualize')
            raise ValueError('Prediction from model is empty')

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = None,
                    in_sample: Optional[bool] = None,
                    validation_blocks: Optional[int] = None) -> dict:
        """Gets quality metrics for the fitted graph

        Args:
            target: the array with target values of test data. If ``None``, target specified for fit is used.
            metric_names: the names of required metrics.
            in_sample: used for time series forecasting.
                If True prediction will be obtained as ``.predict(..., in_sample=True)``.
            validation_blocks: number of validation blocks for time series in-sample forecast.
                If ``None``, uses number of validation blocks set during model initialization (default is ``2``).

        Returns:
            The values of quality metrics.
        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if target is not None:
            if self.test_data is None:
                self.test_data = InputData(idx=range(len(self.prediction.predict)),
                                           features=None,
                                           target=target[:len(self.prediction.predict)],
                                           task=self.train_data.task,
                                           data_type=self.train_data.data_type)
            else:
                self.test_data.target = target[:len(self.prediction.predict)]

        metrics = self.metrics.obtain_metrics(metric_names) if metric_names else self.metrics.metric_functions
        metric_names = self.metrics.get_metric_names(metrics)

        in_sample = in_sample if in_sample is not None else self._is_in_sample_prediction

        if in_sample:
            validation_blocks = validation_blocks or self.params.get('validation_blocks')
        else:
            validation_blocks = None

        objective = MetricsObjective(metrics)
        obj_eval = PipelineObjectiveEvaluate(objective=objective,
                                             data_producer=lambda: (yield self.train_data, self.test_data),
                                             validation_blocks=validation_blocks, eval_n_jobs=self.params.n_jobs,
                                             do_unfit=False)

        metrics = obj_eval.evaluate(self.current_pipeline).values
        metrics = {metric_name: round(abs(metric), 3) for (metric_name, metric) in zip(metric_names, metrics)}

        return metrics

    def save_predict(self, predicted_data: OutputData):
        # TODO unify with OutputData.save_to_csv()
        """ Saves pipeline forecasts in csv file """
        if len(predicted_data.predict.shape) >= 2:
            prediction = predicted_data.predict.tolist()
        else:
            prediction = predicted_data.predict
        pd.DataFrame({'Index': predicted_data.idx,
                      'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)
        self.log.message('Predictions was saved in current directory.')

    def export_as_project(self, project_path='fedot_project.zip'):
        export_project_to_zip(zip_name=project_path, opt_history=self.history,
                              pipeline=self.current_pipeline,
                              train_data=self.train_data, test_data=self.test_data)

    def import_as_project(self, project_path='fedot_project.zip'):
        self.current_pipeline, self.train_data, self.test_data, self.history = \
            import_project_from_zip(zip_path=project_path)
        # TODO workaround to init internal fields of API and data
        self.train_data = self.data_processor.define_data(features=self.train_data, is_predict=False)
        self.test_data = self.data_processor.define_data(features=self.test_data, is_predict=True)
        self.predict(self.test_data)

    def explain(self, features: FeaturesType = None,
                method: str = 'surrogate_dt', visualization: bool = True, **kwargs) -> 'Explainer':
        """Creates explanation for *current_pipeline* according to the selected 'method'.
            An :obj:`Explainer` instance will return.

        Args:
            features: samples to be explained. If ``None``, ``train_data`` from last fit will be used.
            method: explanation method, defaults to ``surrogate_dt``
            visualization: print and plot the explanation simultaneously, defaults to ``True``.
        Notes:
            The explanation can be retrieved later by executing :obj:`explainer.visualize()`.
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

    @staticmethod
    def _init_logger(logging_level: int):
        # reset logging level for Singleton
        Log().reset_logging_level(logging_level)
        return default_log(prefix='FEDOT logger')

    def _init_remote_if_necessary(self):
        remote = RemoteEvaluator()
        if remote.is_enabled and remote.remote_task_params is not None:
            task = self.params.task
            if task.task_type is TaskTypesEnum.ts_forecasting:
                task_str = (f'Task(TaskTypesEnum.ts_forecasting, '
                            f'TsForecastingParams(forecast_length={task.task_params.forecast_length}))')
            else:
                task_str = f'Task({str(task.task_type)})'
            remote.remote_task_params.task_type = task_str
            remote.remote_task_params.is_multi_modal = isinstance(self.train_data, MultiModalData)

            if isinstance(self.target, str) and remote.remote_task_params.target is None:
                remote.remote_task_params.target = self.target

    def _train_pipeline_on_full_dataset(self, recommendations: Optional[dict],
                                        full_train_not_preprocessed: Union[InputData, MultiModalData]):
        """Applies training procedure for obtained pipeline if dataset was clipped
        """

        if recommendations is not None:
            # if data was cut we need to refit pipeline on full data
            self.data_processor.accept_and_apply_recommendations(full_train_not_preprocessed,
                                                                 {k: v for k, v in recommendations.items()
                                                                  if k != 'cut'})
        self.current_pipeline.fit(
            full_train_not_preprocessed,
            n_jobs=self.params.n_jobs
        )
