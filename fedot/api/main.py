import logging
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.api_data_analyser import DataAnalyser
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.predefined_model import PredefinedModel
from fedot.core.constants import DEFAULT_API_TIMEOUT_MINUTES
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.visualisation import plot_biplot, plot_forecast, plot_roc_auc
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast, convert_forecast_to_output
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import TaskParams, TaskTypesEnum
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.core.visualisation.opt_viz_extra import visualise_pareto
from fedot.explainability.explainer_template import Explainer
from fedot.explainability.explainers import explain_pipeline
from fedot.preprocessing.preprocessing import merge_preprocessors
from fedot.remote.remote_evaluator import RemoteEvaluator
from fedot.utilities.project_import_export import export_project_to_zip, import_project_from_zip

NOT_FITTED_ERR_MSG = 'Model not fitted yet'

FeaturesType = Union[str, np.ndarray, pd.DataFrame, InputData, dict]
TargetType = Union[str, np.ndarray, pd.Series, dict]


class Fedot:
    """Main class for FEDOT API.

    Facade for ApiDataProcessor, ApiComposer, ApiMetrics, ApiInitialAssumptions.

    Args:
        problem: the name of modelling problem to solve

            .. details:: possible ``problem`` options:

                - ``classification`` -> for classification task
                - ``regression`` -> for regression task
                - ``ts_forecasting`` -> for time serires forecasting task

        timeout: time for model design (in minutes): ``None`` or ``-1`` means infinite time
        task_params: additional parameters of the task
        seed: value for fixed random seed
        logging_level: logging levels are the same as in 'logging'

            .. details:: possible ``logging_level`` options:

                    - ``50`` -> critical
                    - ``40`` -> error
                    - ``30`` -> warning
                    - ``20`` -> info
                    - ``10`` -> debug
                    - ``0`` -> nonset

        safe_mode: if set ``True`` it will cut large datasets to prevent memory overflow and use label encoder
            instead of oneHot encoder if summary cardinality of categorical features is high.
        n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
        max_depth: max depth of the pipeline
        max_arity: max arity of the pipeline nodes
        pop_size: population size for composer
        num_of_generations: number of generations for composer
        keep_n_best: Number of the best individuals of previous generation to keep in next generation.
        available_operations: list of model names to use
        stopping_after_n_generation': composer will stop after ``n`` generation without improving
        with_tuning: allow hyperparameters tuning for the model
        cv_folds: number of folds for cross-validation
        validation_blocks: number of validation blocks for time series forecasting
        max_pipeline_fit_time: time constraint for operation fitting (minutes)
        initial_assumption: initial assumption for composer
        genetic_scheme: name of the genetic scheme
        history_folder: name of the folder for composing history
        composer_metric:  metric for quality calculation during composing
        collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
        preset: name of preset for model building (e.g. 'best_quality', 'fast_train', 'gpu'):

            .. details:: possible ``preset`` options:

                - ``best_quality`` -> All models that are available for this data type and task are used
                - ``fast_train`` -> Models that learn quickly. This includes preprocessing operations
                  (data operations) that only reduce the dimensionality of the data, but cannot increase it.
                  For example, there are no polynomial features and one-hot encoding operations
                - ``stable`` -> The most reliable preset in which the most stable operations are included.
                - ``auto`` -> Automatically determine which preset should be used.
                - ``gpu`` -> Models that use GPU resources for computation.
                - ``ts`` -> A special preset with models for time series forecasting task.
                - ``automl`` -> A special preset with only AutoML libraries such as TPOT and H2O as operations.

        tuner_metric:  metric for quality calculation during tuning
        use_pipelines_cache: bool indicating whether to use pipeline structures caching, enabled by default.
        use_preprocessing_cache: bool indicating whether to use optional preprocessors caching, enabled by default.
        cache_folder: path to the place where cache files should be stored (if any cache is enabled).
        show_progress: bool indicating whether to show progress using tqdm/tuner or not
    """

    def __init__(self,
                 problem: str,
                 timeout: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES,
                 task_params: TaskParams = None,
                 seed=None, logging_level: int = logging.ERROR,
                 safe_mode=False,
                 n_jobs: int = 1,
                 **composer_tuner_params
                 ):

        # Classes for dealing with metrics, data sources and hyperparameters
        self.metrics = ApiMetrics(problem)
        self.api_composer = ApiComposer(problem)
        self.params = ApiParams()

        # Define parameters, that were set via init in init
        input_params = {'problem': self.metrics.main_problem, 'timeout': timeout,
                        'composer_tuner_params': composer_tuner_params, 'task_params': task_params,
                        'seed': seed, 'logging_level': logging_level, 'n_jobs': n_jobs}
        self.params.initialize_params(input_params)

        # Initialize ApiComposer's cache parameters via ApiParams
        self.api_composer.init_cache(self.params.api_params['use_pipelines_cache'],
                                     self.params.api_params['use_preprocessing_cache'],
                                     self.params.api_params['cache_folder'])

        # Initialize data processors for data preprocessing and preliminary data analysis
        self.data_processor = ApiDataProcessor(task=self.params.api_params['task'])
        self.data_analyser = DataAnalyser(safe_mode=safe_mode)

        self.target: Optional[TargetType] = None
        self.prediction: Optional[OutputData] = None
        self.train_data: Optional[InputData] = None
        self.test_data: Optional[InputData] = None

        # Outputs
        self.current_pipeline: Optional[Pipeline] = None
        self.best_models: Sequence[Pipeline] = ()
        self.history: Optional[OptHistory] = None

    def fit(self,
            features: Union[str, np.ndarray, pd.DataFrame, InputData, dict],
            target: TargetType = 'target',
            predefined_model: Union[str, Pipeline] = None) -> Pipeline:
        """Fits the graph with a predefined structure or compose and fit the new graph

        Args:
            features: the array with features of train data
            target: the array with target values of train data
            predefined_model: the name of the atomic model or Pipeline instance.
                If argument is ``auto``, perform initial assumption generation and then fit the pipeline

        Returns:
            Pipeline object

        """
        self.target = target

        self.train_data = self.data_processor.define_data(features=features, target=target, is_predict=False)

        # Launch data analyser - it gives recommendations for data preprocessing
        full_train_not_preprocessed = deepcopy(self.train_data)
        recommendations = self.data_analyser.give_recommendation(self.train_data)
        self.data_processor.accept_and_apply_recommendations(self.train_data, recommendations)
        self.params.accept_and_apply_recommendations(self.train_data, recommendations)
        self._init_remote_if_necessary()
        self.params.api_params['train_data'] = self.train_data

        if predefined_model is not None:
            # Fit predefined model and return it without composing
            self.current_pipeline = PredefinedModel(predefined_model,
                                                    self.train_data,
                                                    self.params.api_params['logger']).fit()
        else:
            self.current_pipeline, self.best_models, self.history = \
                self.api_composer.obtain_model(**self.params.api_params)

            # Final fit for obtained pipeline on full dataset
            if self.history and not self.history.is_empty() or not self.current_pipeline.is_fitted:
                self._train_pipeline_on_full_dataset(recommendations, full_train_not_preprocessed)
                self.params.api_params['logger'].message('Final pipeline was fitted')
            else:
                self.params.api_params['logger'].message('Already fitted initial pipeline is used')

        # Store data encoder in the pipeline if it is required
        self.current_pipeline.preprocessor = merge_preprocessors(self.data_processor.preprocessor,
                                                                 self.current_pipeline.preprocessor)

        self.params.api_params['logger'].message(f'Final pipeline: {str(self.current_pipeline)}')

        return self.current_pipeline

    def predict(self,
                features: FeaturesType,
                save_predictions: bool = False) -> np.ndarray:
        """Predicts new target using already fitted model

        Args:
            features: the array with features of test data
            save_predictions: if ``True`` - save predictions as csv-file in working directory

        Returns:
            the array with prediction values
        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        self.test_data = self.data_processor.define_data(target=self.target, features=features, is_predict=True)

        self.prediction = self.data_processor.define_predictions(current_pipeline=self.current_pipeline,
                                                                 test_data=self.test_data)

        if save_predictions:
            self.save_predict(self.prediction)

        return self.prediction.predict

    def predict_proba(self,
                      features: FeaturesType,
                      save_predictions: bool = False,
                      probs_for_all_classes: bool = False) -> np.ndarray:
        """Predicts the probability of new target using already fitted classification model

        Args:
            features: the array with features of test data
            save_predictions: if ``True`` - save predictions as csv-file in working directory
            probs_for_all_classes: if ``True`` - return probability for each class even for binary case

        Returns:
            the array with prediction values
        """

        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.params.api_params['task'].task_type == TaskTypesEnum.classification:
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
            pre_history: the array with features for pre-history of the forecast
            horizon: num of steps to forecast
            save_predictions: if ``True`` save predictions as csv-file in working directory

        Returns:
            the array with prediction values
        """
        self._check_forecast_applicable()

        forecast_length = self.train_data.task.task_params.forecast_length
        horizon = horizon if horizon is not None else forecast_length
        pre_history = pre_history if pre_history is not None else self.train_data
        self.test_data = self.data_processor.define_data(target=self.target,
                                                         features=pre_history,
                                                         is_predict=True)
        predict = out_of_sample_ts_forecast(self.current_pipeline, self.test_data, horizon)
        self.prediction = convert_forecast_to_output(self.test_data, predict)
        if save_predictions:
            self.save_predict(self.prediction)
        return self.prediction.predict

    def _check_forecast_applicable(self):
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.params.api_params['task'].task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

    def load(self, path):
        """Loads saved graph from disk

        Args:
            path: path to ``json`` file with model
        """
        self.current_pipeline = Pipeline()
        self.current_pipeline.load(path)
        self.data_processor.preprocessor = self.current_pipeline.preprocessor

    def plot_pareto(self):
        metric_names = self.params.metric_to_compose
        # archive_history stores archives of the best models.
        # Each archive is sorted from the best to the worst model,
        # so the best_candidates is sorted too.
        best_candidates = self.history.archive_history[-1]
        visualise_pareto(front=best_candidates,
                         objectives_names=metric_names,
                         show=True)

    def plot_prediction(self, target: Optional[Any] = None):
        """Plots the prediction obtained from graph

        Args:
            target: user-specified name of target variable for :obj:`MultiModalData`
        """
        if self.prediction is not None:
            if self.params.api_params['task'].task_type == TaskTypesEnum.ts_forecasting:
                plot_forecast(self.test_data, self.prediction, target)
            elif self.params.api_params['task'].task_type == TaskTypesEnum.regression:
                plot_biplot(self.prediction)
            elif self.params.api_params['task'].task_type == TaskTypesEnum.classification:
                self.predict_proba(self.test_data)
                plot_roc_auc(self.test_data, self.prediction)
            else:
                self.params.api_params['logger'].error('Not supported yet')
                raise NotImplementedError(f"For task {self.params.api_params['task']} plot prediction is not supported")
        else:
            self.params.api_params['logger'].error('No prediction to visualize')
            raise ValueError('Prediction from model is empty')

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = None) -> dict:
        """Gets quality metrics for the fitted graph

        Args:

            target: the array with target values of test data
            metric_names: the names of required metrics

        Returns:
            the values of quality metrics
        """
        if metric_names is None:
            metric_names = self.params.metric_name

        if target is not None:
            if self.test_data is None:
                self.test_data = InputData(idx=range(len(self.prediction.predict)),
                                           features=None,
                                           target=target[:len(self.prediction.predict)],
                                           task=self.train_data.task,
                                           data_type=self.train_data.data_type)
            else:
                self.test_data.target = target[:len(self.prediction.predict)]

        # TODO change to sklearn metrics
        metric_names = ensure_wrapped_in_sequence(metric_names)

        calculated_metrics = dict()
        for metric_name in metric_names:
            if self.metrics.get_metrics_mapping(metric_name) is NotImplemented:
                self.params.api_params['logger'].warning(f'{metric_name} is not available as metric')
            else:
                composer_metric = self.metrics.get_metrics_mapping(metric_name)
                metric_cls = MetricsRepository().metric_class_by_id(composer_metric)
                prediction = deepcopy(self.prediction)
                if metric_name == "roc_auc":  # for roc-auc we need probabilities
                    prediction.predict = self.predict_proba(self.test_data)
                else:
                    prediction.predict = self.predict(self.test_data)
                real = deepcopy(self.test_data)

                # Work inplace - correct predictions
                self.data_processor.correct_predictions(real=real,
                                                        prediction=prediction)

                # real.target = np.ravel(real.target)

                metric_value = abs(metric_cls.metric(reference=real,
                                                     predicted=prediction))
                calculated_metrics[metric_name] = metric_value

        return calculated_metrics

    def save_predict(self, predicted_data: OutputData):
        # TODO unify with OutputData.save_to_csv()
        """ Saves pipeline forecasts in csv file """
        if len(predicted_data.predict.shape) >= 2:
            prediction = predicted_data.predict.tolist()
        else:
            prediction = predicted_data.predict
        pd.DataFrame({'Index': predicted_data.idx,
                      'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)
        self.params.api_params['logger'].message('Predictions was saved in current directory.')

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
                method: str = 'surrogate_dt', visualize: bool = True, **kwargs) -> 'Explainer':
        """Creates explanation for *current_pipeline* according to the selected 'method'.
            An :obj:`Explainer` instance will return.

        Args:
            features: samples to be explained. If ``None``, ``train_data`` from last fit will be used.
            method: explanation method, defaults to ``surrogate_dt``
            visualize: print and plot the explanation simultaneously, defaults to ``True``.
        Notes:
            The explanation can be retrieved later by executing :obj:`explainer.visualize()`
        """
        pipeline = self.current_pipeline
        if features is None:
            data = self.train_data
        else:
            data = self.data_processor.define_data(features=features,
                                                   is_predict=False)
        explainer = explain_pipeline(pipeline=pipeline, data=data, method=method,
                                     visualize=visualize, **kwargs)

        return explainer

    def _init_remote_if_necessary(self):
        remote = RemoteEvaluator()
        if remote.use_remote and remote.remote_task_params is not None:
            task = self.params.api_params['task']
            if task.task_type is TaskTypesEnum.ts_forecasting:
                task_str = \
                    f'Task(TaskTypesEnum.ts_forecasting, ' \
                    f'TsForecastingParams(forecast_length={task.task_params.forecast_length}))'
            else:
                task_str = f'Task({str(task.task_type)})'
            remote.remote_task_params.task_type = task_str
            remote.remote_task_params.is_multi_modal = isinstance(self.train_data, MultiModalData)

            if isinstance(self.target, str):
                remote.remote_task_params.target = self.target

    def _train_pipeline_on_full_dataset(self, recommendations: dict, full_train_not_preprocessed):
        """Applies training procedure for obtained pipeline if dataset was clipped
        """

        if recommendations:
            # if data was cut we need to refit pipeline on full data
            self.data_processor.accept_and_apply_recommendations(full_train_not_preprocessed,
                                                                 {k: v for k, v in recommendations.items()
                                                                  if k != 'cut'})
        self.current_pipeline.fit(
            full_train_not_preprocessed,
            n_jobs=self.params.api_params['n_jobs'],
        )
