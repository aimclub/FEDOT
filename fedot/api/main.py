from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fedot.api.api_utils.api_utils import ApiFacade
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.visualisation import plot_biplot, plot_roc_auc, plot_forecast
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import TaskParams, TaskTypesEnum
from fedot.explainability.explainers import explain_pipeline
from fedot.remote.remote_evaluator import RemoteEvaluator

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


class Fedot:
    """
    Main class for FEDOT API

    :param problem: the name of modelling problem to solve:
        - classification
        - regression
        - ts_forecasting
        - clustering
    :param preset: name of preset for model building (e.g. 'light', 'ultra-light')
    :param timeout: time for model design (in minutes)
    :param composer_params: parameters of pipeline optimisation
        The possible parameters are:
            'max_depth' - max depth of the pipeline
            'max_arity' - max arity of the pipeline nodes
            'pop_size' - population size for composer
            'num_of_generations' - number of generations for composer
            'timeout' - composing time (minutes)
            'available_operations' - list of model names to use
            'with_tuning' - allow huperparameters tuning for the model
            'cv_folds' - number of folds for cross-validation
            'validation_blocks' - number of validation blocks for time series forecasting
            'initial_pipeline' - initial assumption for composing
            'genetic_scheme' - name of the genetic scheme
            'history_folder' - name of the folder for composing history
    :param task_params:  additional parameters of the task
    :param seed: value for fixed random seed
    :param verbose_level: level of the output detailing
        (-1 - nothing, 0 - errors, 1 - messages,
        2 - warnings and info, 3-4 - basic and detailed debug)
    """

    def __init__(self,
                 problem: str,
                 preset: str = None,
                 timeout: Optional[float] = None,
                 composer_params: dict = None,
                 task_params: TaskParams = None,
                 seed=None, verbose_level: int = 0,
                 initial_pipeline: Pipeline = None):

        self.helper = ApiFacade(**{'problem': problem,
                                   'preset': preset,
                                   'timeout': timeout,
                                   'composer_params': composer_params,
                                   'task_params': task_params,
                                   'seed': seed,
                                   'verbose_level': verbose_level,
                                   'initial_pipeline': initial_pipeline})

        self.composer_dict = self.helper.initialize_params()
        self.composer_dict['current_model'] = None
        self.task_metrics, self.composer_metrics, self.tuner_metrics = self.helper.get_metrics_for_task(
            self.composer_dict['problem'],
            self.composer_dict['metric_name'])
        self.composer_dict['tuner_metric'] = self.tuner_metrics

        if timeout is not None:
            self.composer_dict['timeout'] = timeout

        if initial_pipeline is not None:
            self.composer_dict['initial_pipeline'] = initial_pipeline

    def fit(self,
            features: Union[str, np.ndarray, pd.DataFrame, InputData, dict],
            target: Union[str, np.ndarray, pd.Series] = 'target',
            predefined_model: Union[str, Pipeline] = None):
        """
        Fit the graph with a predefined structure or compose and fit the new graph

        :param features: the array with features of train data
        :param target: the array with target values of train data
        :param predefined_model: the name of the atomic model or Pipeline instance
        :return: Pipeline object
        """

        self.target_name = target
        self.train_data = self.helper.define_data(ml_task=self.composer_dict['task'],
                                                  features=features,
                                                  target=target,
                                                  is_predict=False)

        self._init_remote_if_necessary()

        is_composing_required = True
        if self.composer_dict['current_model'] is not None:
            is_composing_required = False

        if predefined_model is not None:
            is_composing_required = False
            if isinstance(predefined_model, Pipeline):
                self.current_pipeline = predefined_model
            elif isinstance(predefined_model, str):
                categorical_preprocessing = PrimaryNode('one_hot_encoding')
                scaling_preprocessing = SecondaryNode('scaling', nodes_from=[categorical_preprocessing])
                model = SecondaryNode(predefined_model, nodes_from=[scaling_preprocessing])
                self.current_pipeline = Pipeline(model)
            else:
                raise ValueError(f'{type(predefined_model)} is not supported as Fedot model')
            self.composer_dict['current_model'] = self.current_pipeline

        self.composer_dict['is_composing_required'] = is_composing_required
        self.composer_dict['train_data'] = self.train_data
        self.current_pipeline, self.best_models, self.history = self.helper.obtain_model(**self.composer_dict)
        return self.current_pipeline

    def predict(self,
                features: Union[str, np.ndarray, pd.DataFrame, InputData, dict],
                save_predictions: bool = False):
        """
        Predict new target using already fitted model

        :param features: the array with features of test data
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :return: the array with prediction values
        """
        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        self.test_data = self.helper.define_data(ml_task=self.composer_dict['task'], target=self.target_name,
                                                 features=features, is_predict=True)

        self.prediction = self.helper.define_predictions(task_type=self.composer_dict['task'].task_type,
                                                         current_pipeline=self.current_pipeline,
                                                         test_data=self.test_data)

        if save_predictions:
            self.helper.save_predict(self.prediction)

        return self.prediction.predict

    def predict_proba(self,
                      features: Union[str, np.ndarray, pd.DataFrame, InputData, dict],
                      save_predictions: bool = False,
                      probs_for_all_classes: bool = False):
        """
        Predict the probability of new target using already fitted classification model

        :param features: the array with features of test data
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :param probs_for_all_classes: return probability for each class even for binary case
        :return: the array with prediction values
        """

        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.composer_dict['task'].task_type == TaskTypesEnum.classification:
            self.test_data = self.helper.define_data(ml_task=self.composer_dict['task'], target=self.target_name,
                                                     features=features, is_predict=True)

            mode = 'full_probs' if probs_for_all_classes else 'probs'

            self.prediction = self.current_pipeline.predict(self.test_data, output_mode=mode)

            if save_predictions:
                self.helper.save_predict(self.prediction)
        else:
            raise ValueError('Probabilities of predictions are available only for classification')

        return self.prediction.predict

    def forecast(self,
                 pre_history: Union[str, Tuple[np.ndarray, np.ndarray], InputData, dict],
                 forecast_length: int = 1,
                 save_predictions: bool = False):
        """
        Forecast the new values of time series

        :param pre_history: the array with features for pre-history of the forecast
        :param forecast_length: num of steps to forecast
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :return: the array with prediction values
        """

        # TODO use forecast length

        if self.current_pipeline is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.composer_dict['task'].task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

        self.problem = self.train_data.task

        self.test_data = self.helper.define_data(ml_task=self.composer_dict['task'],
                                                 target=self.target_name,
                                                 features=pre_history,
                                                 is_predict=True)

        self.current_pipeline = Pipeline(self.current_pipeline.root_node)
        # TODO add incremental forecast
        self.prediction = self.current_pipeline.predict(self.test_data)
        if len(self.prediction.predict.shape) > 1:
            self.prediction.predict = np.squeeze(self.prediction.predict)

        if save_predictions:
            self.helper.save_predict(self.prediction)
        return self.prediction.predict

    def load(self, path):
        """
        Load saved graph from disk

        :param path to json file with model
        """
        self.current_pipeline.load(path)

    def plot_prediction(self):
        """
        Plot the prediction obtained from graph
        """

        if self.prediction is not None:
            if self.composer_dict['task'].task_type == TaskTypesEnum.ts_forecasting:
                plot_forecast(self.test_data, self.prediction)
            elif self.composer_dict['task'].task_type == TaskTypesEnum.regression:
                plot_biplot(self.prediction)
            elif self.composer_dict['task'].task_type == TaskTypesEnum.classification:
                self.predict_proba(self.test_data)
                plot_roc_auc(self.test_data, self.prediction)
            else:
                # TODO implement other visualizations
                self.composer_dict['logger'].error('Not supported yet')

        else:
            self.composer_dict['logger'].error('No prediction to visualize')

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = None) -> dict:
        """
        Get quality metrics for the fitted graph

        :param target: the array with target values of test data
        :param metric_names: the names of required metrics
        :return: the values of quality metrics
        """
        if metric_names is None:
            metric_names = self.composer_dict['metric_name']

        if target is not None:
            if self.test_data is None:
                self.test_data = InputData(idx=range(len(self.prediction.predict)),
                                           features=None,
                                           target=target[:len(self.prediction.predict)],
                                           task=self.train_data.task,
                                           data_type=self.train_data.data_type)
            else:
                self.test_data.target = target[:len(self.prediction.predict)]

        real = self.test_data

        # TODO change to sklearn metrics
        if not isinstance(metric_names, List):
            metric_names = [metric_names]

        calculated_metrics = dict()
        for metric_name in metric_names:
            if self.helper.get_composer_metrics_mapping(metric_name) is NotImplemented:
                self.composer_dict['logger'].warn(f'{metric_name} is not available as metric')
            else:
                metric_cls = MetricsRepository().metric_class_by_id(
                    self.helper.get_composer_metrics_mapping(metric_name))
                prediction = deepcopy(self.prediction)
                if metric_name == "roc_auc":  # for roc-auc we need probabilities
                    prediction.predict = self.predict_proba(self.test_data)
                real = deepcopy(self.test_data)
                real.target, prediction.predict = self.helper.check_prediction_shape(
                    task=self.composer_dict['task'].task_type,
                    metric_name=metric_name,
                    real=real,
                    prediction=prediction)

                metric_value = abs(metric_cls.metric(reference=real,
                                                     predicted=prediction))

                calculated_metrics[metric_name] = metric_value

        return calculated_metrics


    def explain(self, features: Union[str, np.ndarray, pd.DataFrame, InputData, dict] = None,
                method: str = 'surrogate_dt', visualize: bool = True, **kwargs) -> 'Explainer':
        """Create explanation for 'current_pipeline' according to the selected 'method'.
        An `Explainer` instance is returned.

        :param features: samples to be explained. If `None`, `train_data` from last fit is used.
        :param method: explanation method, defaults to 'surrogate_dt'. Options: ['surrogate_dt', ...]
        :param visualize: print and plot the explanation simultaneously, defaults to True.
            The explanation can be retrieved later by executing `explainer.visualize()`.
        """
        pipeline = self.current_pipeline
        if features is None:
            data = self.train_data
        else:
            data = self.helper.define_data(
                ml_task=self.composer_dict['task'],
                features=features,
                # target=self.train_data,
                is_predict=False,
            )
        explainer = explain_pipeline(pipeline=pipeline, data=data, method=method, visualize=visualize, **kwargs)

        return explainer

    def _init_remote_if_necessary(self):
        remote = RemoteEvaluator()
        if remote.use_remote and remote.remote_task_params is not None:
            task = self.composer_dict['task']
            if task.task_type == TaskTypesEnum.ts_forecasting:
                task_str = \
                    f'Task(TaskTypesEnum.ts_forecasting, ' \
                    f'TsForecastingParams(forecast_length={task.task_params.forecast_length}))'
            else:
                task_str = f'Task({str(task.task_type)})'
            remote.remote_task_params.task_type = task_str
            remote.remote_task_params.is_multi_modal = isinstance(self.train_data, MultiModalData)

            if isinstance(self.target_name, str):
                remote.remote_task_params.target = self.target_name
