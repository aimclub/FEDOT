import random
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from deap import tools

from fedot.api.api_utils import (array_to_input_data, compose_fedot_model,
                                 filter_operations_by_preset, composer_metrics_mapping,
                                 save_predict)
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.composer.optimisers.utils.pareto import ParetoFront
from fedot.core.data.data import InputData
from fedot.core.data.visualisation import plot_forecast
from fedot.core.log import default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import Task, TaskParams, TaskTypesEnum
from fedot.core.repository.tasks import TsForecastingParams

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


def default_evo_params():
    """ Dictionary with default parameters for composer """
    params = {'max_depth': 2,
              'max_arity': 3,
              'pop_size': 20,
              'num_of_generations': 20,
              'learning_time': 2,
              'preset': 'light_tun'}
    return params


default_test_metric_dict = {
    'regression': ['rmse', 'mae'],
    'classification': ['roc_auc', 'f1'],
    'multiclassification': 'f1',
    'clustering': 'silhouette',
    'ts_forecasting': ['rmse', 'mae']
}


class Fedot:
    """
    Main class for FEDOT API

    :param problem: the name of modelling problem to solve:
        - classification
        - regression
        - ts_forecasting
        - clustering
    :param preset: name of preset for model building (e.g. 'light', 'ultra-light')
    :param learning_time: time for model design (in minutes)
    :param composer_params: parameters of pipeline optimisation
        The possible parameters are:
            'max_depth' - max depth of the chain
            'max_arity' - max arity of the chain
            'pop_size' - population size for composer
            'num_of_generations' - number of generations for composer
            'learning_time':- composing time (minutes)
            'available_operations' - list of model names to use
            'with_tuning' - allow huperparameters tuning for the model
    :param task_params:  additional parameters of the task
    :param seed: value for fixed random seed
    :param verbose_level: level of the output detailing
        (-1 - nothing, 0 - errors, 1 - messages,
        2 - warnings and info, 3-4 - basic and detailed debug)
    """

    def __init__(self,
                 problem: str,
                 preset: str = None,
                 learning_time: Optional[float] = None,
                 composer_params: dict = None,
                 task_params: TaskParams = None,
                 seed=None, verbose_level: int = 1):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # metainfo
        self.problem = problem
        self.composer_params = composer_params
        self.task_params = task_params

        # model to use
        self.current_model = None

        # best models for multi-objective case
        self.best_models = None

        # datasets
        self.train_data = None
        self.test_data = None
        self.prediction = None
        self.prediction_labels = None  # classification-only
        self.target_name = None

        self.log = default_log('FEDOT logger', verbose_level=verbose_level)

        if self.composer_params is None:
            self.composer_params = default_evo_params()
        else:
            self.composer_params = {**default_evo_params(), **self.composer_params}

        self.metric_to_compose = None
        if 'metric' in self.composer_params:
            self.composer_params['composer_metric'] = self.composer_params['metric']
            del self.composer_params['metric']
            self.metric_to_compose = self.composer_params['composer_metric']

        if learning_time is not None:
            self.composer_params['learning_time'] = learning_time
            self.composer_params['num_of_generations'] = 10000  # num of generation is limited by time now

        if self.problem == 'ts_forecasting' and task_params is None:
            self.task_params = TsForecastingParams(forecast_length=30)

        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=self.task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=self.task_params),
                     'clustering': Task(TaskTypesEnum.clustering, task_params=self.task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=self.task_params)
                     }

        if self.problem == 'clustering':
            raise ValueError('This type of task is not not supported in API now')

        self.metric_name = default_test_metric_dict[self.problem]
        self.problem = task_dict[self.problem]

        if preset is None and 'preset' in self.composer_params:
            preset = self.composer_params['preset']

        if 'preset' in self.composer_params:
            del self.composer_params['preset']

        if preset is not None:
            available_operations = filter_operations_by_preset(self.problem, preset)
            self.composer_params['available_operations'] = available_operations
            self.composer_params['with_tuning'] = '_tun' in preset or preset is None

    def _get_params(self):
        param_dict = {'train_data': self.train_data,
                      'task': self.problem,
                      'logger': self.log
                      }
        return {**param_dict, **self.composer_params}

    def _obtain_model(self, is_composing_required: bool = True):
        execution_params = self._get_params()
        if is_composing_required:
            self.current_model, self.best_models = compose_fedot_model(**execution_params)

        if isinstance(self.best_models, tools.ParetoFront):
            self.best_models.__class__ = ParetoFront
            self.best_models.objective_names = self.metric_to_compose

        self.current_model.fit_from_scratch(self.train_data)

        return self.current_model

    def clean(self):
        """
        Cleans fitted model and obtained predictions
        """
        self.prediction = None
        self.prediction_labels = None
        self.current_model = None

    def fit(self,
            features: Union[str, np.ndarray, pd.DataFrame, InputData],
            target: Union[str, np.ndarray, pd.Series] = 'target',
            predefined_model: Union[str, Chain] = None):
        """
        Fit the chain with a predefined structure or compose and fit the new chain

        :param features: the array with features of train data
        :param target: the array with target values of train data
        :param predefined_model: the name of the atomic model or Chain instance
        :return: Chain object
        """

        self.target_name = target
        self.train_data = _define_data(ml_task=self.problem,
                                       features=features,
                                       target=target,
                                       is_predict=False)

        is_composing_required = True
        if self.current_model is not None:
            is_composing_required = False

        if predefined_model is not None:
            is_composing_required = False
            if isinstance(predefined_model, Chain):
                self.current_model = predefined_model
            elif isinstance(predefined_model, str):
                self.current_model = Chain(PrimaryNode(predefined_model))
            else:
                raise ValueError(f'{type(predefined_model)} is not supported as Fedot model')

        return self._obtain_model(is_composing_required)

    def predict(self,
                features: Union[str, np.ndarray, pd.DataFrame, InputData],
                save_predictions: bool = False):
        """
        Predict new target using already fitted model

        :param features: the array with features of test data
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :return: the array with prediction values
        """
        if self.current_model is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        self.test_data = _define_data(ml_task=self.problem, target=self.target_name,
                                      features=features, is_predict=True)

        if self.problem.task_type == TaskTypesEnum.classification:
            self.prediction_labels = self.current_model.predict(self.test_data, output_mode='labels')
            self.prediction = self.current_model.predict(self.test_data, output_mode='probs')
            output_prediction = self.prediction
        elif self.problem.task_type == TaskTypesEnum.ts_forecasting:
            # Convert forecast into one-dimensional array
            self.prediction = self.current_model.predict(self.test_data)
            forecast = np.ravel(np.array(self.prediction.predict))
            self.prediction.predict = forecast
            output_prediction = self.prediction
        else:
            self.prediction = self.current_model.predict(self.test_data)
            output_prediction = self.prediction

        if save_predictions:
            save_predict(self.prediction)
        return output_prediction.predict

    def predict_proba(self,
                      features: Union[str, np.ndarray, pd.DataFrame, InputData],
                      save_predictions: bool = False,
                      probs_for_all_classes: bool = False):
        """
        Predict the probability of new target using already fitted classification model

        :param features: the array with features of test data
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :param probs_for_all_classes: return probability for each class even for binary case
        :return: the array with prediction values
        """

        if self.current_model is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.problem.task_type == TaskTypesEnum.classification:
            self.test_data = _define_data(ml_task=self.problem, target=self.target_name,
                                          features=features, is_predict=True)

            mode = 'full_probs' if probs_for_all_classes else 'probs'

            self.prediction = self.current_model.predict(self.test_data, output_mode=mode)
            self.prediction_labels = self.current_model.predict(self.test_data, output_mode='labels')

            if save_predictions:
                save_predict(self.prediction)
        else:
            raise ValueError('Probabilities of predictions are available only for classification')

        return self.prediction.predict

    def forecast(self,
                 pre_history: Union[str, Tuple[np.ndarray, np.ndarray], InputData],
                 forecast_length: int = 1,
                 save_predictions: bool = False):
        """
        Forecast the new values of time series

        :param pre_history: the array with features for pre-history of the forecast
        :param forecast_length: num of steps to forecast
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :return: the array with prediction values
        """

        if self.current_model is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.problem.task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

        self.problem = self.train_data.task

        self.test_data = _define_data(ml_task=self.problem,
                                      target=self.target_name,
                                      features=pre_history,
                                      is_predict=True)

        self.current_model = Chain(self.current_model.root_node)

        self.prediction = self.current_model.predict(self.test_data)

        if save_predictions:
            save_predict(self.prediction)
        return self.prediction.predict

    def load(self, path):
        """
        Load saved chain from disk

        :param path to json file with model
        """
        self.current_model.load(path)

    def plot_prediction(self):
        """
        Plot the prediction obtained from chain
        """
        if self.prediction is not None:
            if self.problem.task_type == TaskTypesEnum.ts_forecasting:
                plot_forecast(pre_history=self.train_data, forecast=self.prediction)
            else:
                # TODO implement other visualizations
                self.log.error('Not supported yet')

        else:
            self.log.error('No prediction to visualize')

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = None) -> dict:
        """
        Get quality metrics for the fitted chain

        :param target: the array with target values of test data
        :param metric_names: the names of required metrics
        :return: the values of quality metrics
        """
        if metric_names is None:
            metric_names = self.metric_name

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
            if composer_metrics_mapping[metric_name] is NotImplemented:
                self.log.warn(f'{metric_name} is not available as metric')
            else:
                prediction = self.prediction
                metric_cls = MetricsRepository().metric_class_by_id(composer_metrics_mapping[metric_name])
                if metric_cls.output_mode == 'labels':
                    prediction = self.prediction_labels
                if self.problem.task_type == TaskTypesEnum.ts_forecasting:
                    real.target = real.target[~np.isnan(prediction.predict)]
                    prediction.predict = prediction.predict[~np.isnan(prediction.predict)]

                metric_value = abs(metric_cls.metric(reference=real,
                                                     predicted=prediction))
                calculated_metrics[metric_name] = metric_value

        return calculated_metrics


def _define_data(ml_task: Task,
                 features: Union[str, np.ndarray, pd.DataFrame, InputData],
                 target: Union[str, np.ndarray, pd.Series] = None,
                 is_predict=False):
    if type(features) == InputData:
        # native FEDOT format for input data
        data = features
        data.task = ml_task
    elif type(features) == pd.DataFrame:
        # pandas format for input data
        if target is None:
            target = np.array([])

        if isinstance(target, str) and target in features.columns:
            target_array = features[target]
            del features[target]
        else:
            target_array = target

        data = array_to_input_data(features_array=np.asarray(features),
                                   target_array=np.asarray(target_array),
                                   task=ml_task)
    elif type(features) == np.ndarray:
        # numpy format for input data
        if target is None:
            target = np.array([])

        if isinstance(target, str):
            target_array = features[target]
            del features[target]
        else:
            target_array = target

        data = array_to_input_data(features_array=features,
                                   target_array=target_array,
                                   task=ml_task)
    elif type(features) == tuple:
        data = array_to_input_data(features_array=features[0],
                                   target_array=features[1],
                                   task=ml_task)
    elif type(features) == str:
        # CSV files as input data, by default - table data
        if target is None:
            target = 'target'

        data_type = DataTypesEnum.table
        if ml_task.task_type == TaskTypesEnum.ts_forecasting:
            # For time series forecasting format - time series
            data = InputData.from_csv_time_series(task=ml_task,
                                                  file_path=features,
                                                  target_column=target,
                                                  is_predict=is_predict)
        else:
            # Make default features table
            # CSV files as input data
            if target is None:
                target = 'target'
            data = InputData.from_csv(features, task=ml_task,
                                      target_column=target,
                                      data_type=data_type)
    else:
        raise ValueError('Please specify a features as path to csv file or as Numpy array')

    return data
