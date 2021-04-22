import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from fedot.api.api_utils import (array_to_input_data, compose_fedot_model, filter_models_by_preset, metrics_mapping,
                                 save_predict)
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData
from fedot.core.data.visualisation import plot_forecast
from fedot.core.log import default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import (Task, TaskParams,
                                         TaskTypesEnum, TsForecastingParams)


def default_evo_params(problem):
    if problem == 'ts_forecasting':
        return {'max_depth': 1,
                'max_arity': 2,
                'pop_size': 20,
                'num_of_generations': 20,
                'learning_time': 2,
                'preset': 'light'}
    else:
        return {'max_depth': 2,
                'max_arity': 3,
                'pop_size': 20,
                'num_of_generations': 20,
                'learning_time': 2,
                'preset': 'light_tun'
                }


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
            'available_model_types' - list of model names to use
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
                 learning_time: int = 2,
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

        # datasets
        self.train_data = None
        self.test_data = None
        self.prediction = None
        self.prediction_labels = None  # classification-only

        self.log = default_log('FEDOT logger', verbose_level=verbose_level)

        if self.composer_params is None:
            self.composer_params = default_evo_params(self.problem)
        else:
            self.composer_params = {**default_evo_params(self.problem), **self.composer_params}

        if learning_time is not None:
            self.composer_params['learning_time'] = learning_time

        if self.problem == 'ts_forecasting' and task_params is None:
            # TODO auto-estimate
            self.task_params = TsForecastingParams(forecast_length=30,
                                                   max_window_size=30,
                                                   make_future_prediction=True)

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
            del self.composer_params['preset']

        if preset is not None:
            available_model_types = filter_models_by_preset(self.problem, preset)
            self.composer_params['available_model_types'] = available_model_types
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
            self.current_model = compose_fedot_model(**execution_params)

        self.current_model.fit_from_scratch(self.train_data)

        return self.current_model

    def _check_num_classes(self,
                           train_data: InputData):
        if len(np.unique(train_data.target)) > 2:
            self.metric_name = 'f1'

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
        self.train_data = _define_data(ml_task=self.problem,
                                       features=features,
                                       target=target)

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

        self._check_num_classes(self.train_data)

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
            raise ValueError('The model was not fitted yet')

        self.test_data = _define_data(ml_task=self.problem,
                                      features=features, is_predict=True)

        if self.problem.task_type == TaskTypesEnum.classification:
            self.prediction_labels = self.current_model.predict(self.test_data, output_mode='labels')
            self.prediction = self.current_model.predict(self.test_data, output_mode='probs')
        else:
            self.prediction = self.current_model.predict(self.test_data)

        if save_predictions:
            save_predict(self.prediction)
        return self.prediction.predict

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
            raise ValueError('The model was not fitted yet')

        if self.problem.task_type == TaskTypesEnum.classification:

            self.test_data = _define_data(ml_task=self.problem,
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
            raise ValueError('The model was not fitted yet')

        if self.problem.task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

        self.problem = self.train_data.task

        self.train_data = _define_data(ml_task=self.problem,
                                       features=pre_history, is_predict=True)

        self.current_model = TsForecastingChain(self.current_model.root_node)

        last_ind = int(round(self.train_data.idx[-1]))

        supp_data = InputData(idx=list(range(last_ind, last_ind + forecast_length)),
                              features=None, target=None,
                              data_type=DataTypesEnum.ts,
                              task=self.problem)

        self.prediction = self.current_model.forecast(initial_data=self.train_data, supplementary_data=supp_data)

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

        # TODO change to sklearn metrics
        if not isinstance(metric_names, List):
            metric_names = [metric_names]

        calculated_metrics = dict()
        for metric_name in metric_names:
            if metrics_mapping[metric_name] is NotImplemented:
                self.log.warn(f'{metric_name} is not available as metric')
            else:
                prediction = self.prediction
                metric_cls = MetricsRepository().metric_class_by_id(metrics_mapping[metric_name])
                if metric_cls.output_mode == 'labels':
                    prediction = self.prediction_labels
                metric_value = abs(metric_cls.metric(reference=self.test_data,
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
    elif type(features) == pd.DataFrame:
        # pandas format for input data
        if target is None:
            target = np.array([])

        data = array_to_input_data(features_array=np.asarray(features),
                                   target_array=np.asarray(target),
                                   task=ml_task)
    elif type(features) == np.ndarray:
        # numpy format for input data
        if target is None:
            target = np.array([])

        data = array_to_input_data(features_array=features,
                                   target_array=target,
                                   task=ml_task)
    elif type(features) == tuple:
        data = array_to_input_data(features_array=features[0],
                                   target_array=features[1],
                                   task=ml_task)
    elif type(features) == str:
        # CSV files as input data
        if target is None:
            target = 'target'
        elif is_predict:
            target = None
        data_type = DataTypesEnum.table
        if ml_task.task_type == TaskTypesEnum.ts_forecasting:
            data_type = DataTypesEnum.ts
        data = InputData.from_csv(features, task=ml_task, target_column=target, data_type=data_type)
    else:
        raise ValueError('Please specify a features as path to csv file or as Numpy array')

    return data
