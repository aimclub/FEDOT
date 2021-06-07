from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fedot.api.api_utils.api_utils import Api_facade
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.data.visualisation import plot_forecast

from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import TaskParams, TaskTypesEnum

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
            'cv_folds' - number of folds for cross-validation
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
                 seed=None, verbose_level: int = 0):

        self.helper = Api_facade(**{'problem': problem,
                                    'preset': preset,
                                    'learning_time': learning_time,
                                    'composer_params': composer_params,
                                    'task_params': task_params,
                                    'seed': seed,
                                    'verbose_level': verbose_level})

        self.composer_dict = self.helper.initialize_params()
        self.composer_dict['current_model'] = None
        self.task_metrics, self.composer_metrics, \
        self.tuner_metrics = self.helper.get_metrics_for_task(self.composer_dict['problem'],
                                                              self.composer_dict['metric_name'])
        self.composer_dict['tuner_metric'] = self.tuner_metrics

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
        self.train_data = self.helper.define_data(ml_task=self.composer_dict['task'],
                                                  features=features,
                                                  target=target,
                                                  is_predict=False)

        is_composing_required = True
        if self.composer_dict['current_model'] is not None:
            is_composing_required = False

        if predefined_model is not None:
            is_composing_required = False
            if isinstance(predefined_model, Chain):
                self.current_model = predefined_model
            elif isinstance(predefined_model, str):
                self.current_model = Chain(PrimaryNode(predefined_model))
            else:
                raise ValueError(f'{type(predefined_model)} is not supported as Fedot model')
            self.composer_dict['current_model'] = self.current_model

        self.composer_dict['is_composing_required'] = is_composing_required
        self.composer_dict['train_data'] = self.train_data
        self.current_model, self.best_models, self.history = self.helper.obtain_model(**self.composer_dict)
        return self.current_model

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

        self.test_data = self.helper.define_data(ml_task=self.composer_dict['task'], target=self.target_name,
                                                 features=features, is_predict=True)

        if self.composer_dict['task'].task_type == TaskTypesEnum.classification:
            self.prediction_labels = self.current_model.predict(self.test_data, output_mode='labels')
            self.prediction = self.current_model.predict(self.test_data, output_mode='probs')
            output_prediction = self.prediction
        elif self.composer_dict['task'].task_type == TaskTypesEnum.ts_forecasting:
            # Convert forecast into one-dimensional array
            self.prediction = self.current_model.predict(self.test_data)
            forecast = np.ravel(np.array(self.prediction.predict))
            self.prediction.predict = forecast
            output_prediction = self.prediction
        else:
            self.prediction = self.current_model.predict(self.test_data)
            output_prediction = self.prediction

        if save_predictions:
            self.helper.save_predict(self.prediction)
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

        if self.composer_dict['task'].task_type == TaskTypesEnum.classification:
            self.test_data = self.helper.define_data(ml_task=self.composer_dict['task'], target=self.target_name,
                                                     features=features, is_predict=True)

            mode = 'full_probs' if probs_for_all_classes else 'probs'

            self.prediction = self.current_model.predict(self.test_data, output_mode=mode)
            self.prediction_labels = self.current_model.predict(self.test_data, output_mode='labels')

            if save_predictions:
                self.helper.save_predict(self.prediction)
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

        if self.composer_dict['current_model'] is None:
            raise ValueError(NOT_FITTED_ERR_MSG)

        if self.composer_dict['task'].task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

        self.problem = self.train_data.task

        self.test_data = self.helper.define_data(ml_task=self.composer_dict['task'],
                                                 target=self.target_name,
                                                 features=pre_history,
                                                 is_predict=True)

        self.current_model = Chain(self.current_model.root_node)

        self.prediction = self.current_model.predict(self.test_data)

        if save_predictions:
            self.helper.save_predict(self.prediction)
        return self.prediction.predict

    def load(self, path):
        """
        Load saved chain from disk

        :param path to json file with model
        """
        self.composer_dict['current_model'].load(path)

    def clean(self):
        """
        Cleans fitted model and obtained predictions
        """
        self.prediction = None
        self.prediction_labels = None
        self.composer_dict['current_model'] = None

    def plot_prediction(self):
        """
        Plot the prediction obtained from chain
        """
        if self.prediction is not None:
            if self.composer_dict['task'].task_type == TaskTypesEnum.ts_forecasting:
                plot_forecast(pre_history=self.train_data, forecast=self.prediction)
            else:
                # TODO implement other visualizations
                self.composer_dict['log'].error('Not supported yet')

        else:
            self.composer_dict['log'].error('No prediction to visualize')

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
                self.composer_dict['log'].warn(f'{metric_name} is not available as metric')
            else:
                prediction = self.prediction
                metric_cls = MetricsRepository().metric_class_by_id(
                    self.helper.get_composer_metrics_mapping(metric_name))
                if metric_cls.output_mode == 'labels':
                    prediction = self.prediction_labels
                if self.composer_dict['task'].task_type == TaskTypesEnum.ts_forecasting:
                    real.target = real.target[~np.isnan(prediction.predict)]
                    prediction.predict = prediction.predict[~np.isnan(prediction.predict)]

                metric_value = abs(metric_cls.metric(reference=real,
                                                     predicted=prediction))
                calculated_metrics[metric_name] = metric_value

        return calculated_metrics
