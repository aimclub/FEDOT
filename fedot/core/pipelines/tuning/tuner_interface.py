import sys
from hyperopt.early_stop import no_progress_loss

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import Callable, ClassVar, Optional

import numpy as np

from fedot.core.data.data import data_type_is_ts
from fedot.core.log import Log, default_log
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.tune.simple import fit_predict_one_fold
from fedot.core.validation.tune.tabular import cv_tabular_predictions
from fedot.core.validation.tune.time_series import cv_time_series_predictions
from sklearn.preprocessing import LabelEncoder

MAX_METRIC_VALUE = sys.maxsize


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    :attribute pipeline: pipeline to optimize
    :attribute task: task (classification, regression, ts_forecasting, clustering)
    :attribute iterations: max number of iterations
    :attribute search_space: SearchSpace instance
    :attribute algo: algorithm for hyperparameters optimization with signature similar to hyperopt.tse.suggest
    """

    def __init__(self, pipeline, task,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = None):
        self.pipeline = pipeline
        self.task = task
        self.iterations = iterations
        iteration_stop_count = early_stopping_rounds or max(100, int(np.sqrt(iterations) * 10))
        self.early_stop_fn = no_progress_loss(iteration_stop_count=iteration_stop_count)
        self.max_seconds = int(timeout.seconds) if timeout is not None else None
        self.init_pipeline = None
        self.init_metric = None
        self.obtained_metric = None
        self.is_need_to_maximize = None
        self.cv_folds = None
        self.validation_blocks = None
        self.search_space = search_space
        self.algo = algo

        self.log = default_log(self.__class__.__name__)

    @abstractmethod
    def tune_pipeline(self, input_data, loss_function, loss_params=None,
                      cv_folds: int = None, validation_blocks: int = None):
        """
        Function for hyperparameters tuning on the pipeline

        :param input_data: data used for hyperparameter searching
        :param loss_function: function to minimize (or maximize) the metric,
        such function should take vector with true values as first values and
        predicted array as the second
        :param loss_params: dictionary with parameters for loss function
        :param cv_folds: number of folds for cross validation
        :param validation_blocks: number of validation blocks for time series forecasting

        :return fitted_pipeline: pipeline with optimized hyperparameters
        """
        raise NotImplementedError()

    def get_metric_value(self, data, pipeline, loss_function, loss_params):
        """
        Method calculates metric for algorithm validation

        :param data: InputData for validation
        :param pipeline: pipeline to validate
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function

        :return: value of loss function
        """

        try:
            if self.cv_folds is None:
                preds, test_target = self._one_fold_validation(data, pipeline)
            else:
                preds, test_target = self._cross_validation(data, pipeline)

            # Calculate metric
            metric_value = _calculate_loss_function(loss_function, loss_params, test_target, preds)
        except Exception as ex:
            self.log.debug(f'Tuning metric evaluation warning: {ex}. Continue.')
            # Return default metric: too small (for maximization) or too big (for minimization)
            return self._default_metric_value

        if self.is_need_to_maximize:
            return -metric_value
        else:
            return metric_value

    def init_check(self, data, loss_function, loss_params) -> None:
        """
        Method get metric on validation set before start optimization

        :param data: InputData for validation
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function
        """
        self.log.info('Hyperparameters optimization start')

        # Train pipeline
        self.init_pipeline = deepcopy(self.pipeline)

        self.init_metric = self.get_metric_value(data=data,
                                                 pipeline=self.init_pipeline,
                                                 loss_function=loss_function,
                                                 loss_params=loss_params)

    def final_check(self, data, tuned_pipeline, loss_function, loss_params):
        """
        Method propose final quality check after optimization process

        :param data: InputData for validation
        :param tuned_pipeline: tuned pipeline
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function
        """

        self.obtained_metric = self.get_metric_value(data=data,
                                                     pipeline=tuned_pipeline,
                                                     loss_function=loss_function,
                                                     loss_params=loss_params)

        if self.obtained_metric == self._default_metric_value:
            self.obtained_metric = None

        self.log.info('Hyperparameters optimization finished')

        prefix_tuned_phrase = 'Return tuned pipeline due to the fact that obtained metric'
        prefix_init_phrase = 'Return init pipeline due to the fact that obtained metric'

        # 5% deviation is acceptable
        deviation = (self.init_metric / 100.0) * 5

        if self.is_need_to_maximize:
            # Maximization
            init_metric = -1 * (self.init_metric - deviation)
            if self.obtained_metric is None:
                self.log.info(f'{prefix_init_phrase} is None. Initial metric is {init_metric:.3f}')
                return self.init_pipeline

            self.obtained_metric *= -1
            if self.obtained_metric >= init_metric:
                self.log.info(f'{prefix_tuned_phrase} {self.obtained_metric:.3f} equal or '
                              f'bigger than initial (- 5% deviation) {init_metric:.3f}')
                return tuned_pipeline
            else:
                self.log.info(f'{prefix_init_phrase} {self.obtained_metric:.3f} '
                              f'smaller than initial (- 5% deviation) {init_metric:.3f}')
                return self.init_pipeline
        else:
            # Minimization
            init_metric = self.init_metric + deviation
            if self.obtained_metric is None:
                self.log.info(f'{prefix_init_phrase} is None. Initial metric is {init_metric:.3f}')
                return self.init_pipeline
            elif self.obtained_metric <= init_metric:
                self.log.info(f'{prefix_tuned_phrase} {self.obtained_metric:.3f} equal or '
                              f'smaller than initial (+ 5% deviation) {init_metric:.3f}')
                return tuned_pipeline
            else:
                self.log.info(f'{prefix_init_phrase} {self.obtained_metric:.3f} '
                              f'bigger than initial (+ 5% deviation) {init_metric:.3f}')
                return self.init_pipeline

    @staticmethod
    def _one_fold_validation(data, pipeline):
        """ Perform simple (hold-out) validation """

        if data.task.task_type is TaskTypesEnum.classification:
            test_target, preds = fit_predict_one_fold(data, pipeline)
        else:
            # For regression and time series forecasting
            test_target, preds = fit_predict_one_fold(data, pipeline)
            # Convert predictions into one dimensional array
            preds = np.ravel(np.array(preds))
            test_target = np.ravel(test_target)

        return preds, test_target

    def _cross_validation(self, data, pipeline):
        """ Perform cross validation for metric evaluation """

        preds, test_target = [], []
        if data.data_type is DataTypesEnum.table or data.data_type is DataTypesEnum.text or \
                data.data_type is DataTypesEnum.image:
            preds, test_target = cv_tabular_predictions(pipeline, data,
                                                        cv_folds=self.cv_folds)

        elif data_type_is_ts(data):
            if self.validation_blocks is None:
                self.log.info('For ts cross validation validation_blocks number was changed from None to 3 blocks')
                self.validation_blocks = 3

            preds, test_target = cv_time_series_predictions(pipeline, data,
                                                            cv_folds=self.cv_folds,
                                                            validation_blocks=self.validation_blocks)
        return preds, test_target

    @property
    def _default_metric_value(self):
        if self.is_need_to_maximize:
            return -MAX_METRIC_VALUE
        else:
            return MAX_METRIC_VALUE


def _create_multi_target_prediction(target):
    """ Function creates an array of shape (target len, num classes)
    with classes probabilities from target values

    :param target: target for define what problem is solving (max or min)

    :return: 2d-array of classes probabilities
    """

    nb_classes = len(np.unique(target))

    assert np.issubdtype(target.dtype, np.integer), 'Impossible to create multi target array from non integers'
    multi_target = np.eye(nb_classes)[target.ravel()]

    return multi_target


def _greater_is_better(loss_function, loss_params) -> bool:
    """ Function checks is metric (loss function) need to be minimized or maximized

    :param loss_function: loss function
    :param loss_params: parameters for loss function

    :return: bool value is it good to maximize metric or not
    """

    ground_truth = np.array([[0], [1]])
    precise_prediction = np.array([[0], [1]])
    approximate_prediction = np.array([[0], [0]])

    if loss_params is None:
        loss_params = {}

    try:
        optimal_metric, non_optimal_metric = [
            loss_function(ground_truth, score, **loss_params) for score in [precise_prediction, approximate_prediction]]
    except Exception:
        multiclass_precise_pred, multiclass_approximate_pred = [
            _create_multi_target_prediction(score) for score in [precise_prediction, approximate_prediction]]

        optimal_metric, non_optimal_metric = [
            loss_function(ground_truth, score, **loss_params)
            for score in [multiclass_precise_pred, multiclass_approximate_pred]
        ]

    return optimal_metric > non_optimal_metric


def _calculate_loss_function(loss_function, loss_params, target, preds):
    """ Function processing preds and calculating metric (loss function)

    :param loss_function: loss function
    :param loss_params: parameters for loss function
    :param target: target for evaluation
    :param preds: prediction for evaluation

    :return: calculated loss_function
    """

    if loss_params is None:
        loss_params = {}
    try:
        # actual for regression and classification metrics that requires all classes probabilities
        metric_value = loss_function(target, preds, **loss_params)
    except ValueError:
        try:
            # transform 1st class probability to assigned class, actual for accuracy-like metrics with binary
            metric_value = loss_function(target, preds.round(), **loss_params)
        except ValueError:
            # transform class probabilities to assigned class, actual for accuracy-like metrics with multiclass
            metric_value = loss_function(target, np.argmax(preds, axis=1), **loss_params)

    return metric_value
