from abc import ABC, abstractmethod
from typing import Callable, ClassVar
from copy import deepcopy, copy
from datetime import timedelta

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.validation.tune.time_series import cv_time_series_predictions
from fedot.core.validation.tune.tabular import cv_tabular_predictions
from fedot.core.validation.tune.simple import fit_predict_one_fold
from fedot.core.pipelines.tuning.search_space import SearchSpace

MAX_METRIC_VALUE = 10e6


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    :attribute pipeline: pipeline to optimize
    :attribute task: task (classification, regression, ts_forecasting, clustering)
    :attribute iterations: max number of iterations
    :attribute search_space: SearchSpace instance
    :attribute algo: algorithm for hyperparameters optimization with signature similar to hyperopt.tse.suggest
    """

    def __init__(self, pipeline, task, iterations=100,
                 timeout: timedelta = timedelta(minutes=5),
                 log: Log = None,
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = None):
        self.pipeline = pipeline
        self.task = task
        self.iterations = iterations
        self.max_seconds = int(timeout.seconds) if timeout is not None else None
        self.init_pipeline = None
        self.init_metric = None
        self.is_need_to_maximize = None
        self.cv_folds = None
        self.validation_blocks = None
        self.search_space = search_space
        self.algo = algo

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

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

        :return : value of loss function
        """

        try:
            if self.cv_folds is None:
                preds, test_target = self._one_fold_validation(data, pipeline)
            else:
                preds, test_target = self._cross_validation(data, pipeline)

            # Calculate metric
            if loss_params is None:
                metric_value = loss_function(test_target, preds)
            else:
                metric_value = loss_function(test_target, preds, **loss_params)
        except Exception as ex:
            self.log.debug(f'Tuning metric evaluation warning: {ex}. Continue.')
            # Return default metric: too small (for maximization) or too big (for minimization)
            metric_value = self._default_metric_value

        if self.is_need_to_maximize is True:
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

        obtained_metric = self.get_metric_value(data=data,
                                                pipeline=tuned_pipeline,
                                                loss_function=loss_function,
                                                loss_params=loss_params)

        self.log.info('Hyperparameters optimization finished')

        prefix_tuned_phrase = 'Return tuned pipeline due to the fact that obtained metric'
        prefix_init_phrase = 'Return init pipeline due to the fact that obtained metric'

        # 5% deviation is acceptable
        deviation = (self.init_metric / 100.0) * 5

        if self.is_need_to_maximize is True:
            # Maximization
            init_metric = -1 * (self.init_metric - deviation)
            obtained_metric = -1 * obtained_metric
            if obtained_metric >= init_metric:
                self.log.info(f'{prefix_tuned_phrase} {obtained_metric:.3f} equal or '
                              f'bigger than initial (- 5% deviation) {init_metric:.3f}')
                return tuned_pipeline
            else:
                self.log.info(f'{prefix_init_phrase} {obtained_metric:.3f} '
                              f'smaller than initial (- 5% deviation) {init_metric:.3f}')
                return self.init_pipeline
        else:
            # Minimization
            init_metric = self.init_metric + deviation
            if obtained_metric <= init_metric:
                self.log.info(f'{prefix_tuned_phrase} {obtained_metric:.3f} equal or '
                              f'smaller than initial (+ 5% deviation) {init_metric:.3f}')
                return tuned_pipeline
            else:
                self.log.info(f'{prefix_init_phrase} {obtained_metric:.3f} '
                              f'bigger than initial (+ 5% deviation) {init_metric:.3f}')
                return self.init_pipeline

    @staticmethod
    def _one_fold_validation(data, pipeline):
        """ Perform simple (hold-out) validation """

        if data.task.task_type == TaskTypesEnum.classification:
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

        if data.data_type is DataTypesEnum.table or data.data_type is DataTypesEnum.text or \
                data.data_type is DataTypesEnum.image:
            preds, test_target = cv_tabular_predictions(pipeline, data,
                                                        cv_folds=self.cv_folds)

        elif data.data_type is DataTypesEnum.ts:
            if self.validation_blocks is None:
                self.log.info('For ts cross validation validation_blocks number was changed from None to 3 blocks')
                self.validation_blocks = 3

            preds, test_target = cv_time_series_predictions(pipeline, data, log=self.log,
                                                            cv_folds=self.cv_folds,
                                                            validation_blocks=self.validation_blocks)
        return preds, test_target

    @property
    def _default_metric_value(self):
        if self.is_need_to_maximize is True:
            return -MAX_METRIC_VALUE
        else:
            return MAX_METRIC_VALUE


def _create_multi_target_prediction(target, optimal=True):
    """ Function creates an array of shape (target len, num classes)
    with classes probabilities from target values, used in _greater_is_better

    :param target: target for define what problem is solving (max or min)
    :param optimal: whether return optimal probabilities or not

    :return : 2d-array of classes probabilities
    """

    len_target = target.shape[0]

    if optimal:
        multi_target = csr_matrix((np.ones(len_target), (np.arange(len_target),
                                                         target.reshape((len_target,))))).A
    else:
        multi_target = np.zeros((len_target, len(np.unique(target))))
        multi_target[:, 0] = 1

    return multi_target


def _convert_target_dimension(target):
    """ Function check number of unique classes and converted target
    for multiclass metrics

    :param target: target for define what problem is solving (max or min)

    :return : 2d-array of classes probabilities
    """

    nb_classes = len(np.unique(target))

    if nb_classes > 2:
        target_converted = target.reshape(-1).tolist()
        target_converted = [int(x) for x in target_converted]
        if min(target_converted) == 1:
            target_converted = [x - 1 for x in target_converted]
        target = np.eye(nb_classes)[target_converted]

    return target


def _greater_is_better(target, loss_function, loss_params, data_type) -> bool:
    """ Function checks is metric (loss function) need to be minimized or
    maximized

    :param target: target for define what problem is solving (max or min)
    :param loss_function: loss function
    :param loss_params: parameters for loss function

    :return : bool value is it good to maximize metric or not
    """

    if isinstance(target[0], str):
        # Target for classification contain string objects
        le = LabelEncoder()
        target = le.fit_transform(target)

    if loss_params is None:
        loss_params = {}

    if data_type is not DataTypesEnum.ts or DataTypesEnum.text:
        try:
            target = _convert_target_dimension(target)
        except Exception:
            target = target

    try:
        optimal_metric = loss_function(target, target, **loss_params)
        not_optimal_metric = loss_function(target, np.zeros_like(target), **loss_params)
    except Exception:
        optimal_multi_target = _create_multi_target_prediction(target, True)
        not_optimal_multi_target = _create_multi_target_prediction(target, False)

        optimal_metric = loss_function(target, optimal_multi_target, **loss_params)
        not_optimal_metric = loss_function(target, not_optimal_multi_target, **loss_params)

    if optimal_metric > not_optimal_metric:
        return True
    else:
        return False
