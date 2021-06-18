from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

import numpy as np

from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.tuner_validation import in_sample_ts_validation, \
    fit_predict_one_fold, ts_cross_validation


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    :param pipeline: pipeline to optimize
    :param task: task (classification, regression, ts_forecasting, clustering)
    :param iterations: max number of iterations
    """

    def __init__(self, pipeline, task, iterations=100,
                 timeout: timedelta = timedelta(minutes=5),
                 log: Log = None):
        self.pipeline = pipeline
        self.task = task
        self.iterations = iterations
        self.max_seconds = int(timeout.seconds)
        self.init_pipeline = None
        self.init_metric = None
        self.is_need_to_maximize = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def tune_pipeline(self, input_data, loss_function, loss_params=None):
        """
        Function for hyperparameters tuning on the pipeline

        :param input_data: data used for hyperparameter searching
        :param loss_function: function to minimize (or maximize) the metric,
        such function should take vector with true values as first values and
        predicted array as the second
        :param loss_params: dictionary with parameters for loss function
        :return fitted_pipeline: pipeline with optimized hyperparameters
        """
        raise NotImplementedError()

    @staticmethod
    def get_metric_value(data, pipeline, loss_function, loss_params):
        """
        Method calculates metric for algorithm validation

        :param data: InputData for validation
        :param pipeline: pipeline to validate
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function

        :return : value of loss function
        """

        # Make prediction
        if data.task.task_type == TaskTypesEnum.classification:
            test_target, preds = fit_predict_one_fold(pipeline, data)
        elif data.task.task_type == TaskTypesEnum.ts_forecasting:
            # For time series forecasting task in-sample forecasting is provided
            test_target, preds = ts_cross_validation(pipeline, data)
        else:
            test_target, preds = fit_predict_one_fold(pipeline, data)
            # Convert predictions into one dimensional array
            preds = np.ravel(np.array(preds))
            test_target = np.ravel(test_target)

        # Calculate metric
        if loss_params is None:
            metric = loss_function(test_target, preds)
        else:
            metric = loss_function(test_target, preds, **loss_params)
        return metric

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
                                                 pipeline=self.init_chain,
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
            init_metric = self.init_metric - deviation
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


def _greater_is_better(target, loss_function, loss_params) -> bool:
    """ Function checks is metric (loss function) need to be minimized or
    maximized

    :param target: target for define what problem is solving (max or min)
    :param loss_function: loss function
    :param loss_params: parameters for loss function

    :return : bool value is it good to maximize metric or not
    """

    if loss_params is None:
        metric = loss_function(target, target)
    else:
        try:
            metric = loss_function(target, target, **loss_params)
        except Exception:
            # Multiclass classification task
            metric = 1
    if int(round(metric)) == 0:
        return False
    else:
        return True
