from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

import numpy as np

from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum


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
    def get_metric_value(train_input, predict_input, test_target,
                         pipeline, loss_function, loss_params):
        """
        Method calculates metric for algorithm validation

        :param train_input: data for train pipeline
        :param predict_input: data for prediction
        :param test_target: target array for validation
        :param pipeline: pipeline to process
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function

        :return : value of loss function
        """

        pipeline.fit_from_scratch(train_input)

        # Make prediction
        if train_input.task.task_type == TaskTypesEnum.classification:
            predicted_labels = pipeline.predict(predict_input)
            preds = np.array(predicted_labels.predict)
        else:
            predicted_values = pipeline.predict(predict_input)
            preds = np.ravel(np.array(predicted_values.predict))
            test_target = np.ravel(test_target)

        # Calculate metric
        if loss_params is None:
            metric = loss_function(test_target, preds)
        else:
            metric = loss_function(test_target, preds, **loss_params)
        return metric

    def init_check(self, train_input, predict_input,
                   test_target, loss_function, loss_params) -> None:
        """
        Method get metric on validation set before start optimization

        :param train_input: data for train pipeline
        :param predict_input: data for prediction
        :param test_target: target array for validation
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function
        """
        self.log.info('Hyperparameters optimization start')

        # Train pipeline
        self.init_pipeline = deepcopy(self.pipeline)

        self.init_metric = self.get_metric_value(train_input=train_input,
                                                 predict_input=predict_input,
                                                 test_target=test_target,
                                                 pipeline=self.init_pipeline,
                                                 loss_function=loss_function,
                                                 loss_params=loss_params)

    def final_check(self, train_input, predict_input, test_target,
                    tuned_pipeline, loss_function, loss_params):
        """
        Method propose final quality check after optimization process

        :param train_input: data for train pipeline
        :param predict_input: data for prediction
        :param test_target: target array for validation
        :param tuned_pipeline: tuned pipeline
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function
        """

        obtained_metric = self.get_metric_value(train_input=train_input,
                                                predict_input=predict_input,
                                                test_target=test_target,
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

    :param target: array with target
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
