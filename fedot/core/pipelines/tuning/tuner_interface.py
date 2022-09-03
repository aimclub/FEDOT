import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import Callable, ClassVar

import numpy as np
from hyperopt.early_stop import no_progress_loss

from fedot.core.data.data import InputData, OutputData, data_type_is_ts
from fedot.core.log import default_log
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.tune.cv_prediction import (
    calculate_loss_function,
    cv_tabular_predictions,
    cv_time_series_predictions
)
from fedot.core.validation.tune.simple import fit_predict_one_fold

MAX_METRIC_VALUE = sys.maxsize


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    pipeline: pipeline to optimize
    task: task (classification, regression, ts_forecasting, clustering)
    iterations: max number of iterations
    search_space: SearchSpace instance
    algo: algorithm for hyperparameters optimization with signature similar to hyperopt.tse.suggest
    """

    def __init__(self, pipeline, task,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = None,
                 n_jobs: int = -1):
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
        self.n_jobs = n_jobs

        self.log = default_log(self)

    @abstractmethod
    def tune_pipeline(self, input_data, loss_function,
                      cv_folds: int = None, validation_blocks: int = None):
        """Function for hyperparameters tuning on the pipeline

        Args:
            input_data: data used for hyperparameter searching
            loss_function: loss function to minimize (or maximize) during pipeline tuning,
                such function should take :obj:`InputData` with true values as the first input and
                :obj:`OutputData` with model prediction as the second one
            cv_folds: number of cross-validation folds
            validation_blocks: number of validation blocks for time series forecasting

        Returns:
            Pipeline: pipeline with optimized hyperparameters
        """

        raise NotImplementedError()

    def get_metric_value(self, data, pipeline, loss_function):
        """Method calculates metric for algorithm validation

        Args:
            data: :obj:`InputData` for validation
            pipeline: :obj:`Pipeline` to validate
            loss_function: function to minimize (or maximize)

        Returns:
            value of loss function
        """

        try:
            if self.cv_folds is None:
                metric_value = self._one_fold_validation(data, pipeline, loss_function)

            else:
                metric_value = self._cross_validation(data, pipeline, loss_function)
        except Exception as ex:
            self.log.debug(f'Tuning metric evaluation warning: {ex}. Continue.')
            # Return default metric: too small (for maximization) or too big (for minimization)
            return self._default_metric_value
        return metric_value

    def init_check(self, data, loss_function) -> None:
        """Method get metric on validation set before start optimization

        Args:
            data: InputData for validation
            loss_function: function to minimize (or maximize)
        """

        self.log.info('Hyperparameters optimization start')

        # Train pipeline
        self.init_pipeline = deepcopy(self.pipeline)

        self.init_metric = self.get_metric_value(data=data,
                                                 pipeline=self.init_pipeline,
                                                 loss_function=loss_function)

    def final_check(self, data, tuned_pipeline, loss_function):
        """Method propose final quality check after optimization process

        Args:
            data: :obj:`InputData` for validation
            tuned_pipeline: tuned pipeline
            loss_function: function to minimize (or maximize)
        """

        self.obtained_metric = self.get_metric_value(data=data,
                                                     pipeline=tuned_pipeline,
                                                     loss_function=loss_function)

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
    def _one_fold_validation(data: InputData, pipeline, loss_function: Callable):
        """Perform simple (hold-out) validation
        """

        if data.task.task_type is TaskTypesEnum.classification:
            predict_input, predicted_output = fit_predict_one_fold(data, pipeline)
        else:
            # For regression and time series forecasting
            predict_input, predicted_output = fit_predict_one_fold(data, pipeline)

        metric_value = calculate_loss_function(loss_function, predict_input, predicted_output)
        return metric_value

    def _cross_validation(self, data, pipeline, loss_function: Callable):
        """Perform cross validation for metric evaluation
        """

        if data.data_type is DataTypesEnum.table or data.data_type is DataTypesEnum.text or \
                data.data_type is DataTypesEnum.image:
            metric_value = cv_tabular_predictions(pipeline, data,
                                                  cv_folds=self.cv_folds, loss_function=loss_function)

        elif data_type_is_ts(data):
            if self.validation_blocks is None:
                self.log.info('For ts cross validation validation_blocks number was changed from None to 3 blocks')
                self.validation_blocks = 3

            metric_value = cv_time_series_predictions(pipeline, data, log=self.log,
                                                      cv_folds=self.cv_folds,
                                                      validation_blocks=self.validation_blocks,
                                                      loss_function=loss_function)
        return metric_value

    @property
    def _default_metric_value(self):
        if self.is_need_to_maximize:
            return -MAX_METRIC_VALUE
        else:
            return MAX_METRIC_VALUE


def _create_multi_target_prediction(output_data: OutputData):
    """Function creates an array of shape (target len, num classes)
    with classes probabilities from target values

    Args:
        output_data: define what problem is solving (max or min)

    Returns:
        :obj:`2d-array`: classes probability
    """

    target = output_data.predict
    nb_classes = len(np.unique(target))

    assert np.issubdtype(target.dtype, np.integer), 'Impossible to create multi target array from non integers'
    multi_target = np.eye(nb_classes)[target.ravel()]

    return multi_target


def _greater_is_better(loss_function) -> bool:
    """Function checks is metric (loss function) need to be minimized or maximized

    Args:
        loss_function: loss function

    Returns:
        bool: value is it good to maximize metric or not
    """

    ground_truth = InputData(target=np.array([[0], [1]]), features=[[1], [1]], data_type=DataTypesEnum.table,
                             idx=[0, 1], task=Task(TaskTypesEnum.classification))
    precise_prediction = OutputData(predict=np.array([[0], [1]]), features=[[1], [1]], idx=[0, 1],
                                    data_type=DataTypesEnum.table,
                                    task=Task(TaskTypesEnum.classification))
    approximate_prediction = OutputData(predict=np.array([[0], [0]]), features=[[1], [1]], idx=[0, 1],
                                        data_type=DataTypesEnum.table,
                                        task=Task(TaskTypesEnum.classification))

    try:
        optimal_metric, non_optimal_metric = [
            loss_function(ground_truth, score) for score in [precise_prediction, approximate_prediction]]
    except Exception:
        multiclass_precise_pred, multiclass_approximate_pred = [
            _create_multi_target_prediction(score) for score in [precise_prediction, approximate_prediction]]

        optimal_metric, non_optimal_metric = [
            loss_function(ground_truth, score)
            for score in [multiclass_precise_pred, multiclass_approximate_pred]
        ]

    return optimal_metric > non_optimal_metric
