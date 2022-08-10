import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import Callable, ClassVar

import numpy as np
from hyperopt.early_stop import no_progress_loss

from fedot.core.dag.graph import Graph
from fedot.core.data.data import OutputData, InputData
from fedot.core.log import default_log
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate, ObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task

MAX_METRIC_VALUE = sys.maxsize


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    :param task: task (classification, regression, ts_forecasting, clustering)
    :param iterations: max number of iterations
    :param search_space: SearchSpace instance
    :param algo: algorithm for hyperparameters optimization with signature similar to hyperopt.tse.suggest
    """

    def __init__(self, task,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = None,
                 n_jobs: int = -1):
        self.task = task
        self.iterations = iterations
        iteration_stop_count = early_stopping_rounds or max(100, int(np.sqrt(iterations) * 10))
        self.early_stop_fn = no_progress_loss(iteration_stop_count=iteration_stop_count)
        self.max_seconds = int(timeout.seconds) if timeout is not None else None
        self.init_pipeline = None
        self.init_metric = None
        self.obtained_metric = None
        self.is_need_to_maximize = None
        self.objective_evaluate = None
        self.search_space = search_space
        self.algo = algo
        self.n_jobs = n_jobs

        self.log = default_log(self)

    def get_metric_value(self, pipeline, objective_evaluate):
        """
        Method calculates metric for algorithm validation

        :param pipeline: pipeline to validate

        :return: value of loss function
        # """
        pipeline_fitness = objective_evaluate.evaluate(pipeline)
        metric_value = pipeline_fitness.value
        if not metric_value:
            return self._default_metric_value
        return metric_value

    def init_check(self, pipeline: Pipeline, objective_evaluate: PipelineObjectiveEvaluate) -> None:
        """
        Method get metric on validation set before start optimization

        :param pipeline: Pipeline to calculate objective
        :param objective_evaluate: ObjectiveEvaluate to evaluate the pipeline
        """
        self.log.info('Hyperparameters optimization start')

        # Train pipeline
        self.init_pipeline = deepcopy(pipeline)

        self.init_metric = self.get_metric_value(pipeline=self.init_pipeline,
                                                 objective_evaluate=objective_evaluate)

    def final_check(self, tuned_pipeline: Pipeline, objective_evaluate: PipelineObjectiveEvaluate):
        """
        Method propose final quality check after optimization process

        :param tuned_pipeline: Tuned pipeline to calculate objective
        :param objective_evaluate: ObjectiveEvaluate to evaluate the pipeline
        """

        self.obtained_metric = self.get_metric_value(pipeline=tuned_pipeline,
                                                     objective_evaluate=objective_evaluate)

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

    # TODO: refactor
    @staticmethod
    def _greater_is_better(loss_function) -> bool:
        """ Function checks is metric (loss function) need to be minimized or maximized

        :param loss_function: loss function

        :return: bool value is it good to maximize metric or not
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

    @property
    def _default_metric_value(self):
        if self.is_need_to_maximize:
            return -MAX_METRIC_VALUE
        else:
            return MAX_METRIC_VALUE


def _create_multi_target_prediction(output_data: OutputData):
    """ Function creates an array of shape (target len, num classes)
    with classes probabilities from target values

    :param output_data: output_data for define what problem is solving (max or min)

    :return: 2d-array of classes probabilities
    """
    target = output_data.predict
    nb_classes = len(np.unique(target))

    assert np.issubdtype(target.dtype, np.integer), 'Impossible to create multi target array from non integers'
    multi_target = np.eye(nb_classes)[target.ravel()]

    return multi_target
