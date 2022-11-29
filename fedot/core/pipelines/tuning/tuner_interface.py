from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import Callable, ClassVar

import numpy as np
from hyperopt.early_stop import no_progress_loss

from fedot.core.log import default_log
from fedot.core.optimisers.objective import ObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace

MAX_METRIC_VALUE = np.inf


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library
    
    Args:
      iterations: max number of iterations
      search_space: SearchSpace instance
      algo: algorithm for hyperparameters optimization with signature similar to :obj:`hyperopt.tse.suggest`
    """

    def __init__(self, objective_evaluate: ObjectiveEvaluate,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = None,
                 n_jobs: int = -1):
        self.iterations = iterations
        iteration_stop_count = early_stopping_rounds or max(100, int(np.sqrt(iterations) * 10))
        self.early_stop_fn = no_progress_loss(iteration_stop_count=iteration_stop_count)
        self.max_seconds = int(timeout.seconds) if timeout is not None else None
        self.init_pipeline = None
        self.init_metric = None
        self.obtained_metric = None
        self.objective_evaluate = objective_evaluate
        self._default_metric_value = MAX_METRIC_VALUE
        self.search_space = search_space
        self.algo = algo
        self.n_jobs = n_jobs

        self.log = default_log(self)

    @abstractmethod
    def tune(self, pipeline: Pipeline) -> Pipeline:
        """
        Function for hyperparameters tuning on the pipeline

        Args:
          pipeline: Pipeline for which hyperparameters tuning is needed

        Returns:
          graph with optimized hyperparameters
        """
        raise NotImplementedError()

    def get_metric_value(self, pipeline: Pipeline) -> float:
        """
        Method calculates metric for algorithm validation
        
        Args:
          pipeline: Pipeline to evaluate

        Returns:
          value of loss function
        """
        pipeline.unfit()
        pipeline_fitness = self.objective_evaluate.evaluate(pipeline)
        metric_value = pipeline_fitness.value
        if not pipeline_fitness.valid:
            return self._default_metric_value
        return metric_value

    def init_check(self, pipeline: Pipeline) -> None:
        """
        Method get metric on validation set before start optimization

        Args:
          pipeline: Pipeline to calculate objective
        """
        self.log.info('Hyperparameters optimization start')

        # Train pipeline
        self.init_pipeline = deepcopy(pipeline)

        self.init_metric = self.get_metric_value(pipeline=self.init_pipeline)
        self.log.message(f'Initial pipeline: {self.init_pipeline.structure} \n'
                         f'Initial metric: {abs(self.init_metric):.3f}')

    def final_check(self, tuned_pipeline: Pipeline):
        """
        Method propose final quality check after optimization process

        Args:
          tuned_pipeline: Tuned pipeline to calculate objective
        """

        self.obtained_metric = self.get_metric_value(pipeline=tuned_pipeline)

        if self.obtained_metric == self._default_metric_value:
            self.obtained_metric = None

        self.log.info('Hyperparameters optimization finished')

        prefix_tuned_phrase = 'Return tuned pipeline due to the fact that obtained metric'
        prefix_init_phrase = 'Return init pipeline due to the fact that obtained metric'

        # 5% deviation is acceptable
        deviation = (self.init_metric / 100.0) * 5
        init_metric = self.init_metric + deviation * np.sign(self.init_metric)
        if self.obtained_metric is None:
            self.log.info(f'{prefix_init_phrase} is None. Initial metric is {abs(init_metric):.3f}')
            final_pipeline = self.init_pipeline

        elif self.obtained_metric <= init_metric:
            self.log.info(f'{prefix_tuned_phrase} {abs(self.obtained_metric):.3f} equal or '
                          f'better than initial (+ 5% deviation) {abs(init_metric):.3f}')
            final_pipeline = tuned_pipeline
        else:
            self.log.info(f'{prefix_init_phrase} {abs(self.obtained_metric):.3f} '
                          f'worse than initial (+ 5% deviation) {abs(init_metric):.3f}')
            final_pipeline = self.init_pipeline
        self.log.message(f'Final pipeline: {final_pipeline.structure}')
        if self.obtained_metric is not None:
            self.log.message(f'Final metric: {abs(self.obtained_metric):.3f}')
        else:
            self.log.message(f'Final metric is None')
        return final_pipeline
