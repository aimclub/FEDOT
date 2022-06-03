from datetime import timedelta
from typing import Optional, Callable, Iterable, Tuple

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.operations.model import Model
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.optimisers.fitness import Fitness
from .objective import Objective, to_fitness
from .objective_eval import ObjectiveEvaluate


DataSource = Callable[[], Iterable[Tuple[InputData, InputData]]]


class PipelineObjectiveEvaluate(ObjectiveEvaluate[Pipeline]):
    """
    Evaluator of Objective that requires train and test data for metric evaluation.
    Its role is to prepare graph on train-data and then evaluate metrics on test data.

    :param objective: Objective for evaluating metrics on pipelines.
    :param data_producer: Producer of data folds, each fold is a tuple of (train_data, test_data).
    If it returns a single fold, it's effectively a hold-out validation. For many folds it's k-folds.
    :param time_constraint: Optional time constraint for pipeline.fit.
    :param validation_blocks: Number of validation blocks, optional, used only for time series validation.
    :param cache: Cache manager for fitted models, optional.
    :param log: Logger.
    """

    def __init__(self,
                 objective: Objective,
                 data_producer: DataSource,
                 time_constraint: Optional[timedelta] = None,
                 validation_blocks: Optional[int] = None,
                 cache: Optional[OperationsCache] = None,
                 log: Log = None):
        super().__init__(objective)
        self._data_producer = data_producer
        self._time_constraint = time_constraint
        self._validation_blocks = validation_blocks
        self._cache = cache
        self._log = log or default_log(__name__)

    def evaluate(self, graph: Pipeline) -> Fitness:
        # Seems like a workaround for situation when logger is lost
        #  when adapting and restoring it to/from OptGraph.
        graph.log = self._log

        graph_id = graph.root_node.descriptive_id
        self._log.debug(f'Pipeline {graph_id} fit started')

        folds_metrics = []
        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            try:
                prepared_pipeline = self.prepare_graph(graph, train_data, fold_id)
            except Exception as ex:
                self._log.warn(f'Continuing after pipeline fit error <{ex}> for graph: {graph_id}')
                continue

            evaluated_fitness = self._objective(prepared_pipeline,
                                                reference_data=test_data,
                                                validation_blocks=self._validation_blocks)
            if evaluated_fitness.valid:
                folds_metrics.append(evaluated_fitness.values)
            else:
                self._log.warn(f'Continuing after objective evaluation error for graph: {graph_id}')
                continue

        if folds_metrics:
            folds_metrics = tuple(np.mean(folds_metrics, axis=0))  # averages for each metric over folds
            self._log.debug(f'Pipeline {graph_id} with evaluated metrics: {folds_metrics}')
        else:
            folds_metrics = None
        return to_fitness(folds_metrics, self._objective.is_multi_objective)

    def prepare_graph(self, graph: Pipeline, train_data: InputData, fold_id: Optional[int] = None) -> Pipeline:
        """
        Fit pipeline before metric evaluation can be performed.
        :param graph: pipeline for train & validation
        :param train_data: InputData for training pipeline
        :param fold_id: Id of the fold in cross-validation, used for cache requests.
        """
        graph.fit(
            train_data,
            use_fitted=graph.fit_from_cache(self._cache, fold_id),
            time_constraint=self._time_constraint
        )
        if self._cache is not None:
            self._cache.save_pipeline(graph, fold_id)
        return graph

    def evaluate_intermediate_metrics(self, graph: Pipeline):
        """Evaluate intermediate metrics without any pipeline fit."""
        # Get the last fold on which pipeline was trained to avoid retraining
        last_fold = None
        for last_fold in self._data_producer():
            pass
        # And so test only on the last fold
        test_data = last_fold[1]

        for node in graph.nodes:
            if not isinstance(node.operation, Model):
                continue
            intermediate_graph = Pipeline(node)
            intermediate_graph.preprocessor = graph.preprocessor
            intermediate_fitness = self._objective(intermediate_graph,
                                                   reference_data=test_data,
                                                   validation_blocks=self._validation_blocks)
            # saving only the most important first metric
            node.metadata.metric = intermediate_fitness.values[0]
