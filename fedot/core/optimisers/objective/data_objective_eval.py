import traceback
from datetime import timedelta
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
from golem.core.log import default_log, is_test_session
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective.objective import Objective, to_fitness
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.data.data import InputData
from fedot.core.operations.model import Model
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.debug import is_recording_mode, save_debug_info_for_pipeline

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
    :param operations_cache: Cache manager for fitted models, optional.
    :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
    :param eval_n_jobs: number of jobs used to evaluate the objective.
    :params do_unfit: unfit graph after evaluation
    """

    def __init__(self,
                 objective: Objective,
                 data_producer: DataSource,
                 time_constraint: Optional[timedelta] = None,
                 validation_blocks: Optional[int] = None,
                 operations_cache: Optional[OperationsCache] = None,
                 preprocessing_cache: Optional[PreprocessingCache] = None,
                 predictions_cache: Optional[PredictionsCache] = None,
                 eval_n_jobs: int = 1,
                 do_unfit: bool = True):
        super().__init__(objective, eval_n_jobs=eval_n_jobs)
        self._data_producer = data_producer
        self._time_constraint = time_constraint
        self._validation_blocks = validation_blocks
        self._operations_cache = operations_cache
        self._preprocessing_cache = preprocessing_cache
        self._predictions_cache = predictions_cache
        self._log = default_log(self)
        self._do_unfit = do_unfit

    def evaluate(self, graph: Pipeline) -> Fitness:
        # Seems like a workaround for situation when logger is lost
        #  when adapting and restoring it to/from OptGraph.
        graph.log = self._log

        graph_id = graph.root_node.descriptive_id
        self._log.debug(f'Pipeline {graph_id} fit started')

        folds_metrics = []
        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            try:
                prepared_pipeline = self.prepare_graph(graph, train_data, fold_id, self._eval_n_jobs)
            except Exception as ex:
                self._log.warning(f'Unsuccessful pipeline fit during fitness evaluation. '
                                  f'Skipping the pipeline. Exception <{ex}> on {graph_id}')
                if is_test_session() and not isinstance(ex, TimeoutError):
                    stack_trace = traceback.format_exc()
                    save_debug_info_for_pipeline(graph, train_data, test_data, ex, stack_trace)
                    if not is_recording_mode() and 'catboost' not in graph.descriptive_id:
                        raise ex
                break  # if even one fold fails, the evaluation stops

            evaluated_fitness = self._objective(prepared_pipeline,
                                                reference_data=test_data,
                                                validation_blocks=self._validation_blocks,
                                                predictions_cache=self._predictions_cache,
                                                fold_id=fold_id)

            if evaluated_fitness.valid:
                folds_metrics.append(evaluated_fitness.values)
            else:
                self._log.log_or_raise('warning', ValueError(f'Invalid fitness after objective evaluation. '
                                                             f'Skipping the graph: {graph_id}'))
            if self._do_unfit:
                graph.unfit()
        if folds_metrics:
            folds_metrics = tuple(np.mean(folds_metrics, axis=0))  # averages for each metric over folds
            self._log.debug(f'Pipeline {graph_id} with evaluated metrics: {folds_metrics}')
        else:
            folds_metrics = None

        # prepared_pipeline.
        if self._predictions_cache is not None:
            self._log.debug(f"Predictions cache effectiveness ratio: {self._predictions_cache.effectiveness_ratio}")

        return to_fitness(folds_metrics, self._objective.is_multi_objective)

    def prepare_graph(self, graph: Pipeline, train_data: InputData,
                      fold_id: Optional[int] = None, n_jobs: int = -1) -> Pipeline:
        """
        Fit pipeline before metric evaluation can be performed.
        :param graph: pipeline for train & validation
        :param train_data: InputData for training pipeline
        :param fold_id: id of the fold in cross-validation, used for cache requests.
        :param n_jobs: number of parallel jobs for preparation
        """
        if graph.is_fitted:
            # the expected behaviour for the remote evaluation
            return graph

        graph.unfit()

        # load preprocessing
        graph.try_load_from_cache(self._operations_cache, self._preprocessing_cache, fold_id)
        graph.fit(
            train_data,
            n_jobs=n_jobs,
            time_constraint=self._time_constraint,
            predictions_cache=self._predictions_cache,
            fold_id=fold_id
        )

        if self._operations_cache is not None:
            self._operations_cache.save_pipeline(graph, fold_id)
        if self._preprocessing_cache is not None:
            self._preprocessing_cache.add_preprocessor(graph, fold_id)

        return graph

    def evaluate_intermediate_metrics(self, graph: Pipeline):
        """Evaluate intermediate metrics"""
        # Get the last fold
        last_fold = None
        fold_id = None
        for fold_id, last_fold in enumerate(self._data_producer()):
            pass
        # And so test only on the last fold
        train_data, test_data = last_fold
        graph.try_load_from_cache(self._operations_cache, self._preprocessing_cache, fold_id)
        for node in graph.nodes:
            if not isinstance(node.operation, Model):
                continue
            intermediate_graph = Pipeline(node, use_input_preprocessing=graph.use_input_preprocessing)
            intermediate_graph.fit(
                train_data,
                time_constraint=self._time_constraint,
                n_jobs=self._eval_n_jobs,
            )
            intermediate_fitness = self._objective(intermediate_graph,
                                                   reference_data=test_data,
                                                   validation_blocks=self._validation_blocks)
            # saving only the most important first metric
            node.metadata.metric = intermediate_fitness.values[0]

    @property
    def input_data(self):
        return self._data_producer.args[0]
