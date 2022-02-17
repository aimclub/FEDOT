import gc
import platform
from dataclasses import dataclass
from functools import partial
from multiprocessing import set_start_method
from typing import Any, Callable, List, Optional, Tuple, Union

from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import common_rules, ts_rules, validate
from fedot.core.repository.quality_metrics_repository import (MetricsEnum,
                                                              MetricsRepository)
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.compose.tabular import table_metric_calculation
from fedot.core.validation.compose.time_series import ts_metric_calculation
from fedot.remote.remote_evaluator import RemoteEvaluator, init_data_for_remote_execution

sample_split_ratio_for_tasks = {
    TaskTypesEnum.classification: 0.8,
    TaskTypesEnum.regression: 0.8,
    TaskTypesEnum.ts_forecasting: 0.5
}


def set_multiprocess_start_method():
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)


@dataclass
class PipelineComposerRequirements(ComposerRequirements):
    """
    Dataclass is for defining the requirements for composition process of genetic programming composer

    :attribute pop_size: population size
    :attribute num_of_generations: maximal number of evolutionary algorithm generations
    :attribute crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :attribute mutation_prob: mutation probability
    :attribute mutation_strength: strength of mutation in tree (using in certain mutation types)
    :attribute start_depth: start value of tree depth
    :attribute validation_blocks: number of validation blocks for time series validation
    """
    pop_size: Optional[int] = 20
    num_of_generations: Optional[int] = 20
    crossover_prob: Optional[float] = 0.8
    mutation_prob: Optional[float] = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    start_depth: int = None
    validation_blocks: int = None


class GPComposer(Composer):
    """
    Genetic programming based composer
    :param optimiser: optimiser generated in ComposerBuilder
    :param metrics: metrics used to define the quality of found solution.
    :param composer_requirements: requirements for composition process
    :param initial_pipelines: defines the initial state of the population. If None then initial population is random.
    """

    def __init__(self, optimiser=None,
                 composer_requirements: Optional[PipelineComposerRequirements] = None,
                 metrics: Union[List[MetricsEnum], MetricsEnum] = None,
                 initial_pipelines: Optional[List[Pipeline]] = None,
                 logger: Log = None):

        super().__init__(metrics=metrics, composer_requirements=composer_requirements,
                         initial_pipelines=initial_pipelines)

        self.cache = OperationsCache(log=logger)

        self.optimiser = optimiser
        self.cache_path = None
        self.use_existing_cache = False

        if not logger:
            self.log = default_log(__name__)
        else:
            self.log = logger

    def compose_pipeline(self, data: Union[InputData, MultiModalData], is_visualise: bool = False,
                         is_tune: bool = False,
                         on_next_iteration_callback: Optional[Callable] = None) -> Union[Pipeline, List[Pipeline]]:
        """ Function for optimal pipeline structure searching
        :param data: InputData for pipeline composing
        :param is_visualise: is it needed to visualise
        :param is_tune: is it needed to tune pipeline after composing TODO integrate new tuner
        :param on_next_iteration_callback: TODO add description
        :return best_pipeline: obtained result after composing: one pipeline for single-objective optimization;
            For the multi-objective case, the list of the graph is returned.
            In the list, the pipelines are ordered by the descending of primary metric (the first is the best)
        """

        self.optimiser.graph_generation_params.advisor.task = data.task

        if data.task.task_type == TaskTypesEnum.ts_forecasting:
            self.optimiser.graph_generation_params.rules_for_constraint = ts_rules + common_rules
        else:
            self.optimiser.graph_generation_params.rules_for_constraint = common_rules

        if self.composer_requirements.max_pipeline_fit_time:
            set_multiprocess_start_method()

        if not self.optimiser:
            raise AttributeError(f'Optimiser for graph composition is not defined')

        # shuffle data if necessary
        data.shuffle()

        if self.composer_requirements.cv_folds is not None:
            objective_function_for_pipeline = self._cv_validation_metric_build(data)
        else:
            self.log.info("Hold out validation for graph composing was applied.")
            split_ratio = sample_split_ratio_for_tasks[data.task.task_type]
            train_data, test_data = train_test_data_setup(data, split_ratio)

            if RemoteEvaluator().use_remote:
                init_data_for_remote_execution(train_data)

            objective_function_for_pipeline = partial(self.composer_metric, self.metrics, train_data, test_data)

        if self.cache_path is None:
            self.cache.clear()
        else:
            self.cache.clear(tmp_only=True)
            self.cache = OperationsCache(log=self.log, db_path=self.cache_path,
                                         clear_exiting=not self.use_existing_cache)

        opt_result = self.optimiser.optimise(objective_function_for_pipeline,
                                             on_next_iteration_callback=on_next_iteration_callback)
        best_pipeline = self._convert_opt_results_to_pipeline(opt_result)

        self.log.info('GP composition finished')
        self.cache.clear()
        if is_tune:
            self.tune_pipeline(best_pipeline, data, self.composer_requirements.timeout)
        return best_pipeline

    def _convert_opt_results_to_pipeline(self, opt_result: Union[OptGraph, List[OptGraph]]) -> Pipeline:
        return [self.optimiser.graph_generation_params.adapter.restore(graph)
                for graph in opt_result] if isinstance(opt_result, list) \
            else self.optimiser.graph_generation_params.adapter.restore(opt_result)

    def _cv_validation_metric_build(self, data):
        """ Prepare function for metric evaluation based on task """
        if isinstance(data, MultiModalData):
            raise NotImplementedError('Cross-validation is not supported for multi-modal data')
        task_type = data.task.task_type
        if task_type is TaskTypesEnum.ts_forecasting:
            # Perform time series cross validation
            self.log.info("Time series cross validation for pipeline composing was applied.")
            if self.composer_requirements.validation_blocks is None:
                self.log.info('For ts cross validation validation_blocks number was changed from None to 3 blocks')
                self.composer_requirements.validation_blocks = 3

            metric_function_for_nodes = partial(ts_metric_calculation, data,
                                                self.composer_requirements.cv_folds,
                                                self.composer_requirements.validation_blocks,
                                                self.metrics,
                                                log=self.log)

        else:
            self.log.info("KFolds cross validation for pipeline composing was applied.")

            metric_function_for_nodes = partial(table_metric_calculation, data,
                                                self.composer_requirements.cv_folds,
                                                self.metrics,
                                                log=self.log,
                                                cache=self.cache)
        return metric_function_for_nodes

    def composer_metric(self, metrics,
                        train_data: Union[InputData, MultiModalData],
                        test_data: Union[InputData, MultiModalData],
                        pipeline: Pipeline) -> Optional[Tuple[Any]]:
        try:
            validate(pipeline, task=train_data.task)
            pipeline.log = self.log

            if type(metrics) is not list:
                metrics = [metrics]

            if self.cache is not None:
                pipeline.fit_from_cache(self.cache)

            self.log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit started')

            pipeline.fit(input_data=train_data,
                         time_constraint=self.composer_requirements.max_pipeline_fit_time,
                         use_fitted=self.cache is not None)
            self.cache.save_pipeline(pipeline)

            evaluated_metrics = ()
            for metric in metrics:
                if callable(metric):
                    metric_func = metric
                else:
                    metric_func = MetricsRepository().metric_by_id(metric)
                evaluated_metrics = evaluated_metrics + (metric_func(pipeline, reference_data=test_data),)

            self.log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {list(evaluated_metrics)}')

            # enforce memory cleaning
            pipeline.unfit()
            gc.collect()
        except Exception as ex:
            self.log.info(f'Pipeline assessment warning: {ex}. Continue.')
            evaluated_metrics = None
        return evaluated_metrics

    @staticmethod
    def tune_pipeline(pipeline: Pipeline, data: InputData, time_limit):
        raise NotImplementedError()

    @property
    def history(self):
        return self.optimiser.history
