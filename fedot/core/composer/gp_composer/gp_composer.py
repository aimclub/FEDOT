import platform
from dataclasses import dataclass
from functools import partial
from multiprocessing import set_start_method
from typing import List, Optional, Sequence, Union

from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective.data_objective_builder import DataObjectiveBuilder
from fedot.core.optimisers.opt_history import OptHistory, log_to_history
from fedot.core.optimisers.optimizer import GraphOptimiser
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import common_rules, ts_rules
from fedot.core.repository.tasks import TaskTypesEnum


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
    :attribute n_jobs: num of n_jobs
    :attribute collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
    """
    pop_size: Optional[int] = 20
    num_of_generations: Optional[int] = 20
    offspring_rate: Optional[float] = 0.5
    crossover_prob: Optional[float] = 0.8
    mutation_prob: Optional[float] = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    start_depth: int = None
    validation_blocks: int = None
    n_jobs: int = 1
    collect_intermediate_metric: bool = False


class GPComposer(Composer):
    """
    Genetic programming based composer
    :param optimiser: optimiser generated in ComposerBuilder.
    :param composer_requirements: requirements for composition process.
    :param initial_pipelines: defines the initial state of the population. If None then initial population is random.
    :param logger: optional Log object for logging.
    :param cache: optional cache for Operations.
    """

    def __init__(self, optimiser: GraphOptimiser,
                 composer_requirements: PipelineComposerRequirements,
                 initial_pipelines: Optional[Sequence[Pipeline]] = None,
                 logger: Optional[Log] = None,
                 cache: Optional[OperationsCache] = None):

        super().__init__(optimiser, composer_requirements, initial_pipelines, logger)

        self.optimiser = optimiser
        self.cache: Optional[OperationsCache] = cache

        self._history = OptHistory(self.optimiser.objective, self.optimiser.parameters.history_folder)
        self.objective_builder = DataObjectiveBuilder(self.optimiser.objective,
                                                      self.composer_requirements.max_pipeline_fit_time,
                                                      self.composer_requirements.cv_folds,
                                                      self.composer_requirements.validation_blocks,
                                                      self.cache, self.log)

    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, List[Pipeline]]:
        self.optimiser.graph_generation_params.advisor.task = data.task

        # TODO: move this late-init logic to the point before optimiser is constructed
        if data.task.task_type is TaskTypesEnum.ts_forecasting:
            self.optimiser.graph_generation_params.rules_for_constraint = ts_rules + common_rules
        else:
            self.optimiser.graph_generation_params.rules_for_constraint = common_rules

        if self.composer_requirements.max_pipeline_fit_time:
            set_multiprocess_start_method()

        # shuffle data if necessary
        data.shuffle()

        self._history.clean_results()
        history_callback = partial(log_to_history, self._history)
        self.optimiser.optimisation_callback = history_callback

        objective_evaluator = self.objective_builder.build(data)
        opt_result = self.optimiser.optimise(objective_evaluator)

        best_pipeline = self._convert_opt_results_to_pipeline(opt_result)
        self.log.info('GP composition finished')
        return best_pipeline

    def _convert_opt_results_to_pipeline(self, opt_result: Union[OptGraph, List[OptGraph]]) -> Pipeline:
        return [self.optimiser.graph_generation_params.adapter.restore(graph)
                for graph in opt_result] if isinstance(opt_result, list) \
            else self.optimiser.graph_generation_params.adapter.restore(opt_result)

    @staticmethod
    def tune_pipeline(pipeline: Pipeline, data: InputData, time_limit):
        raise NotImplementedError()

    @property
    def history(self):
        return self._history
