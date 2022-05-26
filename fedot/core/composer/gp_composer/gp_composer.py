from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Collection, Tuple

from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective.data_objective_builder import DataObjectiveBuilder
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.optimisers.optimizer import GraphOptimiser
from fedot.core.pipelines.pipeline import Pipeline


@dataclass
class PipelineComposerRequirements(ComposerRequirements):
    """
    Dataclass is for defining the requirements for composition process of genetic programming composer

    :attribute pop_size: initial population size; if unspecified, default value is used.
    :attribute max_pop_size: maximum population size; optional, if unspecified, then population size is unbound.
    :attribute num_of_generations: maximal number of evolutionary algorithm generations
    :attribute crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :attribute mutation_prob: mutation probability
    :attribute mutation_strength: strength of mutation in tree (using in certain mutation types)
    :attribute max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :attribute start_depth: start value of tree depth
    :attribute validation_blocks: number of validation blocks for time series validation
    :attribute n_jobs: num of n_jobs
    :attribute collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
    """
    pop_size: int = 20
    max_pop_size: Optional[int] = 55
    num_of_generations: int = 20
    offspring_rate: float = 0.5
    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    max_pipeline_fit_time: int = None
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
                 history: Optional[OptHistory] = None,
                 logger: Optional[Log] = None,
                 cache: Optional[OperationsCache] = None):

        super().__init__(optimiser, composer_requirements, initial_pipelines, logger)
        self.composer_requirements = composer_requirements

        self.optimiser: GraphOptimiser = optimiser
        self.cache: Optional[OperationsCache] = cache
        self.history: Optional[OptHistory] = history
        self.best_models: Collection[Pipeline] = ()

        self.objective_builder = DataObjectiveBuilder(optimiser.objective,
                                                      composer_requirements.max_pipeline_fit_time,
                                                      composer_requirements.cv_folds,
                                                      composer_requirements.validation_blocks,
                                                      cache, logger)

    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, Sequence[Pipeline]]:
        # shuffle data if necessary
        data.shuffle()

        # Keep history of optimization
        if self.history:
            self.history.clean_results()

        # Define objective function
        objective_evaluator = self.objective_builder.build(data)
        objective_function = objective_evaluator.evaluate

        # Define callback for computing intermediate metrics if needed
        if self.composer_requirements.collect_intermediate_metric:
            self.optimiser.set_evaluation_callback(objective_evaluator.evaluate_intermediate_metrics)

        # Finally, run optimization process
        opt_result = self.optimiser.optimise(objective_function)

        best_model, self.best_models = self._convert_opt_results_to_pipeline(opt_result)
        self.log.info('GP composition finished')
        return best_model

    def _convert_opt_results_to_pipeline(self, opt_result: Sequence[OptGraph]) -> Tuple[Pipeline, Sequence[Pipeline]]:
        adapter = self.optimiser.graph_generation_params.adapter
        multi_objective = self.optimiser.objective.is_multi_objective
        best_pipelines = [adapter.restore(graph) for graph in opt_result]
        chosen_best_pipeline = best_pipelines if multi_objective else best_pipelines[0]
        return chosen_best_pipeline, best_pipelines

    @staticmethod
    def tune_pipeline(pipeline: Pipeline, data: InputData, time_limit):
        raise NotImplementedError()
