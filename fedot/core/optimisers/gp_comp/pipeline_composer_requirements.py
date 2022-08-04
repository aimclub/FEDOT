import logging
from dataclasses import dataclass
from typing import Optional

from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


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
    :attribute show_progress: bool indicating whether to show progress using tqdm or not
    :attribute logging_level_opt: logging level for optimiser
    :attribute collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
    :attribute keep_n_best: Number of the best individuals of previous generation to keep in next generation.
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
    show_progress: bool = True
    logging_level_opt: int = logging.INFO
    collect_intermediate_metric: bool = False
    keep_n_best: int = 1
