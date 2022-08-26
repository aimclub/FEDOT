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

    :param pop_size: initial population size; if unspecified, default value is used.
    :param max_pop_size: maximum population size; optional, if unspecified, then population size is unbound.
    :param keep_n_best: Number of the best individuals of previous generation to keep in next generation.
    :param num_of_generations: maximal number of evolutionary algorithm generations
    :param offspring_rate: offspring rate used on next population

    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)

    :param adaptive_depth: flag to enable adaptive configuration of tree depth
    :param depth_increase_step: the step of depth increase in automated depth configuration
    if false). Value is defined in ComposerBuilder. Default False.

    :param validation_blocks: number of validation blocks for time series validation
    :param logging_level_opt: level of logging in optimizer
    :param collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
    """
    pop_size: int = 20
    max_pop_size: Optional[int] = 55
    keep_n_best: int = 1
    num_of_generations: int = 20
    offspring_rate: float = 0.5

    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean

    adaptive_depth: bool = False
    depth_increase_step: int = 3

    validation_blocks: int = None
    logging_level_opt: int = logging.INFO
    collect_intermediate_metric: bool = False
