import dataclasses
from dataclasses import dataclass
import logging

from typing import Optional

from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


@dataclass
class PipelineComposerRequirements(ComposerRequirements):
    """Parameters of evolutionary optimizer that define features of the evolutionary algorithm
    and restrictions on the graph composition process.

    Evolutionary optimization options
    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)

    Population and graph parameters, possibly adaptive:
    :param offspring_rate: offspring rate used on next population
    :param pop_size: initial population size; if unspecified, default value is used
    :param max_pop_size: maximum population size; optional, if unspecified, then population size is unbound
    :param keep_n_best: number of the best individuals of previous generation to keep in next generation
    :param adaptive_depth: flag to enable adaptive configuration of graph depth
    :param adaptive_depth_max_stagnation: max number of stagnating populations before adaptive depth increment

    Restrictions on final graphs:
    :param start_depth: start value of adaptive tree depth
    :param max_depth: max depth of the resulting pipeline
    :param max_arity: max number of parents for node
    :param min_arity: min number of parents for node
    """
    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean

    keep_n_best: int = 1
    offspring_rate: float = 0.5
    pop_size: int = 20
    max_pop_size: Optional[int] = 55
    adaptive_depth: bool = False
    adaptive_depth_max_stagnation: int = 3

    start_depth: int = 3
    # TODO it's actually something like 'current_max_depth', not overall max depth. re
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2

    def __post_init__(self):
        super().__post_init__()
        for field_name, field_value in dataclasses.asdict(self).items():
            if isinstance(field_value, Number) and field_value < 0:
                raise ValueError(f'Value of {field_name} must be non-negative')
