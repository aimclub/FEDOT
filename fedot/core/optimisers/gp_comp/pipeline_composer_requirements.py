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
    """Defines options of evolutionary optimization algorithm.

    Evolutionary optimization options
    :param offspring_rate: offspring rate used on next population
    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)

    Graph generation options
    :param adaptive_depth: flag to enable adaptive configuration of tree depth
    :param adaptive_depth_max_stagnation: max number of stagnating populations before adaptive depth increment
    :param start_depth: start value of adaptive tree depth
    :param max_depth: max depth of the resulting pipeline

    :param max_arity: max number of parents for node
    :param min_arity: min number of parents for node
    """
    offspring_rate: float = 0.5
    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean

    adaptive_depth: bool = False
    adaptive_depth_max_stagnation: int = 3
    start_depth: int = 3
    max_depth: int = 3

    max_arity: int = 2
    min_arity: int = 2

    def __post_init__(self):
        super().__post_init__()
        for field_name, field_value in dataclasses.asdict(self).items():
            if isinstance(field_value, Number) and field_value < 0:
                raise ValueError(f'Value of {field_name} must be non-negative')
