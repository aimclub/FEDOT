import dataclasses
from dataclasses import dataclass
from numbers import Number
from typing import Optional, Sequence

from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


@dataclass
class PipelineComposerRequirements(ComposerRequirements):
    """Defines restrictions and requirements for composition of final graphs.

    Evolutionary optimization options
    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)

    Population and graph parameters, possibly adaptive:
    :param keep_n_best: number of the best individuals of previous generation to keep in next generation

    Restrictions on final graphs:
    :param start_depth: start value of adaptive tree depth
    :param max_depth: max depth of the resulting pipeline
    :param min_arity: min number of parents for node
    :param max_arity: max number of parents for node

    :param primary: operation types for :class:`~fedot.core.pipelines.node.PrimaryNode`s
    :param secondary: operation types for :class:`~fedot.core.pipelines.node.SecondaryNode`s

    """
    # TODO: move to graph optimizer params
    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean

    keep_n_best: int = 1

    start_depth: int = 3
    # TODO it's actually something like 'current_max_depth', not overall max depth.
    max_depth: int = 3
    min_arity: int = 2
    max_arity: int = 2

    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()

    def __post_init__(self):
        super().__post_init__()
        for field_name, field_value in dataclasses.asdict(self).items():
            if isinstance(field_value, Number) and field_value < 0:
                raise ValueError(f'Value of {field_name} must be non-negative')
