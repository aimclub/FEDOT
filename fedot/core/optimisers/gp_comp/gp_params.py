from dataclasses import dataclass
from typing import Sequence, Union, Any

from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.elitism import ElitismTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, MutationStrengthEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.optimizer import GraphOptimizerParameters


@dataclass
class GPGraphOptimizerParameters(GraphOptimizerParameters):
    """
    Defines parameters of evolutionary operators and the algorithm of genetic optimizer.

    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)

    :param selection_types: Sequence of selection operators types
    :param crossover_types: Sequence of crossover operators types
    :param mutation_types: Sequence of mutation operators types
    :param elitism_type: type of elitism operator evolution
    :param regularization_type: type of regularization operator
    :param genetic_scheme_type: type of genetic evolutionary scheme
    """

    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean

    selection_types: Sequence[SelectionTypesEnum] = \
        (SelectionTypesEnum.tournament,)
    crossover_types: Sequence[Union[CrossoverTypesEnum, Any]] = \
        (CrossoverTypesEnum.subtree,
         CrossoverTypesEnum.one_point)
    mutation_types: Sequence[Union[MutationTypesEnum, Any]] = \
        (MutationTypesEnum.simple,
         MutationTypesEnum.reduce,
         MutationTypesEnum.growth,
         MutationTypesEnum.local_growth)
    elitism_type: ElitismTypesEnum = ElitismTypesEnum.keep_n_best
    regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none
    genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational
