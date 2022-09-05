from dataclasses import dataclass
from typing import Sequence, Union, Any

from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.elitism import ElitismTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.optimizer import GraphOptimizerParameters


@dataclass
class GPGraphOptimizerParameters(GraphOptimizerParameters):
    """
    This class is for defining the operators and algorithm details of genetic optimizer.

    :param selection_types: Sequence of selection operators types
    :param crossover_types: Sequence of crossover operators types
    :param mutation_types: Sequence of mutation operators types
    :param regularization_type: type of regularization operator
    :param genetic_scheme_type: type of genetic evolutionary scheme
    :param elitism_type: type of elitism operator evolution
    """

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
    regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none
    genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational
    elitism_type: ElitismTypesEnum = ElitismTypesEnum.keep_n_best
