from dataclasses import dataclass
from typing import Sequence, Union, Any

from fedot.core.optimisers.optimizer import GraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.elitism import ElitismTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, MutationStrengthEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum


@dataclass
class GPGraphOptimizerParameters(GraphOptimizerParameters):
    """
    Defines parameters of evolutionary operators and the algorithm of genetic optimizer.

    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param static_mutation_prob: probability of applying same mutation to graph in a cycle of mutations
    :param max_num_of_operator_attempts: max number of unsuccessful operator (mutation/crossover)
    attempts before continuing
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)
    :param min_pop_size_with_elitism: minimal population size with which elitism is applicable

    :param selection_types: Sequence of selection operators types
    :param crossover_types: Sequence of crossover operators types
    :param mutation_types: Sequence of mutation operators types
    :param elitism_type: type of elitism operator evolution
    :param regularization_type: type of regularization operator
    :param genetic_scheme_type: type of genetic evolutionary scheme
    """

    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    static_mutation_prob: float = 0.7
    max_num_of_operator_attempts: int = 100
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    min_pop_size_with_elitism: int = 5

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

    def __post_init__(self):
        if self.multi_objective:
            self.selection_types = (SelectionTypesEnum.spea2,)
            # TODO add possibility of using regularization in MO alg
            self.regularization_type = RegularizationTypesEnum.none
