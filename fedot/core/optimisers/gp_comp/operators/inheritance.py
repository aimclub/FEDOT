from functools import partial
from typing import (Any, List)

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, Selection
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'
    parameter_free = 'parameter_free'


class Inheritance:
    def __init__(self, genetic_scheme_type: GeneticSchemeTypesEnum,
                 selection_types: List[SelectionTypesEnum]):
        self.genetic_scheme_type = genetic_scheme_type
        self.selection_types = selection_types

    def __call__(self, previous_population: PopulationT, new_population: PopulationT, max_size: int) -> PopulationT:
        return self.inheritance_type_by_genetic_scheme(self.genetic_scheme_type, previous_population,
                                                       new_population, max_size)()

    def steady_state_inheritance(self,
                                 prev_population: List[Any],
                                 new_population: List[Any], max_size: int, ):
        selection = Selection(self.selection_types)
        return selection.individuals_selection(individuals=prev_population + new_population, pop_size=max_size)

    def direct_inheritance(self, new_population: List[Any], max_size: int):
        return new_population[:max_size]

    def inheritance_type_by_genetic_scheme(self, genetic_scheme_type: GeneticSchemeTypesEnum,
                                           previous_population: PopulationT,
                                           new_population: PopulationT, max_size: int):
        steady_state_scheme = partial(self.steady_state_inheritance, previous_population,
                                      new_population, max_size)
        generational_scheme = partial(self.direct_inheritance, new_population, max_size)
        inheritance_type_by_genetic_scheme = {
            GeneticSchemeTypesEnum.generational: generational_scheme,
            GeneticSchemeTypesEnum.steady_state: steady_state_scheme,
            GeneticSchemeTypesEnum.parameter_free: steady_state_scheme
        }
        return inheritance_type_by_genetic_scheme[genetic_scheme_type]
