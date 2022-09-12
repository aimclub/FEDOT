from functools import partial
from typing import (Callable, TYPE_CHECKING)

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.optimisers.gp_comp.operators.selection import Selection
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'
    parameter_free = 'parameter_free'


class Inheritance(Operator):
    def __init__(self, parameters: 'GPGraphOptimizerParameters',
                 selection: Selection):
        super().__init__(parameters=parameters)
        self.selection = selection

    def __call__(self, previous_population: PopulationT, new_population: PopulationT) -> PopulationT:
        return self._inheritance_type_by_genetic_scheme(previous_population, new_population)()

    def _inheritance_type_by_genetic_scheme(self, previous_population: PopulationT,
                                            new_population: PopulationT) -> Callable:
        steady_state_scheme = partial(steady_state_inheritance, previous_population,
                                      new_population, self.selection)
        generational_scheme = partial(direct_inheritance, new_population, self.parameters.pop_size)
        inheritance_type_by_genetic_scheme = {
            GeneticSchemeTypesEnum.generational: generational_scheme,
            GeneticSchemeTypesEnum.steady_state: steady_state_scheme,
            GeneticSchemeTypesEnum.parameter_free: steady_state_scheme
        }
        return inheritance_type_by_genetic_scheme[self.parameters.genetic_scheme_type]


def steady_state_inheritance(prev_population: PopulationT, new_population: PopulationT, selection: Selection) \
        -> PopulationT:
    selected_individuals = selection.individuals_selection(individuals=prev_population + new_population)
    return selected_individuals


def direct_inheritance(new_population: PopulationT, pop_size: int) -> PopulationT:
    return new_population[:pop_size]
