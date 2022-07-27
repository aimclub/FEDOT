from functools import partial
from typing import (Any, List, Callable)

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, Selection
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'
    parameter_free = 'parameter_free'


class Inheritance:
    def __init__(self, genetic_scheme_type: GeneticSchemeTypesEnum,
                 selection_types: List[SelectionTypesEnum], requirements: PipelineComposerRequirements):
        self.genetic_scheme_type = genetic_scheme_type
        self.selection_types = selection_types
        self.requirements = requirements

    def __call__(self, previous_population: PopulationT, new_population: PopulationT) -> PopulationT:
        return self._inheritance_type_by_genetic_scheme(self.genetic_scheme_type, previous_population, new_population)()

    def _inheritance_type_by_genetic_scheme(self, genetic_scheme_type: GeneticSchemeTypesEnum,
                                            previous_population: PopulationT, new_population: PopulationT) -> Callable:
        steady_state_scheme = partial(self._steady_state_inheritance, previous_population,
                                      new_population)
        generational_scheme = partial(self._direct_inheritance, new_population)
        inheritance_type_by_genetic_scheme = {
            GeneticSchemeTypesEnum.generational: generational_scheme,
            GeneticSchemeTypesEnum.steady_state: steady_state_scheme,
            GeneticSchemeTypesEnum.parameter_free: steady_state_scheme
        }
        return inheritance_type_by_genetic_scheme[genetic_scheme_type]

    def update_requirements(self, new_requirements: PipelineComposerRequirements):
        self.requirements = new_requirements

    def _steady_state_inheritance(self, prev_population: PopulationT, new_population: PopulationT) -> PopulationT:
        selection = Selection(self.selection_types, self.requirements)
        selected_individuals = selection.individuals_selection(individuals=prev_population + new_population)
        return selected_individuals

    def _direct_inheritance(self, new_population: PopulationT) -> PopulationT:
        return new_population[:self.requirements.pop_size]
