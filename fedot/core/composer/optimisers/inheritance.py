from copy import deepcopy
from typing import (Any, List)

from fedot.core.composer.optimisers.selection import SelectionTypesEnum, individuals_selection
from fedot.core.utils import ComparableEnum as Enum


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'


def inheritance(type: GeneticSchemeTypesEnum, selection_types: List[SelectionTypesEnum],
                prev_population: List[Any], new_population: List[Any], max_size: int) -> List[Any]:
    genetic_scheme_by_type = {
        GeneticSchemeTypesEnum.steady_state: steady_state_inheritance(selection_types,
                                                                      prev_population,
                                                                      new_population,
                                                                      max_size),
        GeneticSchemeTypesEnum.generational: direct_heredity(new_population, max_size)
    }
    return genetic_scheme_by_type[type]


def steady_state_inheritance(selection_types: List[SelectionTypesEnum],
                             prev_population: List[Any],
                             new_population: List[Any], max_size: int):
    return individuals_selection(types=selection_types,
                                 individuals=prev_population + new_population,
                                 pop_size=max_size)


def direct_heredity(new_population: List[Any], max_size: int):
    return deepcopy(new_population[:max_size])
