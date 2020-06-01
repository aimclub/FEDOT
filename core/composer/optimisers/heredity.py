from copy import deepcopy
from enum import Enum
from typing import (Any, List)

from core.composer.optimisers.selection import SelectionTypesEnum
from core.composer.optimisers.selection import individuals_selection


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'


def heredity(type: GeneticSchemeTypesEnum, selection_types: List[SelectionTypesEnum], prev_population: List[Any],
             new_population: List[Any], max_size: int) -> List[Any]:
    genetic_scheme_by_type = {
        GeneticSchemeTypesEnum.steady_state: steady_state_heredity(selection_types, prev_population, new_population,
                                                                   max_size),
        GeneticSchemeTypesEnum.generational: direct_heredity(new_population, max_size)
    }
    if type in genetic_scheme_by_type.keys():
        return genetic_scheme_by_type[type]
    else:
        raise ValueError(f'Required genetic scheme not found: {type}')


def steady_state_heredity(selection_types: List[SelectionTypesEnum], prev_population: List[Any],
                          new_population: List[Any], max_size: int):
    return individuals_selection(types=selection_types,
                                 individuals=prev_population + new_population,
                                 pop_size=max_size)


def direct_heredity(new_population: List[Any], max_size: int):
    return deepcopy(new_population[:max_size])
