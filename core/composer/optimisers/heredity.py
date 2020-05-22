from typing import (Any, List)

from core.composer.optimisers.selection import SelectionTypesEnum
from core.composer.optimisers.selection import individuals_selection


def steady_state_heredity(prev_population: List[Any], new_population: List[Any], max_size: int):
    return individuals_selection(types=[SelectionTypesEnum.tournament],
                                 individuals=prev_population + new_population,
                                 pop_size=max_size)


def direct_heredity(prev_population: List[Any], new_population: List[Any], max_size: int):
    return new_population[:max_size]
