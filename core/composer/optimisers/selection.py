import math
from copy import deepcopy
from enum import Enum
from random import randint, choice
from typing import (
    List,
    Any,
    Tuple
)

import numpy as np


class SelectionTypesEnum(Enum):
    tournament = 'tournament'


def selection(types: List[SelectionTypesEnum], fitness: List[float], population: List[Any], pop_size: int) -> List[
    Tuple[Any]]:
    type = choice(types)
    if type in selection_by_type.keys():
        return selection_by_type[type](fitness, population, pop_size)
    else:
        raise ValueError(f'Required selection not found: {type}')


def random_selection(pop_size: int, group_size: int) -> List[int]:
    return [randint(0, pop_size - 1) for _ in range(group_size)]


def tournament_selection(fitness: List[float], population: List[Any], pop_size: int, num_of_parents=2,
                         fraction=0.1) -> List[Tuple[Any]]:
    group_size = math.ceil(len(fitness) * fraction)
    min_group_size = 2 if len(fitness) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []

    for i in range(pop_size):
        choice_item = []
        for _ in range(num_of_parents):
            group = random_selection(len(fitness), group_size)
            num_of_chosen_ind = group[np.argmin([fitness[ind_num] for ind_num in group])]
            choice_item.append(deepcopy(population[num_of_chosen_ind]))
        chosen.append(tuple(parent for parent in choice_item))
    return chosen


selection_by_type = {
    SelectionTypesEnum.tournament: tournament_selection,
}
