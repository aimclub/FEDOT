from random import randint
from typing import (
    List,
    Any,
    Tuple
)

from enum import Enum
import numpy as np
from copy import deepcopy


class SelectionTypeEnum(Enum):
    tournament = 0


def selection(sel_type, fitness: List[float], population: List[Any]) -> List[Tuple[Any]]:
    if sel_type == SelectionTypeEnum.tournament:
        return tournament_selection(fitness, population)


def random_selection(pop_size: int, group_size: int) -> List[int]:
    return [randint(0, pop_size - 1) for _ in range(group_size)]


def tournament_selection(fitness: List[float], population: List[Any], group_size: int = 9, num_of_parents=2) -> List[
    Tuple[Any]]:
    group_size = min(group_size, len(fitness))
    chosen = []

    for i in range(len(fitness)):
        choice_item = []
        for _ in range(num_of_parents):
            group = random_selection(len(fitness), group_size)
            num_of_chosen_ind = group[np.argmin([fitness[ind_num] for ind_num in group])]
            choice_item.append(deepcopy(population[num_of_chosen_ind]))
        chosen.append(tuple(parent for parent in choice_item))
    return chosen
