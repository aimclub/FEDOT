import math
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


def selection(types: List[SelectionTypesEnum], fitness: List[float], population: List[Any], pop_size: int) -> Tuple[
    List[Any], List[float]]:
    type = choice(types)
    if type in selection_by_type.keys():
        return selection_by_type[type](fitness, population, pop_size)
    else:
        raise ValueError(f'Required selection not found: {type}')


def random_selection(pop_size: int, group_size: int) -> List[int]:
    return [randint(0, pop_size - 1) for _ in range(group_size)]


def tournament_selection(fitness: List[float], population: List[Any], pop_size: int, fraction=0.1) -> Tuple[
    List[Any], List[float]]:
    group_size = math.ceil(len(fitness) * fraction)
    min_group_size = 2 if len(fitness) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    chosen_fitness = []
    for i in range(pop_size):
        group = random_selection(len(fitness), group_size)
        num_of_chosen_ind = group[np.argmin([fitness[ind_num] for ind_num in group])]
        chosen.append(population[num_of_chosen_ind])
        chosen_fitness.append(fitness[num_of_chosen_ind])
    return chosen, chosen_fitness


def individuals_selection(types: List[SelectionTypesEnum], fitness: List[float], population: List[Any],
                          pop_size: int) -> Tuple[List[Any], List[float]]:
    chosen = []
    chosen_fitness = []
    for _ in range(pop_size):
        individual, ind_fitness = selection(types, fitness, population, pop_size=1)
        chosen_fitness.append(ind_fitness[0])
        chosen.append(individual[0])
        chosen_index_in_pop = population.index(individual[0])
        population.remove(population[chosen_index_in_pop])
        fitness.pop(chosen_index_in_pop)
    return chosen, chosen_fitness


selection_by_type = {
    SelectionTypesEnum.tournament: tournament_selection,
}
