import math
from random import choice, randint
from typing import (Any, List)

from fedot.core.utils import ComparableEnum as Enum


class SelectionTypesEnum(Enum):
    tournament = 'tournament'


def selection(types: List[SelectionTypesEnum], population: List[Any], pop_size: int) -> List[Any]:
    selection_by_type = {
        SelectionTypesEnum.tournament: tournament_selection,
    }

    type = choice(types)
    if type in selection_by_type.keys():
        return selection_by_type[type](population, pop_size)
    else:
        raise ValueError(f'Required selection not found: {type}')


def individuals_selection(types: List[SelectionTypesEnum], individuals: List[Any], pop_size: int) -> List[Any]:
    chosen = []
    unselected_indexes = [i for i in range(len(individuals))]
    unselected_individuals = individuals
    for _ in range(pop_size):
        individual = selection(types, unselected_individuals, pop_size=1)[0]
        chosen.append(individual)
        chosen_index_in_pop = unselected_indexes[unselected_individuals.index(individual)]
        unselected_indexes.remove(chosen_index_in_pop)
        unselected_individuals = [individuals[i] for i in unselected_indexes]
    return chosen


def random_selection(individuals: List[Any], pop_size: int) -> List[int]:
    return [individuals[randint(0, len(individuals) - 1)] for _ in range(pop_size)]


def tournament_selection(individuals: List[Any], pop_size: int, fraction: float = 0.1) -> List[Any]:
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = 2 if len(individuals) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    for _ in range(pop_size):
        group = random_selection(individuals, group_size)
        best = min(group, key=lambda ind: ind.fitness)
        chosen.append(best)
    return chosen
