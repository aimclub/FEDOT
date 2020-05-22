import math
from enum import Enum
from random import randint, choice
from typing import (
    List,
    Any,
    Tuple
)


class SelectionTypesEnum(Enum):
    tournament = 'tournament'


def selection(types: List[SelectionTypesEnum], population: List[Any], pop_size: int) -> List[Any]:
    type = choice(types)
    if type in selection_by_type.keys():
        return selection_by_type[type](population, pop_size)
    else:
        raise ValueError(f'Required selection not found: {type}')


def random_selection(pop_size: int, group_size: int) -> List[int]:
    return [randint(0, pop_size - 1) for _ in range(group_size)]


def tournament_selection(individuals: List[Any], pop_size: int, fraction=0.1) -> List[Any]:
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = 2 if len(individuals) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    for _ in range(pop_size):
        group_ids = random_selection(len(individuals), group_size)
        group = [individuals[group_id] for group_id in group_ids]
        best = min(group, key=lambda ind: ind.fitness)
        chosen.append(best)
    return chosen


def individuals_selection(types: List[SelectionTypesEnum], individuals: List[Any],
                          pop_size: int) -> List[Any]:
    chosen = []
    for _ in range(pop_size):
        individual = selection(types, individuals, pop_size=1)
        chosen.append(individual[0])
        chosen_index_in_pop = individuals.index(individual[0])
        individuals.remove(individuals[chosen_index_in_pop])
        # fitness.pop(chosen_index_in_pop)
    return chosen


selection_by_type = {
    SelectionTypesEnum.tournament: tournament_selection,
}
