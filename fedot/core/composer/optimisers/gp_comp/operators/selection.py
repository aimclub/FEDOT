import math
from random import choice, randint
from typing import (Any, List)

from deap import tools

from fedot.core.utils import ComparableEnum as Enum


class SelectionTypesEnum(Enum):
    tournament = 'tournament'
    nsga2 = 'nsga2'
    spea2 = 'spea2'


def selection(types: List[SelectionTypesEnum], population: List[Any], pop_size: int) -> List[Any]:
    """
    Selection of individuals based on specified type of selection
    :param types: The set of selection types
    :param population: A list of individuals to select from.
    :param pop_size: The number of individuals to select.
    """
    selection_by_type = {
        SelectionTypesEnum.tournament: tournament_selection,
        SelectionTypesEnum.nsga2: nsga2_selection,
        SelectionTypesEnum.spea2: spea2_selection
    }

    type = choice(types)
    if type in selection_by_type.keys():
        return selection_by_type[type](population, pop_size)
    else:
        raise ValueError(f'Required selection not found: {type}')


def individuals_selection(types: List[SelectionTypesEnum], individuals: List[Any], pop_size: int) -> List[Any]:
    if pop_size == len(individuals):
        chosen = individuals
    else:
        chosen = []
        remaining_individuals = individuals
        individuals_pool_size = len(individuals)
        for _ in range(pop_size):
            individual = selection(types, remaining_individuals, pop_size=1)[0]
            chosen.append(individual)
            if pop_size <= individuals_pool_size:
                remaining_individuals.remove(individual)
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


def nsga2_selection(individuals: List[Any], pop_size: int) -> List[Any]:
    chosen = tools.selNSGA2(individuals, pop_size)
    return chosen


def spea2_selection(individuals: List[Any], pop_size: int) -> List[Any]:
    chosen = tools.selSPEA2(individuals, pop_size)
    return chosen
