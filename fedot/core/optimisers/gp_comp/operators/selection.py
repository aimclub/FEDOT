import math
from random import choice, randint
from typing import Any, List, TYPE_CHECKING

from deap import tools

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.utils import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams


class SelectionTypesEnum(Enum):
    tournament = 'tournament'
    nsga2 = 'nsga2'
    spea2 = 'spea2'


def selection(types: List[SelectionTypesEnum], population: List[Individual], pop_size: int,
              params: 'GraphGenerationParams') -> List[Any]:
    """
    Selection of individuals based on specified type of selection
    :param types: The set of selection types
    :param population: A list of individuals to select from.
    :param pop_size: The number of individuals to select.
    :param params: params for graph generation and convertation
    """
    selection_by_type = {
        SelectionTypesEnum.tournament: tournament_selection,
        SelectionTypesEnum.nsga2: nsga2_selection,
        SelectionTypesEnum.spea2: spea2_selection
    }

    selection_type = choice(types)
    if selection_type in selection_by_type.keys():
        selected = selection_by_type[selection_type](population, pop_size)
        return selected
    else:
        raise ValueError(f'Required selection not found: {selection_type}')


def individuals_selection(types: List[SelectionTypesEnum], individuals: List[Any], pop_size: int,
                          graph_params: 'GraphGenerationParams') -> List[Any]:
    if pop_size == len(individuals):
        chosen = individuals
    else:
        chosen = []
        remaining_individuals = individuals
        individuals_pool_size = len(individuals)
        for _ in range(pop_size):
            individual = selection(types, remaining_individuals, pop_size=1, params=graph_params)[0]
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
