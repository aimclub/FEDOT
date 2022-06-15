from copy import deepcopy
import math
from random import choice, randint
from typing import Any, List, TYPE_CHECKING
from wsgiref.simple_server import demo_app
from itertools import permutations

from deap import tools

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.utils import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams


class SelectionTypesEnum(Enum):
    tournament = 'tournament'
    tournament_selection_for_MGA = 'tournament_selection_for_MGA'
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
        SelectionTypesEnum.tournament_selection_for_MGA: tournament_selection_for_MGA,
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


# def tournament_selection(individuals: List[Any], pop_size: int, fraction: float = 0.1) -> List[Any]:
#     group_size = math.ceil(len(individuals) * fraction)
#     min_group_size = 2 if len(individuals) > 1 else 1
#     group_size = max(group_size, min_group_size)
#     chosen = []
#     for _ in range(pop_size):
#         group = random_selection(individuals, group_size)
#         best = min(group, key=lambda ind: ind.fitness)
#         chosen.append(best)
#         # best.graph.show()
#     # print('турнир')
#     # print('Родители')
#     # for i in chosen:
#     #     print(i.graph.operator.get_all_edges())

#     return chosen



def tournament_selection(individuals: List[Any], pop_size: int, fraction: float = 0.1) -> List[Any]:
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = 2 if len(individuals) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    for _ in range(pop_size):
        fl = True
        while fl:
            group = random_selection(individuals, group_size)
            best = min(group, key=lambda ind: ind.fitness)
            if not len(chosen) % 2:
                chosen.append(best)
                fl = False
            elif not check_iequv(best, chosen[-1]):
                chosen.append(best)           
                fl = False

    return chosen


def tournament_selection_for_MGA(individuals: List[Any], pop_size: int, fraction: float = 0.1) -> List[Any]:
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = 2 if len(individuals) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    n=0
    while len(chosen)<pop_size:
        group = random_selection(individuals, group_size)
        edges0 = (group[0]).graph.operator.get_all_edges()
        edges1 = (group[1]).graph.operator.get_all_edges()
        edges0_str = list(map(lambda x: str(x), edges0))
        edges1_str = list(map(lambda x: str(x), edges1))
        edges0_str.sort()
        edges1_str.sort()
        if n==10:
            best = min(group, key=lambda ind: ind.fitness)
            chosen.append(best)
            n=0        
            continue
        if edges0_str==edges1_str:
            n+=1
            continue        
        best = min(group, key=lambda ind: ind.fitness)
        chosen.append(best)

    return chosen

def nsga2_selection(individuals: List[Any], pop_size: int) -> List[Any]:
    chosen = tools.selNSGA2(individuals, pop_size)
    return chosen


def spea2_selection(individuals: List[Any], pop_size: int) -> List[Any]:
    chosen = tools.selSPEA2(individuals, pop_size)
    return chosen

def check_iequv(ind1, ind2):

    ## skeletons and immoralities
    (ske1, immor1) = get_skeleton_immor(ind1)
    (ske2, immor2) = get_skeleton_immor(ind2)

    ## comparison. 
    if len(ske1) != len(ske2) or len(immor1) != len(immor2):
        return False

    ## Note that the edges are undirected so we need to check both ordering
    for (n1, n2) in immor1:
        if (n1, n2) not in immor2 and (n2, n1) not in immor2:
            return False
    for (n1, n2) in ske1:
        if (n1, n2) not in ske2 and (n2, n1) not in ske2:
            return False
    return True


def get_skeleton_immor(ind):
    ## skeleton: a list of edges (undirected)
    skeleton = get_skeleton(ind)
    ## find immoralities
    immoral = set()
    for n in ind.graph.nodes:
        if n.nodes_from != None and len(n.nodes_from) > 1:
            perm = list(permutations(n.nodes_from, 2))
            for (per1, per2) in perm:
                p1 = per1.content["name"]
                p2 = per2.content["name"]
                if ((p1, p2) not in skeleton and (p2, p1) not in skeleton 
                    and (p1, p2) not in immoral and (p2, p1) not in immoral):
                    immoral.add((p1, p2))

    return (skeleton, immoral)    

def get_skeleton(ind):
    skeleton = set()
    edges = ind.graph.operator.get_all_edges()
    for e in edges:
        skeleton.add((e[0].content["name"], e[1].content["name"]))
        skeleton.add((e[1].content["name"], e[0].content["name"]))
    return skeleton
