import math
from random import choice, randint
from typing import TYPE_CHECKING, List, Iterable, Sequence, Tuple

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.utilities.data_structures import ComparableEnum as Enum

from itertools import permutations

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams


class SelectionTypesEnum(Enum):
    tournament = 'tournament'
    spea2 = 'spea2'


def selection(types: List[SelectionTypesEnum], population: List[Individual], pop_size: int,
              params: 'GraphGenerationParams') -> List[Individual]:
    """
    Selection of individuals based on specified type of selection
    :param types: The set of selection types
    :param population: A list of individuals to select from.
    :param pop_size: The number of individuals to select.
    :param params: params for graph generation and convertation
    """
    selection_by_type = {
        SelectionTypesEnum.tournament: tournament_selection,
        SelectionTypesEnum.spea2: spea2_selection
    }

    selection_type = choice(types)
    if selection_type in selection_by_type:
        selected = selection_by_type[selection_type](population, pop_size)
        return selected
    else:
        raise ValueError(f'Required selection not found: {selection_type}')


def individuals_selection(types: List[SelectionTypesEnum], individuals: List[Individual], pop_size: int,
                          graph_params: 'GraphGenerationParams') -> List[Individual]:
    if pop_size == len(individuals):
        chosen = individuals
    else:
        chosen = []
        remaining_individuals = individuals
        individuals_pool_size = len(individuals)
        n_iter = 0
        while len(chosen) < pop_size and n_iter < pop_size * 10 and remaining_individuals:
            individual = selection(types, remaining_individuals, pop_size=1, params=graph_params)[0]
            if individual.uid not in (c.uid for c in chosen):
                chosen.append(individual)
                if pop_size <= individuals_pool_size:
                    remaining_individuals.remove(individual)
            n_iter += 1
    return chosen


def random_selection(individuals: List[Individual], pop_size: int) -> List[Individual]:
    chosen = []
    n_iter = 0
    while len(chosen) < pop_size and n_iter < pop_size * 10:
        if not individuals:
            return []
        if len(individuals) <= 1:
            return [individuals[0]] * pop_size
        individual = choice(individuals)
        if individual.uid not in (c.uid for c in chosen):
            chosen.append(individual)
    return chosen


# def tournament_selection(individuals: List[Individual], pop_size: int, fraction: float = 0.1) -> List[Individual]:
#     group_size = math.ceil(len(individuals) * fraction)
#     min_group_size = 2 if len(individuals) > 1 else 1
#     group_size = max(group_size, min_group_size)
#     chosen = []
#     n_iter = 0

#     while len(chosen) < pop_size and n_iter < pop_size * 10:
#         group = random_selection(individuals, group_size)
#         best = max(group, key=lambda ind: ind.fitness)
#         if best.uid not in (c.uid for c in chosen):
#             chosen.append(best)
#         n_iter += 1

#     return chosen

# bamt
def tournament_selection(individuals: List[Individual], pop_size: int, fraction: float = 0.1) -> List[Individual]:
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = 2 if len(individuals) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    n_iter = 0
    while len(chosen) < pop_size and n_iter < pop_size * 10:
        fl = True
        while fl:
            group = random_selection(individuals, group_size)
            best = max(group, key=lambda ind: ind.fitness)
            if not len(chosen) % 2:
                chosen.append(best)
                fl = False
            elif not check_iequv(best, chosen[-1]):
                chosen.append(best)           
                fl = False
            n_iter += 1
    
    return chosen

# Code of spea2 selection is modified part of DEAP library (Library URL: https://github.com/DEAP/deap).
def spea2_selection(individuals: List[Individual], pop_size: int) -> List[Individual]:
    """
    Apply SPEA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *n* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *n* will have no effect other
    than sorting the population according to a strength Pareto scheme. The
    list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param pop_size: The number of individuals to select.
    :returns: A list of selected individuals
    """

    inds_len = len(individuals)
    fitness_len = len(individuals[0].fitness.values)
    inds_len_sqrt = math.sqrt(inds_len)
    strength_fits = [0] * inds_len
    fits = [0] * inds_len
    dominating_inds = [list() for _ in range(inds_len)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1:], i + 1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    for i in range(inds_len):
        for j in dominating_inds[i]:
            fits[i] += strength_fits[j]

    # Choose all non-dominated individuals
    chosen_indices = [i for i in range(inds_len) if fits[i] < 1]

    if len(chosen_indices) < pop_size:  # The archive is too small
        for i in range(inds_len):
            distances = [0.0] * inds_len
            for j in range(i + 1, inds_len):
                dist = 0.0
                for idx in range(fitness_len):
                    val = individuals[i].fitness.values[idx] - \
                          individuals[j].fitness.values[idx]
                    dist += val * val
                distances[j] = dist
            kth_dist = _randomized_select(distances, 0, inds_len - 1, inds_len_sqrt)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density

        next_indices = [(fits[i], i) for i in range(inds_len)
                        if i not in chosen_indices]
        next_indices.sort()
        # print next_indices
        chosen_indices += [i for _, i in next_indices[:pop_size - len(chosen_indices)]]

    elif len(chosen_indices) > pop_size:  # The archive is too large
        inds_len = len(chosen_indices)
        distances = [[0.0] * inds_len for i in range(inds_len)]
        sorted_indices = [[0] * inds_len for i in range(inds_len)]
        for i in range(inds_len):
            for j in range(i + 1, inds_len):
                dist = 0.0
                for idx in range(fitness_len):
                    val = individuals[chosen_indices[i]].fitness.values[idx] - \
                          individuals[chosen_indices[j]].fitness.values[idx]
                    dist += val * val
                distances[i][j] = dist
                distances[j][i] = dist
            distances[i][i] = -1

        # Insert sort is faster than quick sort for short arrays
        for i in range(inds_len):
            for j in range(1, inds_len):
                idx = j
                while idx > 0 and distances[i][j] < distances[i][sorted_indices[i][idx - 1]]:
                    sorted_indices[i][idx] = sorted_indices[i][idx - 1]
                    idx -= 1
                sorted_indices[i][idx] = j

        size = inds_len
        to_remove = []
        while size > pop_size:
            # Search for minimal distance
            min_pos = 0
            for i in range(1, inds_len):
                for j in range(1, size):
                    dist_i_sorted_j = distances[i][sorted_indices[i][j]]
                    dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

                    if dist_i_sorted_j < dist_min_sorted_j:
                        min_pos = i
                        break
                    elif dist_i_sorted_j > dist_min_sorted_j:
                        break

            # Remove minimal distance from sorted_indices
            for i in range(inds_len):
                distances[i][min_pos] = float("inf")
                distances[min_pos][i] = float("inf")

                for j in range(1, size - 1):
                    if sorted_indices[i][j] == min_pos:
                        sorted_indices[i][j] = sorted_indices[i][j + 1]
                        sorted_indices[i][j + 1] = min_pos

            # Remove corresponding individual from chosen_indices
            to_remove.append(min_pos)
            size -= 1

        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]

    return [individuals[i] for i in chosen_indices]


def crossover_parents_selection(population: Sequence[Individual]) -> Iterable[Tuple[Individual, Individual]]:
    return zip(population[::2], population[1::2])


# Auxiliary algorithmic functions for spea2_selection
# This code is a part of DEAP library (Library URL: https://github.com/DEAP/deap).
def _randomized_select(array: List[float], begin: int, end: int, i: float) -> float:
    """Allows to select the ith smallest element from array without sorting it.
    Runtime is expected to be O(n).
    """
    if begin == end:
        return array[begin]
    q = _randomized_partition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomized_select(array, begin, q, i)
    else:
        return _randomized_select(array, q + 1, end, i - k)


def _randomized_partition(array: List[float], begin: int, end: int) -> int:
    i = randint(begin, end)
    array[begin], array[i] = array[i], array[begin]
    return _partition(array, begin, end)


def _partition(array: List[float], begin: int, end: int) -> int:
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j

# bamt
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