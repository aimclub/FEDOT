import functools
import math
from copy import copy
from random import choice, randint, sample
from typing import Callable, List, Optional

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class SelectionTypesEnum(Enum):
    tournament = 'tournament'
    spea2 = 'spea2'


class Selection(Operator):
    def __call__(self, population: PopulationT) -> PopulationT:
        """
        Selection of individuals based on specified type of selection
        :param population: A list of individuals to select from.
        """
        selection_type = choice(self.parameters.selection_types)
        return self._selection_by_type(selection_type)(population, self.parameters.pop_size)

    @staticmethod
    def _selection_by_type(selection_type: SelectionTypesEnum) -> Callable[[PopulationT, int], PopulationT]:
        selections = {
            SelectionTypesEnum.tournament: tournament_selection,
            SelectionTypesEnum.spea2: spea2_selection
        }
        if selection_type in selections:
            return selections[selection_type]
        else:
            raise ValueError(f'Required selection not found: {selection_type}')

    def individuals_selection(self, individuals: PopulationT) -> PopulationT:
        return self.__call__(individuals)


def default_selection_behaviour(selection_func: Optional[Callable] = None, *, ensure_unique: bool = True,
                                populate_by_single: bool = True):
    def func_wrapper(func: Callable):
        @functools.wraps(func)
        def wrapper(individuals: PopulationT, pop_size: int, *args, **kwargs):
            if ensure_unique:
                individuals = list({ind.uid: ind for ind in individuals}.values())
            else:
                individuals = copy(individuals)

            if populate_by_single and len(individuals) == 1:
                return individuals * pop_size

            if len(individuals) <= pop_size:
                return individuals

            return func(individuals, pop_size, *args, **kwargs)
        return wrapper

    if selection_func:
        return func_wrapper(selection_func)  # Allows to decorate without args.
    return func_wrapper  # Allows to decorate with args but no selection_func specified.


@default_selection_behaviour
def tournament_selection(individuals: PopulationT, pop_size: int, fraction: float = 0.1) -> PopulationT:
    individuals = list(individuals)  # don't modify original
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = min(2, len(individuals))
    group_size = max(group_size, min_group_size)
    chosen = []
    iterations_limit = pop_size * 10
    for _ in range(iterations_limit):
        if len(chosen) >= pop_size:
            break
        group = sample(individuals, min(group_size, len(individuals)))
        best = max(group, key=lambda ind: ind.fitness)
        individuals.remove(best)
        chosen.append(best)
    return chosen


@default_selection_behaviour
def random_selection(individuals: PopulationT, pop_size: int) -> PopulationT:
    return sample(individuals, pop_size)


# Code of spea2 selection is modified part of DEAP library (Library URL: https://github.com/DEAP/deap).
@default_selection_behaviour
def spea2_selection(individuals: PopulationT, pop_size: int) -> PopulationT:
    """
    Apply SPEA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *n* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *n* will have no effect other
    than sorting the population according to a strength Pareto scheme. The
    list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
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
                    val = \
                        individuals[i].fitness.values[idx] - \
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
        distances = [[0.0] * inds_len for _ in range(inds_len)]
        sorted_indices = [[0] * inds_len for _ in range(inds_len)]
        for i in range(inds_len):
            for j in range(i + 1, inds_len):
                dist = 0.0
                for idx in range(fitness_len):
                    val = \
                        individuals[chosen_indices[i]].fitness.values[idx] - \
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


# Auxiliary algorithmic functions for spea2_selection
# This code is a part of DEAP library (Library URL: https://github.com/DEAP/deap).
def _randomized_select(array: List[float], begin: int, end: int, i: float) -> float:
    """
    Allows to select the ith smallest element from array without sorting it.
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
