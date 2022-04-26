from abc import ABC, abstractmethod
from typing import Protocol, Any, Union, Sequence, Set, Optional

import numpy as np
from deap.tools import HallOfFame

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.utils.pareto import ParetoFront
from fedot.core.utilities.data_structures import Comparable

ArchiveType = HallOfFame


class SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool: ...


class LexicographicKey(Comparable):
    KeyType = Optional[Union[int, float]]

    def __init__(self, *keys: KeyType):
        if not keys:
            raise ValueError('Expected at least one key for lexicographic key')
        self.keys = keys

    def __eq__(self, other: 'LexicographicKey') -> bool:
        return all(self._key_eq(k1, k2) for k1, k2 in zip(self.keys, other.keys))

    def __lt__(self, other: 'LexicographicKey') -> bool:
        for this_key, other_key in zip(self.keys, other.keys):
            if this_key is None and other_key is not None:
                return True
            elif this_key is not None and other_key is None:
                return False
            elif self._key_eq(this_key, other_key):
                continue
            else:
                return this_key < other_key

    @staticmethod
    def _key_eq(this_key: KeyType, other_key: KeyType) -> bool:
        return (this_key is None and other_key is None
                or np.isclose(this_key, other_key).all())


def ind_compound_key(individual: Individual) -> LexicographicKey:
    """Return comparison key combining fitness and structural complexity.
    Makes sense only for single-objective fitness."""
    return LexicographicKey(*individual.fitness.values, individual.graph.length)


def best_individual(individuals: Sequence[Individual]) -> Individual:
    """Find the best individual given lexicographic key of (fitness, structural_complexity)."""
    return min(individuals, key=ind_compound_key)


class ImprovementWatcher(ABC):
    """Interface that allows to check if optimization progresses or stagnates."""

    @property
    @abstractmethod
    def last_improved(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def metric_improved(self, metric: MetricsEnum) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def quality_improved(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def complexity_improved(self) -> bool:
        raise NotImplementedError()


class GenerationKeeper(ImprovementWatcher):
    """Generation keeper that primarily tracks number of generations and stagnation duration.

    :param initial_generation: first generation;
     NB: if None then keeper is created in inconsistent state!
    :param is_multi_objective:
    :param archive:
    :param keep_n_best: How many best individuals to keep from all generations.
    NB: relevant only for single-objective optimization!
    """

    def __init__(self,
                 initial_generation: PopulationT = None,
                 is_multi_objective: bool = False,
                 archive: ArchiveType = None,
                 keep_n_best: int = 1):
        self._generation_num = -1
        self._stagnation_counter = 0
        if initial_generation is not None:
            self.append(initial_generation)
        if archive is None:
            if is_multi_objective:
                archive = ParetoFront()
            else:
                archive = HallOfFame(maxsize=keep_n_best,
                                     similar=lambda ind1, ind2: ind1.fitness == ind2.fitness)
        elif not isinstance(archive, HallOfFame):
            raise TypeError(f'Invalid archive type. Expected HallOfFame, got {type(archive)}')
        self.archive = archive

    @property
    def best_individuals(self) -> Sequence[Individual]:
        return self.archive.items

    @property
    def generation_num(self) -> int:
        return self._generation_num

    @property
    def stagnation_length(self) -> int:
        return self._stagnation_counter

    @property
    def last_improved(self) -> bool:
        return self._stagnation_counter == 0

    @property
    def archive_fitness(self) -> Set[int]:
        return {ind.fitness for ind in self.archive.items}

    def append(self, population: PopulationT):
        previous_archive_fitness = self.archive_fitness
        self.archive.update(population)
        improved = previous_archive_fitness != self.archive_fitness
        self._stagnation_counter = 0 if improved else self._stagnation_counter + 1
        self._generation_num += 1  # becomes 0 on first population
