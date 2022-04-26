from abc import ABC, abstractmethod
from typing import Protocol, Any, Union, Sequence, Set, Optional, Type, Iterable, Dict

import numpy as np
from deap.tools import HallOfFame

from fedot.core.optimisers.fitness.fitness import Fitness
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.utils.pareto import ParetoFront
from fedot.core.repository.quality_metrics_repository import MetricsEnum, QualityMetricsEnum, ComplexityMetricsEnum
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
                 metrics: Sequence[MetricsEnum] = (),
                 archive: ArchiveType = None,
                 keep_n_best: int = 1):
        self._generation_num = -1  # -1 means no generations
        self._stagnation_counter = 0  # Initialized in non-stagnated state
        self._metrics_improvement = {metric_id: False for metric_id in metrics}

        if archive is None:
            archive = ParetoFront() if is_multi_objective else HallOfFame(maxsize=keep_n_best)
        elif not isinstance(archive, HallOfFame):
            raise TypeError(f'Invalid archive type. Expected HallOfFame, got {type(archive)}')
        self.archive = archive

        if initial_generation is not None:
            self.append(initial_generation)

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
        return any(self._metrics_improvement.values())

    def metric_improved(self, metric: MetricsEnum) -> bool:
        return self._metrics_improvement.get(metric, False)

    @property
    def quality_improved(self) -> bool:
        return self._metric_kind_improved(QualityMetricsEnum)

    @property
    def complexity_improved(self) -> bool:
        return self._metric_kind_improved(ComplexityMetricsEnum)

    def _metric_kind_improved(self, metric_cls: Type[MetricsEnum]) -> bool:
        return any(improved for metric, improved in self._metrics_improvement.items()
                   if isinstance(metric, metric_cls))

    @property
    def _metric_ids(self) -> Iterable[MetricsEnum]:
        return self._metrics_improvement.keys()

    def append(self, population: PopulationT):
        previous_archive_fitness = self._archive_fitness()
        self.archive.update(population)
        self._update_improvements(previous_archive_fitness)

    def _archive_fitness(self) -> Dict[MetricsEnum, Sequence[float]]:
        archive_pop_metrics = (ind.fitness.values for ind in self.archive.items)
        archive_fitness_per_metric = zip(*archive_pop_metrics)  # transpose nested array
        archive_fitness_per_metric = dict(zip(self._metric_ids, archive_fitness_per_metric))
        return archive_fitness_per_metric

    def _update_improvements(self, previous_metric_archive):
        self._reset_metrics_improvement()
        current_metric_archive = self._archive_fitness()
        for metric in self._metric_ids:
            # NB: Assuming we perform maximisation, so worst==minimum
            previous_worst = np.min(previous_metric_archive.get(metric, -np.inf))
            current_worst = np.min(current_metric_archive.get(metric, -np.inf))
            # archive metric has improved if metric of its worst individual has improved
            if current_worst > previous_worst:
                self._metrics_improvement[metric] = True

        self._stagnation_counter = 0 if self.last_improved else self._stagnation_counter + 1
        self._generation_num += 1  # becomes 1 on first population

    def _reset_metrics_improvement(self):
        self._metrics_improvement = {metric_id: False for metric_id in self._metric_ids}
