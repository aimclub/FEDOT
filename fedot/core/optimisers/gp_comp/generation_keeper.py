from abc import ABC, abstractmethod
from typing import Sequence, Type, Iterable, Dict

import numpy as np
from deap.tools import HallOfFame, ParetoFront

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.repository.quality_metrics_repository import MetricsEnum, QualityMetricsEnum, ComplexityMetricsEnum


class ImprovementWatcher(ABC):
    """Interface that allows to check if optimization progresses or stagnates."""

    @property
    @abstractmethod
    def is_any_improved(self) -> bool:
        """Check if any of the metrics has improved."""
        raise NotImplementedError()

    @abstractmethod
    def is_metric_improved(self, metric: MetricsEnum) -> bool:
        """Check if specified metric has improved."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_quality_improved(self) -> bool:
        """Check if any of the quality metrics has improved."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_complexity_improved(self) -> bool:
        """Check if any of the complexity metrics has improved."""
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
                 metrics: Sequence[MetricsEnum] = (),
                 is_multi_objective: bool = False,
                 keep_n_best: int = 1,
                 initial_generation: PopulationT = None):
        self._generation_num = -1  # -1 means state before initial generation is added
        self._stagnation_counter = 0  # Initialized in non-stagnated state
        self._metrics_improvement = {metric_id: False for metric_id in metrics}
        self.archive = ParetoFront() if is_multi_objective else HallOfFame(maxsize=keep_n_best)

        if initial_generation is not None:
            self.append(initial_generation)

    @property
    def best_individuals(self) -> Sequence[Individual]:
        return self.archive.items

    @property
    def generation_num(self) -> int:
        return self._generation_num

    @property
    def stagnation_duration(self) -> int:
        return self._stagnation_counter

    @property
    def is_any_improved(self) -> bool:
        return any(self._metrics_improvement.values())

    def is_metric_improved(self, metric: MetricsEnum) -> bool:
        return self._metrics_improvement.get(metric, False)

    @property
    def is_quality_improved(self) -> bool:
        return self._is_metric_kind_improved(QualityMetricsEnum)

    @property
    def is_complexity_improved(self) -> bool:
        return self._is_metric_kind_improved(ComplexityMetricsEnum)

    def _is_metric_kind_improved(self, metric_cls: Type[MetricsEnum]) -> bool:
        """Check that any metric of the specified subtype of MetricsEnum has improved."""
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

        self._stagnation_counter = 0 if self.is_any_improved else self._stagnation_counter + 1
        self._generation_num += 1  # becomes 1 on first population

    def _reset_metrics_improvement(self):
        self._metrics_improvement = {metric_id: False for metric_id in self._metric_ids}

    def __str__(self) -> str:
        return (f'{self.archive.__class__.__name__} archive fitness: '
                f'{[item.fitness.values for item in self.best_individuals]}')
