from abc import ABC, abstractmethod
from typing import Dict, Iterable, Sequence, Type, Optional, Any, Callable

import numpy as np
import datetime

from fedot.core.optimisers.fitness import is_metric_worse, Fitness
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.opt_history_objects.individual import Individual
from .individuals_containers import HallOfFame, ParetoFront

PARETO_MAX_POP_SIZE_MULTIPLIER = 10

class ImprovementWatcher(ABC):
    """Interface that allows to check if optimization progresses or stagnates."""

    @property
    def stagnation_iter_count(self) -> int:
        """Returns number of generations for which any metrics has not improved."""
        raise NotImplementedError()

    @property
    def stagnation_time_duration(self) -> float:
        """Returns time duration for which any metrics has not improved."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_any_improved(self) -> bool:
        """Check if any of the metrics has improved."""
        raise NotImplementedError()

    @abstractmethod
    def is_metric_improved(self, metric_id) -> bool:
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


def _individuals_same(ind1: Individual, ind2: Individual) -> bool:
    return (ind1.fitness == ind2.fitness and
            ind1.native_generation == ind2.native_generation and
            ind1.graph == ind2.graph)


class GenerationKeeper(ImprovementWatcher):
    """Generation keeper that primarily tracks number of generations and stagnation duration.

    Args:
        objective: Objective that specifies metrics and if it's multi objective optimization.
        keep_n_best: How many best individuals to keep from all generations.
         NB: relevant only for single-objective optimization.
        initial_generation: Optional first generation;
         NB: if None then keeper is created in inconsistent state and requires an initial .append().
        similarity_criteria: a function that in the case of multi-objective optimization
         tells the Pareto front whether two individuals are similar, optional.
    """

    def __init__(self,
                 objective: Optional[Objective] = None,
                 keep_n_best: int = 1,
                 initial_generation: PopulationT = None,
                 similarity_criteria: Callable = _individuals_same):
        self._generation_num = 0  # 0 means state before initial generation is added
        self._stagnation_counter = 0  # Initialized in non-stagnated state
        self._stagnation_start_time = datetime.datetime.now()

        self._objective = objective
        self._metrics_improvement: Dict[Any, bool] = {}
        self._reset_metrics_improvement()

        if objective.is_multi_objective:
            self.archive = ParetoFront(maxsize=keep_n_best * PARETO_MAX_POP_SIZE_MULTIPLIER,
                                       similar=similarity_criteria)
        else:
            self.archive = HallOfFame(maxsize=keep_n_best)

        if initial_generation is not None:
            self.append(initial_generation)

    @property
    def stagnation_start_time(self):
        return self._stagnation_start_time

    @property
    def best_individuals(self) -> Sequence[Individual]:
        return self.archive.items

    @property
    def generation_num(self) -> int:
        return self._generation_num

    @property
    def stagnation_iter_count(self) -> int:
        return self._stagnation_counter

    @property
    def stagnation_time_duration(self) -> float:
        return (datetime.datetime.now() - self._stagnation_start_time).seconds / 60

    @property
    def is_any_improved(self) -> bool:
        return any(self._metrics_improvement.values())

    def is_metric_improved(self, metric_id) -> bool:
        return self._metrics_improvement[metric_id]

    @property
    def is_quality_improved(self) -> bool:
        return any(self._metrics_improvement[metric_id]
                   for metric_id in self._objective.quality_metrics)

    @property
    def is_complexity_improved(self) -> bool:
        return any(self._metrics_improvement[metric_id]
                   for metric_id in self._objective.complexity_metrics)

    @property
    def _metric_ids(self) -> Iterable[Any]:
        return self._objective.metric_names

    def append(self, population: PopulationT):
        previous_archive_fitness = self._archive_fitness()
        self.archive.update(population)
        self._update_improvements(previous_archive_fitness)

    def _archive_fitness(self) -> Dict[Any, Sequence[float]]:
        archive_pop_metrics = (ind.fitness.values for ind in self.archive.items)
        archive_fitness_per_metric = zip(*archive_pop_metrics)  # transpose nested array
        archive_fitness_per_metric = dict(zip(self._metric_ids, archive_fitness_per_metric))
        return archive_fitness_per_metric

    def _update_improvements(self, previous_metric_archive):
        self._reset_metrics_improvement()
        current_metric_archive = self._archive_fitness()
        for metric in self._metric_ids:
            # NB: Assuming we perform minimisation, so worst==max
            previous_worst = np.max(previous_metric_archive.get(metric, np.inf))
            current_worst = np.max(current_metric_archive.get(metric, np.inf))
            # archive metric has improved if metric of its worst individual has improved
            if is_metric_worse(previous_worst, current_worst):
                self._metrics_improvement[metric] = True

        self._generation_num += 1  # becomes 1 on first population
        self._stagnation_start_time = datetime.datetime.now() \
            if self.is_any_improved or self._generation_num == 1 else self._stagnation_start_time
        self._stagnation_counter = 0 if self.is_any_improved else self._stagnation_counter + 1

    def _reset_metrics_improvement(self):
        self._metrics_improvement = {metric_id: False for metric_id in self._metric_ids}

    def __str__(self) -> str:
        ff = self._objective.format_fitness
        fitnesses = [ff(ind.fitness) for ind in self.best_individuals]
        return f'{self.archive.__class__.__name__} archive fitness ({len(fitnesses)}): {fitnesses}'
