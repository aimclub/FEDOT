import math
from typing import Optional

from .parameter import AdaptiveParameter
from fedot.core.utilities.data_structures import BidirectionalIterator
from fedot.core.utilities.sequence_iterator import fibonacci_sequence, SequenceIterator
from ..gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.archive.generation_keeper import ImprovementWatcher
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT

PopulationSize = AdaptiveParameter[int]


class ConstRatePopulationSize(PopulationSize):
    def __init__(self, pop_size: int, offspring_rate: float, max_pop_size: Optional[int] = None):
        self._offspring_rate = offspring_rate
        self._initial = pop_size
        self._max_size = max_pop_size

    @property
    def initial(self) -> int:
        return self._initial

    def next(self, population: PopulationT) -> int:
        pop_size = len(population)
        if not self._max_size or pop_size < self._max_size:
            pop_size += math.ceil(pop_size * self._offspring_rate)
        if self._max_size:
            pop_size = min(pop_size, self._max_size)
        return pop_size


class AdaptivePopulationSize(PopulationSize):
    def __init__(self,
                 improvement_watcher: ImprovementWatcher,
                 progression_iterator: BidirectionalIterator[int]):
        self._improvements = improvement_watcher
        self._iterator = progression_iterator
        self._initial = self._iterator.next() if self._iterator.has_next() else self._iterator.prev()

    @property
    def initial(self) -> int:
        return self._initial

    def next(self, population: PopulationT) -> int:
        fitness_improved = self._improvements.is_quality_improved
        complexity_decreased = self._improvements.is_complexity_improved
        progress_in_both_goals = fitness_improved and complexity_decreased
        no_progress = not fitness_improved and not complexity_decreased

        pop_size = len(population)
        if progress_in_both_goals and pop_size > 2:
            if self._iterator.has_prev():
                pop_size = self._iterator.prev()
        elif no_progress:
            if self._iterator.has_next():
                pop_size = self._iterator.next()

        return pop_size


def init_adaptive_pop_size(requirements: GPGraphOptimizerParameters,
                           improvement_watcher: ImprovementWatcher) -> PopulationSize:
    genetic_scheme_type = requirements.genetic_scheme_type
    if genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
        pop_size = ConstRatePopulationSize(
            pop_size=requirements.pop_size,
            offspring_rate=1.0,
            max_pop_size=requirements.max_pop_size,
        )
    elif genetic_scheme_type == GeneticSchemeTypesEnum.generational:
        pop_size = ConstRatePopulationSize(
            pop_size=requirements.pop_size,
            offspring_rate=requirements.offspring_rate,
            max_pop_size=requirements.max_pop_size,
        )
    elif genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
        pop_size_progression = SequenceIterator(sequence_func=fibonacci_sequence,
                                                start_value=requirements.pop_size,
                                                min_sequence_value=1,
                                                max_sequence_value=requirements.max_pop_size)
        pop_size = AdaptivePopulationSize(improvement_watcher, pop_size_progression)
    else:
        raise ValueError(f"Unknown genetic type scheme {genetic_scheme_type}")
    return pop_size
