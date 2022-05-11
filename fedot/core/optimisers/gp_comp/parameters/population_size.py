import math

from .parameter import AdaptiveParameter
from fedot.core.utilities.data_structures import BidirectionalIterator
from fedot.core.optimisers.generation_keeper import ImprovementWatcher
from ..operators.operator import PopulationT

PopulationSize = AdaptiveParameter[int]


class ConstRatePopulationSize(PopulationSize):
    def __init__(self, pop_size: int, offspring_rate: float):
        self._offspring_rate = offspring_rate
        self._initial = pop_size

    @property
    def initial(self) -> int:
        return self._initial

    def next(self, population: PopulationT) -> int:
        return math.ceil(len(population) * self._offspring_rate)


class AdaptivePopulationSize(PopulationSize):
    def __init__(self,
                 improvement_watcher: ImprovementWatcher,
                 progression_iterator: BidirectionalIterator[int]):
        self._improvements = improvement_watcher
        self._iterator = progression_iterator
        self._initial = next(self._iterator)

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
