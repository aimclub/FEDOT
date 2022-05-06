import math

from .parameter import AdaptiveParameter
from fedot.core.utilities.data_structures import BidirectionalIterator
from fedot.core.optimisers.generation_keeper import ImprovementWatcher

PopulationSize = AdaptiveParameter[int]


class ConstRatePopulationSize(PopulationSize):
    def __init__(self, pop_size: int, offspring_rate: float):
        self._offspring_rate = offspring_rate
        self._initial = pop_size

    @property
    def initial(self) -> int:
        return self._initial

    def next(self, current: int) -> int:
        return math.ceil(current * self._offspring_rate)


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

    def next(self, current: int) -> int:
        fitness_improved = self._improvements.is_quality_improved
        complexity_decreased = self._improvements.is_complexity_improved
        progress_in_both_goals = fitness_improved and complexity_decreased
        no_progress = not fitness_improved and not complexity_decreased

        if progress_in_both_goals and current > 2:
            if self._iterator.has_prev():
                current = self._iterator.prev()
        elif no_progress:
            if self._iterator.has_next():
                current = self._iterator.next()

        return current
