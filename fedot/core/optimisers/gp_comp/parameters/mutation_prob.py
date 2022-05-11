import numpy as np

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.parameters.parameter import AdaptiveParameter


class AdaptiveMutationProb(AdaptiveParameter[float]):

    def __init__(self, default_prob: float = 0.5):
        self._current_std = 0.
        self._max_std = 0.
        self._default_prob = default_prob

    @property
    def initial(self) -> float:
        return self._default_prob

    def next(self, population: PopulationT) -> float:
        self._update_std(population)

        if len(population) < 2:
            mutation_prob = 1.0
        elif self._max_std == 0:
            mutation_prob = self._default_prob
        else:
            mutation_prob = 1. - (self._current_std / self._max_std)
        return mutation_prob

    def _update_std(self, population: PopulationT):
        self._current_std = self._calc_std(population)
        self._max_std = max(self._current_std, self._max_std)

    @staticmethod
    def _calc_std(population: PopulationT) -> float:
        return float(np.std([ind.fitness.value for ind in population]))
