from abc import abstractmethod
from collections.abc import Hashable
from typing import Sequence, Any, Union, Optional, Tuple

import numpy as np

from fedot.core.utilities.data_structures import Comparable


class Fitness(Comparable):
    """Abstracts comparable fitness values that can be in invalid state.
    Fitness comparison handles invalid fitness: invalid fitness is never
    less than any other fitness. Fitness implementations must ensure this contract."""

    @property
    @abstractmethod
    def values(self) -> Sequence[float]:
        raise NotImplementedError()

    @values.setter
    @abstractmethod
    def values(self, new_values: Optional[Sequence[float]]):
        raise NotImplementedError()

    @values.deleter
    @abstractmethod
    def values(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def valid(self) -> bool:
        """Assess if a fitness is valid or not."""
        raise NotImplementedError()

    def dominates(self, other: 'Fitness', selector: Any = None) -> bool:
        """Implementation-specific test for fitness domination.
        By default behaves same as less-than operator for valid fitness.

        :param other: another fitness for dominates test.
        :param selector: optionally specifies which objectives of the fitness to use.
        """
        return self < other

    def reset(self):
        del self.values

    def __hash__(self) -> int:
        # try to avoid numeric precision errors in hash comparisons
        vals = tuple(np.round(value, 8) if value is not None else None
                     for value in self.values)
        return hash(vals)

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s%r" % (self.__module__, self.__class__.__name__,
                            tuple(self.values) if self.valid else tuple())

    def __eq__(self, other: 'Fitness') -> bool:
        return (isinstance(other, self.__class__) and
                self.valid and other.valid and
                self.allclose(self.values, other.values))

    @staticmethod
    def allclose(values1, values2) -> bool:
        return np.allclose(values1, values2, rtol=1e-8, atol=1e-10)


class PrioritisedFitness(Fitness):
    """Implements lexicographic comparison on one or multiple fitness values.
    Primary value is allowed to be None and determines if fitness is valid.
    Secondary values must be not None and define supplementary metrics."""

    def __init__(self, primary_value: Optional[float] = None, *secondary_values: float):
        self._values: Tuple = (primary_value, *secondary_values)

    @property
    def values(self) -> Sequence[float]:
        return self._values

    @values.setter
    def values(self, new_values: Optional[Sequence[float]]):
        if new_values is None:
            self.reset()
        if any(secondary_value is None for secondary_value in new_values[1:]):
            raise ValueError('Secondary values must not be None for prioritized fitness')
        self._values = new_values

    @values.deleter
    def values(self):
        self._values = (None,)

    @property
    def valid(self) -> bool:
        return self._values[0] is not None

    def __hash__(self) -> int:
        # __hash__ required explicit super() call
        return super().__hash__()

    def __lt__(self, other: 'PrioritisedFitness') -> bool:
        # NB: in the case of both invalid the other takes precedence
        if not self.valid:
            return True
        elif not other.valid:
            return False
        # both are valid
        return self._values < other._values  # lexicographic comparison

    def __str__(self) -> str:
        if len(self._values) == 1:
            return str(self._values[0])
        else:
            return str(self._values)


def single_value_fitness(value: Optional[float]) -> PrioritisedFitness:
    """Alias for creating simple single-value fitness."""
    return PrioritisedFitness(primary_value=value)


def none_fitness() -> PrioritisedFitness:
    """Alias for creating default-initialised single-value fitness."""
    return PrioritisedFitness(primary_value=None)
