from abc import abstractmethod
from typing import Sequence, Any, Optional, Tuple

import numpy as np

from fedot.core.utilities.data_structures import Comparable


class Fitness(Comparable):
    """Abstracts comparable fitness values that can be in invalid state.

    Importantly, Fitness comparison is semantic: `more-than` means `better-than`.
    Fitness can be compared using standard operators ``>``, ``<``, ``>=``, etc.

    Fitness comparison handles invalid fitness: invalid fitness is never better
    than any other fitness. Fitness implementations must ensure this contract.

    Default Fitness comparison is lexicographic (even for multi-objective fitness,
    to ensure total ordering). For proper comparison of multi-objective fitness
    use method `dominates`.
    """

    @property
    def value(self) -> Optional[float]:
        """Return primary fitness value"""
        return self.values[0]

    @property
    @abstractmethod
    def values(self) -> Sequence[float]:
        """Return individual metric values.
        Returned values are already weighted, if weights are used."""
        raise NotImplementedError()

    @values.setter
    @abstractmethod
    def values(self, new_values: Optional[Sequence[float]]):
        """Assign individual metric values. Accepts unweighted values."""
        raise NotImplementedError()

    @values.deleter
    @abstractmethod
    def values(self):
        """Clear internal metric values."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def weights(self) -> Sequence[float]:
        """Return weights used for weighting individual metrics."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def valid(self) -> bool:
        """Assess if a fitness is valid or not."""
        raise NotImplementedError()

    def dominates(self, other: 'Fitness', selector: Any = None) -> bool:
        """Implementation-specific test for fitness domination.
        By the default behaves same as less-than operator for valid fitness.
        Less means worse; so the better fitness is a dominating one.

        :param other: another fitness for dominates test.
        :param selector: optionally specifies which objectives of the fitness to use.
        """
        return self > other

    def reset(self):
        del self.values

    def __lt__(self, other: 'Fitness') -> bool:
        """'Less-than' for fitness means 'worse-than'.
        NB: in the case of both invalid the other takes precedence
        """
        if not self.valid:
            # invalid self is worse
            return True
        elif not other.valid:
            # valid self is NOT worse than invalid other
            return False
        # if both are valid then compare normally
        return is_metric_worse(self.values, other.values)  # lexicographic comparison

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


class SingleObjFitness(Fitness):
    """Single-objective fitness with optional supplementary values
    for distinguishing cases when primary fitness values are equal.
    This fitness implements lexicographic comparison on its fitness values.

    :param primary_value: Primary fitness metric, may be None. It determines if fitness is valid.
    :param supplementary_values: Define supplementary metrics, must not be None.
    """

    def __init__(self, primary_value: Optional[float] = None, *supplementary_values: float):
        self._values: Tuple = (primary_value, *supplementary_values)

    @property
    def values(self) -> Sequence[float]:
        return self._values

    @values.setter
    def values(self, new_values: Optional[Sequence[float]]):
        if new_values is None:
            self.reset()
        if any(secondary_value is None for secondary_value in new_values[1:]):
            raise ValueError('Secondary values must not be None for prioritized fitness')
        self._values = tuple(new_values)

    @values.deleter
    def values(self):
        self._values = (None,)

    @property
    def weights(self) -> Sequence[float]:
        # Return default weights
        return (1.,) * len(self.values)

    @property
    def valid(self) -> bool:
        return self._values[0] is not None

    def __hash__(self) -> int:
        # __hash__ required explicit super() call
        return super().__hash__()

    def __str__(self) -> str:
        # For single objective return only the primary value
        return str(round(self.value, 4)) if self.value is not None else 'null_fitness'


def null_fitness() -> SingleObjFitness:
    """Alias for creating default-initialised single-value fitness."""
    return SingleObjFitness(primary_value=None)


def is_metric_worse(left_value, right_value) -> bool:
    if isinstance(left_value, Fitness):
        # Fitness object already handles metric comparison in the right way
        return left_value < right_value
    else:
        # Less is better -- minimisation task on raw metric values
        return left_value > right_value
