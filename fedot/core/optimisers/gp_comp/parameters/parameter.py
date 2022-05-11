from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT

T = TypeVar('T')


class AdaptiveParameter(ABC, Generic[T]):
    """Abstract interface for parameters of evolutionary algorithm and its Operators.
    Adaptive parameter is defined by `initial` value and subsequent `next` values.
    Parameter can potentially change on each call to next().
    The specific policy of adaptivity is determined by implementations."""

    @property
    @abstractmethod
    def initial(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def next(self, population: PopulationT) -> T:
        raise NotImplementedError()


class ConstParameter(AdaptiveParameter[T]):
    """Stub implementation of AdaptiveParameter for constant parameters."""

    def __init__(self, value: T):
        self._value = value

    @property
    def initial(self) -> T:
        return self._value

    def next(self, population: PopulationT) -> T:
        return self._value
