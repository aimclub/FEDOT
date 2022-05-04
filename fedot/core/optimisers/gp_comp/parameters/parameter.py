from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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
    def next(self, current: T) -> T:
        raise NotImplementedError()
