from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence

from fedot.core.optimisers.gp_comp.individual import Individual

T = TypeVar('T')

PopulationT = Sequence[Individual]  # TODO: provisional


class Operator(ABC, Generic[T]):
    """Base abstract functional interface for genetic operators.
    Specific signatures are:
    - Evaluation: Population -> Population
    - Selection: Population -> Population
    - Inheritance: Population -> Population
    - Regularization: Population -> Population
    - Reproduction: Population -> Population
    - Mutation: Individual -> Individual
    - Crossover: (Individual, Individual) -> (Individual, Individual)
    """

    @abstractmethod
    def __call__(self, operand: T) -> T:
        pass
