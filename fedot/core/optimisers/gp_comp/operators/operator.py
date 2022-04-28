from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Sequence, Any

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.fitness import Fitness
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.pipelines.pipeline import Pipeline

T = TypeVar('T')
G = TypeVar('G', bound=Graph)

PopulationT = Sequence[Individual]  # TODO: provisional
ObjectiveFunction = Callable[[G], Fitness]


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
