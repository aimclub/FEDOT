from abc import ABC, abstractmethod
from typing import Sequence, Callable, TypeVar

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.composer_requirements import ComposerRequirements


G = TypeVar('G')

PopulationT = Sequence[Individual[G]]  # TODO: provisional
EvaluationOperator = Callable[[PopulationT], PopulationT]


class Operator(ABC):
    """ Base abstract functional interface for genetic operators.
    Specific signatures are:
    - Selection: Population -> Population
    - Inheritance: [Population, Population] -> Population
    - Regularization: [Population, EvaluationOperator] -> Population
    - Mutation: Union[Individual, Population] -> Union[Individual, Population]
    - Crossover: Population -> Population
    - Elitism: [Population, Population] -> Population
    """

    @abstractmethod
    def update_requirements(self, new_requirements: ComposerRequirements):
        pass
