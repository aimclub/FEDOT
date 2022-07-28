from abc import ABC, abstractmethod
from typing import Sequence, Callable, TYPE_CHECKING

from fedot.core.optimisers.gp_comp.individual import Individual

if TYPE_CHECKING:
    from fedot.core.composer.composer import ComposerRequirements

PopulationT = Sequence[Individual]  # TODO: provisional
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
    def update_requirements(self, new_requirements: 'ComposerRequirements'):
        pass
