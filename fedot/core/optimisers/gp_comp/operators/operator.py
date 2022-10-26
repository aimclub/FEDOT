from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.opt_history_objects.individual import Individual

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphOptimizerParameters

PopulationT = Sequence[Individual]
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

    def __init__(self,
                 parameters: Optional['GraphOptimizerParameters'] = None,
                 requirements: Optional[PipelineComposerRequirements] = None):
        self.requirements = requirements
        self.parameters = parameters
        self.log = default_log(self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def update_requirements(self,
                            parameters: Optional['GraphOptimizerParameters'] = None,
                            requirements: Optional[PipelineComposerRequirements] = None):
        if requirements:
            self.requirements = requirements
        if parameters:
            self.parameters = parameters
