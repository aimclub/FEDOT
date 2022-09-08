from abc import ABC, abstractmethod
from typing import Sequence, Callable, Optional

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
# from fedot.core.optimisers.optimizer import GraphOptimizerParameters  # TODO: fix import loop

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

    def __init__(self,
                 parameters: Optional['GraphOptimizerParameters'] = None,
                 requirements: Optional[PipelineComposerRequirements] = None):
        self.requirements = requirements
        self.parameters = parameters

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
