from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Sequence, Union, Iterable, TYPE_CHECKING

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams
    from fedot.core.composer.composer import ComposerRequirements

GenerationFunction = Callable[[], Graph]
IndividualSampler = Callable[[], Individual]


class InitialGraphsGenerator(ABC):
    def __init__(self,
                 generation_params: 'GraphGenerationParams',
                 requirements: 'ComposerRequirements'):
        self.generation_params = generation_params
        self.requirements = requirements

    @abstractmethod
    def get_initial_graphs(self, **kwargs) -> Sequence[Union[Graph, OptGraph]]:
        """Method for generating initial graphs for GraphOptimizer
        """
        pass


class InitialPopulationGenerator(InitialGraphsGenerator):
    """Generates initial population using three approaches.
    One is with initial graphs.
    Another is with initial graphs generation function which generates a graph
    that will be added to initial population.
    The third way is random graphs generation according to GraphGenerationParameters and ComposerRequirements.
    The last approach is applied when neither initial graphs nor initial graphs generation function were provided."""

    _max_generation_attempts = 1000

    def __init__(self,
                 generation_params: 'GraphGenerationParams',
                 requirements: 'ComposerRequirements'):
        super(InitialPopulationGenerator, self).__init__(generation_params, requirements)
        self.generation_function: Optional[GenerationFunction] = None
        self.initial_graphs: Optional[Sequence[Graph]] = None
        self.log = default_log(self)

    def with_initial_graphs(self, initial_graphs: Union[Graph, OptGraph, Sequence[Graph], Sequence[OptGraph]]):
        """Use initial graphs as initial population."""
        if isinstance(initial_graphs, Graph):
            self.initial_graphs = [initial_graphs]
        elif isinstance(initial_graphs, Iterable):
            self.initial_graphs = list(initial_graphs)
        else:
            raise ValueError(f'Incorrect type of initial_assumption: '
                             f'Sequence[Graph] or Graph needed, but has {type(initial_graphs)}')
        return self

    def with_custom_generation_function(self, generation_func: GenerationFunction):
        """Use custom graph generation function to create initial population."""
        self.generation_function = generation_func
        return self

    def get_initial_graphs(self, pop_size: int, max_depth: int) -> Sequence[Union[Graph, OptGraph]]:
        verifier = self.generation_params.verifier
        if self.initial_graphs:
            if len(self.initial_graphs) > pop_size:
                self.initial_graphs = self.initial_graphs[:pop_size]
            return self.initial_graphs

        if not self.generation_function:
            self.generation_function = partial(random_graph, verifier, self.requirements, max_depth)

        population = []
        n_iter = 0
        while len(population) < pop_size:
            new_graph = self.generation_function()
            if new_graph not in population and verifier(new_graph):
                population.append(new_graph)
            n_iter += 1
            if n_iter >= self._max_generation_attempts:
                self.log.warning(f'Exceeded max number of attempts for generating initial graphs, stopping.'
                                 f'Generated {len(population)} instead of {pop_size} graphs.')
                break
        return population
