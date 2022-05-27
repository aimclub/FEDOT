from typing import Callable, Optional, Sequence

import numpy as np

from fedot.core.composer.constraint import constraint_function
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import Operator, PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams


GraphSampler = Callable[[], OptGraph]


class InitialPopulationBuilder:
    """Generates initial population using two approaches.
    One is with initial graphs that are augmented and randomized with mutation operator.
    Another is just sampling and validating graphs from provided graph sampler."""

    _max_generation_attempts = 1000

    def __init__(self, graph_generation_params: GraphGenerationParams, log: Log):
        self.graph_gen_params = graph_generation_params
        self.mutation_operator: Callable[[Individual], Individual] = lambda ind: ind
        self.graph_sampler: Optional[GraphSampler] = None
        self.initial_graphs: Sequence[OptGraph] = ()
        self.log = log

    def with_mutation(self, mutation_operator: Operator[Individual]):
        """Enables mutation of sampled graphs with provided operator."""
        self.mutation_operator = mutation_operator
        return self

    def with_initial_graphs(self, initial_graphs: Sequence[OptGraph]):
        """Use initial graphs as a sampling population."""
        if initial_graphs:
            self.initial_graphs = initial_graphs
            self.graph_sampler = lambda: np.random.choice(self.initial_graphs)
        return self

    def with_custom_sampler(self, sampler: GraphSampler):
        """Use custom graph sampler for sampling graphs."""
        self.graph_sampler = sampler
        return self

    def build(self, pop_size: int) -> PopulationT:
        if self.graph_sampler is None:
            raise ValueError("Can not generate initial graphs, provide graph sampler!")

        population = []
        population.extend(Individual(graph) for graph in self.initial_graphs)
        n_iter = 0
        while len(population) < pop_size:
            new_ind = Individual(self.graph_sampler())
            new_ind = self.mutation_operator(new_ind)
            if constraint_function(new_ind.graph, self.graph_gen_params) and new_ind not in population:
                population.append(new_ind)
            n_iter += 1
            if n_iter >= self._max_generation_attempts:
                self.log.warn(f'Exceeded max number of attempts for generating initial graphs, stopping.'
                              f'Generated {len(population)} instead of {pop_size} graphs.')
                break
        return population
