from typing import Callable, Optional, Sequence

import numpy as np

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import Operator, PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.dag.graph_verifier import GraphVerifier

GraphSampler = Callable[[], OptGraph]
IndividualSampler = Callable[[], Individual]


class InitialPopulationBuilder:
    """Generates initial population using two approaches.
    One is with initial graphs (or individuals) that are augmented and randomized with mutation operator.
    Another is just sampling and validating graphs from provided graph sampler."""

    _max_generation_attempts = 1000

    def __init__(self, verifier: GraphVerifier):
        self.verifier = verifier
        self.mutation_operator: Callable[[Individual], Individual] = lambda ind: ind
        self.individual_sampler: Optional[IndividualSampler] = None
        self.initial_individuals: Sequence[Individual] = ()
        self.log = default_log(self.__class__.__name__)

    def with_mutation(self, mutation_operator: Operator[Individual]):
        """Enables mutation of sampled graphs with provided operator."""
        self.mutation_operator = mutation_operator
        return self

    def with_initial_individuals(self, initial_individuals: Sequence[Individual]):
        """Use initial individuals as a sampling population."""
        if initial_individuals:
            self.initial_individuals = initial_individuals
            self.individual_sampler = lambda: np.random.choice(self.initial_individuals)
        return self

    def with_custom_sampler(self, sampler: GraphSampler):
        """Use custom graph sampler for sampling individuals."""
        self.individual_sampler = lambda: Individual(sampler())
        return self

    def build(self, pop_size: int) -> PopulationT:
        if self.individual_sampler is None:
            raise ValueError("Can not generate initial population, provide graph sampler or initial individuals!")

        population = []
        population.extend(self.initial_individuals)
        n_iter = 0
        while len(population) < pop_size:
            sampled_ind = self.individual_sampler()
            new_ind = self.mutation_operator(sampled_ind)
            if new_ind not in population and self.verifier(new_ind.graph):
                population.append(new_ind)
            n_iter += 1
            if n_iter >= self._max_generation_attempts:
                self.log.warn(f'Exceeded max number of attempts for generating initial graphs, stopping.'
                              f'Generated {len(population)} instead of {pop_size} graphs.')
                break
        return population
