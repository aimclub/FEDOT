from random import randint
import numpy as np
from core.composer.composer import Composer, ComposerRequirements
from typing import (
    List,
    Callable,
    Optional,
    SupportsInt
)

from core.composer.chain import Chain

from core.models.model import Model
from core.models.data import InputData
from core.optimisers.gp_chain_optimiser import GPChainOptimiser
from core.composer.gp_composer.gp_node import GP_NodeGenerator


class GPComposer_requirements(ComposerRequirements):
    def __init__(self, primary_requirements: List[Model], secondary_requirements: List[Model],
                 max_depth: Optional[SupportsInt], max_arity: Optional[SupportsInt], pop_size: Optional[SupportsInt], num_of_generations: SupportsInt):
        super().__init__(primary_requirements=primary_requirements, secondary_requirements=secondary_requirements,
                         max_arity=max_arity, max_depth=max_depth)
        self.pop_size = pop_size
        self.num_of_generations = num_of_generations


class GPComposer(Composer):
    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposer_requirements],
                      metrics: Optional[Callable]) -> Chain:
        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements, primary_node_func=GP_NodeGenerator.primary_node, secondary_node_func=GP_NodeGenerator.secondary_node)
        best_chain = optimiser.optimise()
        return best_chain
