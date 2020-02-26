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
from core.models.data import Data
from core.optimisers.gp_chain_optimiser import GPChainOptimiser
from core.composer.gp_composer.gp_node import GP_Secondary_Node,GP_Primary_Node
from core.composer.tree_drawing import Tree_Drawing

class GPComposer_requirements(ComposerRequirements):
    def __init__(self, primary_requirements: List[Model], secondary_requirements: List[Model],
                 max_depth: Optional[SupportsInt], max_arity: Optional[SupportsInt], pop_size:Optional[SupportsInt]):
        super().__init__(primary_requirements=primary_requirements, secondary_requirements=secondary_requirements, max_arity= max_arity, max_depth=max_depth)
        self.pop_size = pop_size

class GPComposer(Composer):
    def compose_chain(self, data:Optional[Data], initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposer_requirements],
                      metrics: Optional[Callable]) -> Chain:

        best_chain = GPChainOptimiser(initial_chain=initial_chain,
                                      requirements=composer_requirements,
                                      input_data=data).run_evolution()
        return best_chain















