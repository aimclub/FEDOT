from random import randint
import numpy as np
from core.composer.composer import Composer
from typing import (
    List,
    Callable,
    Optional
)

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import Model
from core.models.data import Data
from core.optimisers.gp_chain_optimiser import GPChainOptimiser


class GPComposer(Composer):
    def compose_chain(self, initial_chain: Optional[Chain],
                      primary_requirements: List[Model],
                      secondary_requirements: List[Model],
                      metrics: Optional[Callable], max_depth=5, max_arity=3) -> Chain:
        self.max_depth = max_depth
        self.max_arity = max_arity
        new_chain = Chain()
        empty_data = Data(np.zeros(1), np.zeros(1), np.zeros(1))
        root = NodeGenerator.get_primary_node(primary_requirements[randint(0,len(primary_requirements)-1)], empty_data)
        best_chain = GPChainOptimiser(initial_chain = None, primary_requirements=primary_requirements, secondary_requirements=secondary_requirements, metrics= metrics)
















