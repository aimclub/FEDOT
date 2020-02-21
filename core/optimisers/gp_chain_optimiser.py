from typing import (
    List,
    Callable,
    Optional
)
from core.models.model import Model
from core.composer.chain import Chain


class GPChainOptimiser():
    def __init__(self, initial_chains, requirements, input_data):
        if not initial_chains:
            self.population = [Chain()._flat_nodes_tree(requirements,input_data) for i in range(requirements.pop_size)]
        else:
            self.population = initial_chains


    def evo_opt(self):
        return Chain()
