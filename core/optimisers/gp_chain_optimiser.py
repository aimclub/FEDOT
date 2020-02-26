from typing import (
    List,
    Callable,
    Optional
)
from core.composer.gp_composer.gp_node import GP_Node
from core.models.model import Model
from core.composer.chain import Chain
from core.models.data import Data
from core.composer.gp_composer.gp_node import GP_NodeGenerator
from random import choice, randint


class GPChainOptimiser():
    def __init__(self, initial_chain, requirements, input_data: Data):
        self.requirements = requirements
        self.input_data = input_data
        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

    def run_evolution(self) -> Chain:
        return Chain()

    def _make_population(self, pop_size) -> List[GP_Node]:
        return [self._tree_generation() for _ in range(pop_size)]

    def _tree_generation(self) -> GP_Node:
        root = GP_NodeGenerator.get_secondary_node(choice(self.requirements.secondary_requirements))
        self._tree_growth(node_from=root)
        print("root depth", root)
        return root

    def _tree_growth(self, node_from):
        offspring_size = randint(2, self.requirements.max_arity)
        for offspring_node in range(offspring_size):
            if node_from.get_depth_up() >= self.requirements.max_depth or (
                    node_from.get_depth_up() < self.requirements.max_depth and self.requirements.max_depth and randint(
                0, 1)):

                new_node = GP_NodeGenerator.get_primary_node(choice(self.requirements.primary_requirements),
                                                             nodes_from=node_from, input_data=self.input_data)
                node_from.offspring_fill(new_node)
            else:
                new_node = GP_NodeGenerator.get_secondary_node(choice(self.requirements.secondary_requirements),
                                                               nodes_from=node_from)
                self._tree_growth(new_node)
                node_from.offspring_fill(new_node)
