from typing import (
    List,
    Callable,
    Optional
)
from core.composer.gp_composer.gp_node import GP_Node
from core.models.model import Model
from core.composer.chain import Chain
from core.models.data import Data
from random import choice, randint
from core.composer.tree_drawing import Tree_Drawing


class GPChainOptimiser():
    def __init__(self, initial_chain, requirements, primary_node_func: Callable, secondary_node_func: Callable):
        self.requirements = requirements
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func
        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

        Tree_Drawing().draw_branch(node=self.population[1], jpeg="tree.png")
        print("J")

    def optimise(self, metric_function_for_nodes) -> List[GP_Node]:
        #fitness = [round(metric_function_for_nodes(tree_root), 3) for tree_root in self.population]
        fitness = round(metric_function_for_nodes(self.population[1]), 3)
        print("fitness", fitness)
        return self.population[0]

    def _make_population(self, pop_size) -> List[GP_Node]:
        return [self._random_tree() for _ in range(pop_size)]

    def _random_tree(self) -> GP_Node:
        root = self.__secondary_node_func(choice(self.requirements.secondary_requirements))
        self._tree_growth(node_parent=root)
        return root

    def _tree_growth(self, node_parent):
        offspring_size = randint(2, self.requirements.max_arity)
        node_offspring = []
        for offspring_node in range(offspring_size):
            if node_parent.get_depth_up() >= self.requirements.max_depth or (
                    node_parent.get_depth_up() < self.requirements.max_depth and self.requirements.max_depth and randint(
                0, 1)):

                new_node = self.__primary_node_func(choice(self.requirements.primary_requirements),
                                                    nodes_to=node_parent, input_data=None)
                node_offspring.append(new_node)
            else:
                new_node = self.__secondary_node_func(choice(self.requirements.secondary_requirements),
                                                      nodes_to=node_parent)
                self._tree_growth(new_node)
                node_offspring.append(new_node)
        node_parent.nodes_from = node_offspring
