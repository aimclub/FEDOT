from typing import Optional

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import Data
from core.composer.gp_composer.gp_node import GP_NodeGenerator
from random import randint, choice


class Chain:
    def __init__(self):
        self.nodes = []
        self.root = None

    def evaluate(self) -> Data:
        return self.root_node.apply()

    def add_node(self, new_node: Node):
        # Append new node to chain
        self.nodes.append(new_node)

    def update_node(self, new_node: Node):
        raise NotImplementedError()

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not any([(node in other_node.nodes_from)
                            for other_node in self.nodes if isinstance(other_node, SecondaryNode)])][0]
        return root

    @property
    def length(self):
        return len(self.nodes)

    @property
    def depth(self):
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryNode):
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        return _depth_recursive(self.root_node)

    def _flat_nodes_tree(self, requirements, input_data: Data):
        self.root = self.population = GP_NodeGenerator.get_secondary_node(
            choice(requirements.secondary_requirements))
        self.tree_generation(node_from=self.root, requirements=requirements, input_data=input_data)
        print("end")

    def tree_generation(self, node_from, requirements, input_data:Data):
        offspring_size = randint(2, requirements.max_arity)
        for offspring_node in range(offspring_size):
            print("self.get_depth_up(node_from)", self.get_depth_up(node_from))
            if self.get_depth_up(node_from) >= requirements.max_depth or (self.get_depth_up(
                    node_from) < requirements.max_depth and requirements.max_depth and randint(0, 1)):

                new_node = GP_NodeGenerator.get_primary_node(choice(requirements.primary_requirements),
                                                             nodes_from=node_from,input_data=input_data)
                node_from.offspring_fill(new_node)
            else:
                new_node = GP_NodeGenerator.get_secondary_node(choice(requirements.secondary_requirements),
                                                               nodes_from=node_from)
                self.tree_generation(new_node, requirements, input_data)
                node_from.offspring_fill(new_node)

    def get_depth_up(self, node):
        if node.nodes_from:
            depth = self.get_depth_up(node.nodes_from) + 1
            return depth
        else:
            return 0

    def get_depth_down(self):
        return
