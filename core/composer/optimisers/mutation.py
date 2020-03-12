from random import random, choice
from typing import Any

from copy import deepcopy
from core.composer.gp_composer.gp_node import GPNode


def standard_mutation(root_node: Any, secondary: Any, primary: Any,
                      secondary_node_func: Any = None, primary_node_func: Any = None, mutation_prob: bool = 0.8,
                      node_mutate_prob="strong"):
    if mutation_prob:
        if random() > mutation_prob:
            return deepcopy(root_node)

    if node_mutate_prob == "weak":
        probability = 1.0 / (5.0 * root_node.get_depth_to_primary())
    elif node_mutate_prob == "mean":
        probability = 1.0 / root_node.get_depth_to_primary()
    elif node_mutate_prob == "strong":
        probability = 5.0 / root_node.get_depth_to_primary()

    result = random_mutation(root_node=root_node, probability=probability, primary_node_func=primary_node_func,
                             secondary_node_func=secondary_node_func, secondary=secondary,
                             primary=primary)

    return result


def random_mutation(root_node, probability, primary_node_func, secondary_node_func, secondary, primary):
    def _random_node_recursive(node, parent=None):

        if node.nodes_from:
            if random() < probability:
                rand_func = choice(secondary)

                node = GPNode(chain_node=secondary_node_func(model=rand_func, nodes_from=node.nodes_from),
                              node_to=parent)
            else:
                node.node_to = parent
            for i, child in enumerate(node.nodes_from):
                if child.nodes_from:
                    node.nodes_from[i] = _random_node_recursive(child, parent=node)
                else:
                    if random() < probability:
                        node.nodes_from[i] = GPNode(chain_node=primary_node_func(model=choice(primary),
                                                                                 input_data=node.nodes_from[
                                                                                     i].input_data), node_to=node)
                    else:
                        node.nodes_from[i].node_to = node
        return node

    root_node_copy = deepcopy(root_node)
    root_node_copy = _random_node_recursive(node=root_node_copy)
    return root_node_copy
