from copy import deepcopy
from enum import Enum
from random import random, choice
from typing import Any

from core.composer.gp_composer.gp_node import GPNode


class mutation_power_enum(Enum):
    weak = 0
    mean = 1
    strong = 2


def get_mutation_prob(mut_id, root_node):
    if mut_id == 0:
        return 1.0 / (5.0 * root_node.get_depth_to_primary())
    elif mut_id == 1:
        return 1.0 / root_node.get_depth_to_primary()
    elif mut_id == 2:
        return 5.0 / root_node.get_depth_to_primary()


def standard_mutation(root_node: Any, secondary: Any, primary: Any,
                      secondary_node_func: Any = None, primary_node_func: Any = None, mutation_prob: bool = 0.8,
                      node_mutate_type=mutation_power_enum.mean):
    if mutation_prob:
        if random() > mutation_prob:
            return deepcopy(root_node)

    probability = get_mutation_prob(mut_id=node_mutate_type.value, root_node=root_node)

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
