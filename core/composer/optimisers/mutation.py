from random import random, choice, randint
from typing import (
    List,
    Any
)
from core.composer.tree_drawing import TreeDrawing

from copy import deepcopy


def standard_mutation(root_node: Any, secondary: Any, primary: Any,
                      secondary_node_func: Any = None, primary_node_func: Any = None, mutation_prob: bool = 0.8,
                      pair_num: int = None, pop_num: int = None, verbose: bool = True):  # temporary variables
    if not mutation_prob:
        probability = 1.0 / root_node.get_depth_to_primary()
    else:
        probability = mutation_prob

    if verbose:
        TreeDrawing.draw_branch(node=root_node, path="mutation", before_mutation=True, generation_num=pop_num,
                                ind_number=pair_num)

    result = random_mutation(root_node=root_node, probability=probability, primary_node_func=primary_node_func,
                             secondary_node_func=secondary_node_func, secondary=secondary,
                             primary=primary)

    if verbose:
        TreeDrawing.draw_branch(node=result, path="mutation", before_mutation=False, generation_num=pop_num,
                                ind_number=pair_num)

    return result


def random_mutation(root_node, probability, primary_node_func, secondary_node_func, secondary, primary):
    def _random_node_recursive(node, parent=None):

        if node.nodes_from:
            if random() < probability:
                rand_func = choice(secondary)
                node = secondary_node_func(rand_func,
                                           node_to=parent, nodes_from=node.nodes_from)
            for i, child in enumerate(node.nodes_from):
                if child.nodes_from:
                    node.nodes_from[i] = _random_node_recursive(child, parent=node)
                else:
                    if random() < probability:
                        node.nodes_from[i] = primary_node_func(choice(primary), node_to=node,
                                                               input_data=node.input_data)
        return node

    root_node_copy = deepcopy(root_node)
    _random_node_recursive(node=root_node_copy)
    return root_node_copy
