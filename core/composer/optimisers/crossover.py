import random
from random import randint, choice
from typing import (
    Any
)

from core.composer.gp_composer.gp_node import swap_nodes


def standard_crossover(tree1: Any, tree2: Any, max_depth: int, crossover_prob: float = 0.8) -> Any:
    if tree1 is tree2 or random.random() > crossover_prob:
        return tree1.duplicate
    tree1_copy = tree1.duplicate
    random_layer_in_tree1 = randint(0, tree1_copy.depth - 1)
    random_layer_in_tree2 = randint(0, tree2.depth - 1)
    if random_layer_in_tree1 == 0 and random_layer_in_tree2 == 0:
        return tree2.duplicate

    node_from_tree1 = choice(tree1_copy.nodes_from_height(random_layer_in_tree1))
    node_from_tree2 = choice(tree2.nodes_from_height(random_layer_in_tree2))

    if random_layer_in_tree1 == 0:
        return tree1_copy

    if random_layer_in_tree1 + node_from_tree2.depth <= max_depth:
        swap_nodes(node_from_tree1, node_from_tree2)

        return tree1_copy
    else:
        return tree1_copy
