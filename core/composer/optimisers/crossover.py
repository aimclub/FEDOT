from core.composer.tree_drawing import TreeDrawing
from copy import deepcopy
from core.composer.gp_composer.gp_node import swap_nodes
import random
from random import randint, choice
from typing import (
    List,
    Any
)


def standard_crossover (tree1: Any, tree2: Any, max_depth:int, crossover_prob: float = 0.8, verbose: bool = True, pair_num: int = None, pop_num: int = None):
    if tree1 is tree2 or random.random() > crossover_prob:
        return deepcopy(tree1)
    tree1_copy = deepcopy(tree1)
    random_layer_in_tree1 = randint(0, tree1_copy.get_depth_to_primary() - 1)
    random_layer_in_tree2 = randint(0, tree2.get_depth_to_primary() - 1)
    if random_layer_in_tree1 == 0 and random_layer_in_tree2 == 0:
        return deepcopy(tree2)

    node_from_tree1 = choice(tree1_copy.get_nodes_from_layer(random_layer_in_tree1))
    node_from_tree2 = choice(tree2.get_nodes_from_layer(random_layer_in_tree2))

    if verbose:
        TreeDrawing.draw_branch(node=tree1, path="crossover", ind_number=pair_num, generation_num=pop_num,ind_id ="p1",
                                tree_layer=random_layer_in_tree1,
                                model_name=node_from_tree1.eval_strategy.model.__class__.__name__)

        TreeDrawing.draw_branch(node=tree2, path="crossover", ind_number=pair_num, generation_num=pop_num, ind_id ="p2",
                                tree_layer=random_layer_in_tree2,
                                model_name=node_from_tree2.eval_strategy.model.__class__.__name__)

    if random_layer_in_tree1 == 0:
        return tree1_copy

    if node_from_tree1.get_depth_to_final() + node_from_tree2.get_depth_to_primary() <= max_depth:
        swap_nodes(node_from_tree1,node_from_tree2)
        if verbose:
            TreeDrawing.draw_branch(node=tree1_copy, path="crossover", ind_number=pair_num, generation_num=pop_num)

        return tree1_copy
    else:
        return tree1_copy
