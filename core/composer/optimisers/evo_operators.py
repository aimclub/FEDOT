import random
from copy import deepcopy
from random import randint, choice
from typing import (
    List,
    SupportsFloat,
    SupportsInt,
    Any
)

import numpy as np

from core.composer.tree_drawing import TreeDrawing


class EvolutuionaryOperators:

    @staticmethod
    def tournament_selection(fitnesses: List[SupportsFloat], minimization=True, group_size: SupportsInt = 5):

        if group_size > len(fitnesses):
            group_size = len(fitnesses)
        selected = []
        pair_num = 0
        for j in range(len(fitnesses) * 2):
            if not j % 2:
                selected.append([])
                if j > 1:
                    pair_num += 1

            tournir = [randint(0, len(fitnesses) - 1) for _ in range(group_size)]
            fitness_obj_from_tour = [fitnesses[tournir[i]] for i in range(group_size)]

            if minimization:
                selected[pair_num].append(tournir[np.argmin(fitness_obj_from_tour)])
            else:
                selected[pair_num].append(tournir[np.argmax(fitness_obj_from_tour)])

        return selected

    @staticmethod
    def standard_crossover(tree1: Any, tree2: Any, max_depth: SupportsInt, crossover_prob: SupportsFloat,
                           pair_num: SupportsInt = None, pop_num: SupportsInt = None, verbouse: bool = False):
        if tree1 is tree2 or random.random() > crossover_prob:
            return deepcopy(tree1)
        tree1_copy = deepcopy(tree1)
        random_layer_in_tree1 = randint(0, tree1_copy.get_depth_down() - 1)
        random_layer_in_tree2 = randint(0, tree2.get_depth_down() - 1)
        if random_layer_in_tree1 == 0 and random_layer_in_tree2 == 0:
            return deepcopy(tree2)

        node_from_tree1 = choice(tree1_copy.get_nodes_from_layer(random_layer_in_tree1))
        node_from_tree2 = choice(tree2.get_nodes_from_layer(random_layer_in_tree2))

        if verbouse:
            TreeDrawing().draw_branch(node=tree1,
                                      jpeg=f'crossover/p1_pair{pair_num}_pop{pop_num}_rnlayer{random_layer_in_tree1}({node_from_tree1.eval_strategy.model.__class__.__name__}).png')
            TreeDrawing().draw_branch(node=tree2,
                                      jpeg=f'crossover/p2_pair{pair_num}_pop{pop_num}_rnselflayer{random_layer_in_tree2}({node_from_tree2.eval_strategy.model.__class__.__name__}).png')

        if random_layer_in_tree1 == 0:
            return tree1_copy

        if node_from_tree1.get_depth_up() + node_from_tree2.get_depth_down() <= max_depth:
            node_from_tree1.swap_nodes(node_from_tree2)
            if verbouse:
                TreeDrawing().draw_branch(node=tree1_copy, jpeg=f'crossover/result_pair{pair_num}_pop{pop_num}.png')
            return tree1_copy
        else:
            return tree1_copy

    @staticmethod
    def standard_mutation(root_node: Any, secondary_requirements: List[Any], primary_requirements: List[Any],
                          probability: SupportsFloat = None, pair_num: SupportsInt = None,
                          pop_num: SupportsInt = None, verbouse: bool = False):
        if not probability:
            probability = 1.0 / root_node.get_depth_down()

        if verbouse:
            TreeDrawing().draw_branch(node=root_node, jpeg=f'mutation/tree(mut)_pop{pop_num}_ind{pair_num}.png')

        def _node_mutate(node):
            if node.nodes_from:
                if random.random() < probability:
                    node.eval_strategy.model = random.choice(secondary_requirements)
                for child in node.nodes_from:
                    _node_mutate(child)
            else:
                if random.random() < probability:
                    node.eval_strategy.model = random.choice(primary_requirements)

        result = deepcopy(root_node)
        _node_mutate(node=result)
        if verbouse:
            TreeDrawing().draw_branch(node=result, jpeg=f'mutation/tree after mut_pop{pop_num}_ind{pair_num}.png')

        return result
