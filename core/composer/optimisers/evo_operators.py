import random
from copy import deepcopy
from random import randint, choice
from typing import (
    List,
    Any
)

import numpy as np

from core.composer.tree_drawing import TreeDrawing


class EvolutuionaryOperators:
    def __init__(self, requirements, primary_node_func, secondary_node_func):
        self.requirements = requirements
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func

    def tournament_selection(self, fitnesses: List[float], minimization=True, group_size: int = 5):

        group_size = min(group_size, len(fitnesses))

        selected = []
        pair_num = 0
        for j in range(len(fitnesses) * 2):
            if not j % 2:
                selected.append([])
                if j > 1:
                    pair_num += 1

            tournir = [randint(0, len(fitnesses) - 1) for _ in range(group_size)]
            fitness_obj_from_tour = [fitnesses[tournir[i]] for i in range(group_size)]

            choice_func = np.argmin if minimization else np.argmax
            selected[pair_num].append(tournir[choice_func(fitness_obj_from_tour)])

        return selected

    def standard_crossover(self, tree1: Any, tree2: Any, pair_num: int = None, pop_num: int = None):
        if tree1 is tree2 or random.random() > self.requirements.crossover_prob:
            return deepcopy(tree1)
        tree1_copy = deepcopy(tree1)
        random_layer_in_tree1 = randint(0, tree1_copy.get_depth_down() - 1)
        random_layer_in_tree2 = randint(0, tree2.get_depth_down() - 1)
        if random_layer_in_tree1 == 0 and random_layer_in_tree2 == 0:
            return deepcopy(tree2)

        node_from_tree1 = choice(tree1_copy.get_nodes_from_layer(random_layer_in_tree1))
        node_from_tree2 = choice(tree2.get_nodes_from_layer(random_layer_in_tree2))

        if self.requirements.verbose:
            TreeDrawing.draw_branch(node=tree1, path="crossover", ind_number=pair_num, generation_num=pop_num,
                                    tree_layer=random_layer_in_tree1,
                                    model_name=node_from_tree1.eval_strategy.model.__class__.__name__)

            TreeDrawing.draw_branch(node=tree1, path="crossover", ind_number=pair_num, generation_num=pop_num,
                                    tree_layer=random_layer_in_tree2,
                                    model_name=node_from_tree2.eval_strategy.model.__class__.__name__)

        if random_layer_in_tree1 == 0:
            return tree1_copy

        if node_from_tree1.get_depth_up() + node_from_tree2.get_depth_down() <= self.requirements.max_depth:
            node_from_tree1.swap_nodes(node_from_tree2)
            if self.requirements.verbose:
                TreeDrawing.draw_branch(node=tree1_copy, path="crossover", ind_number=pair_num, generation_num=pop_num)

            return tree1_copy
        else:
            return tree1_copy

    def standard_mutation(self, root_node: Any, pair_num: int = None, pop_num: int = None):

        if not self.requirements.mutation_prob:
            probability = 1.0 / root_node.get_depth_down()
        else:
            probability = self.requirements.mutation_prob

        if self.requirements.verbose:
            TreeDrawing.draw_branch(node=root_node, path="mutation", before_mutation=True, generation_num=pop_num,
                                    ind_number=pair_num)

        result = self.random_change_tree_nodes(root_node=root_node, probability=probability)

        if self.requirements.verbose:
            TreeDrawing.draw_branch(node=result, path="mutation", before_mutation=False, generation_num=pop_num,
                                    ind_number=pair_num)

        return result

    def random_change_tree_nodes(self, root_node, probability):

        def _random_node_recursive(node, parent=None):

            if node.nodes_from:
                if random.random() < probability:
                    node = self.__secondary_node_func(choice(self.requirements.secondary),
                                                      node_to=parent, nodes_from=node.nodes_from)
                for child in node.nodes_from:
                    _random_node_recursive(child, parent=node)
            else:
                if random.random() < probability:
                    node = self.__primary_node_func(choice(self.requirements.primary), node_to=parent,
                                                    input_data=node.input_data)
            if not node.node_to:
                return node

        root_node_copy = deepcopy(root_node)
        _random_node_recursive(node=root_node_copy)
        return root_node_copy
