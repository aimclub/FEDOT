from random import randint, choice
from copy import deepcopy
import numpy as np
from core.composer.tree_drawing import Tree_Drawing
import random


def tournament_selection(fitnesses, minimization=True, group_size=5):
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


def standard_crossover(tree1, tree2, max_depth, crossover_prob, pair_num=None, pop_num=None):
    if tree1 is tree2 or random.random() > crossover_prob:
        return deepcopy(tree1)
    tree1_copy = deepcopy(tree1)
    rnlayer = randint(0, tree1_copy.get_depth_down() - 1)
    rnselflayer = randint(0, tree2.get_depth_down() - 1)
    if rnlayer == 0 and rnselflayer == 0:
        return deepcopy(tree2)

    changednode = choice(tree1_copy.get_nodes_from_layer(rnlayer))
    nodeforchange = choice(tree2.get_nodes_from_layer(rnselflayer))

    Tree_Drawing().draw_branch(node=tree1,
                               jpeg=f'p1_pair{pair_num}_pop{pop_num}_rnlayer{rnlayer}({changednode.eval_strategy.model.__class__.__name__}).png')
    Tree_Drawing().draw_branch(node=tree2,
                               jpeg=f'p2_pair{pair_num}_pop{pop_num}_rnselflayer{rnselflayer}({nodeforchange.eval_strategy.model.__class__.__name__}).png')

    if rnlayer == 0:
        return tree1_copy

    if changednode.get_depth_up() + nodeforchange.get_depth_down() - nodeforchange.get_depth_up() < max_depth + 1:
        print(changednode.get_depth_up())
        print(nodeforchange.get_depth_down())
        print(nodeforchange.get_depth_up())
        changednode.swap_nodes(nodeforchange)
        Tree_Drawing().draw_branch(node=tree1_copy, jpeg=f'result_pair{pair_num}_pop{pop_num}.png')
        return tree1_copy
    else:
        return tree1_copy


def standard_mutation(root_node, secondary_requirements, primary_requirements, probability=None):
    if not probability:
        probability = 1.0 / root_node.get_depth_down()

    Tree_Drawing().draw_branch(node=root_node, jpeg=f'tree(mut).png')

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
    Tree_Drawing().draw_branch(node=result, jpeg=f'tree after mut.png')

    return result
