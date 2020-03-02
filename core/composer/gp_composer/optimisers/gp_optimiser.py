from typing import (
    List,
    Callable
)
from core.composer.gp_composer.gp_node import GP_Node
from random import choice, randint
from core.composer.tree_drawing import Tree_Drawing
import numpy as np
from core.composer.gp_composer.optimisers.evo_operators import tournament_selection, standard_crossover, \
    standard_mutation
from copy import deepcopy


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

    def optimise(self, metric_function_for_nodes) -> GP_Node:
        for generation_num in range(self.requirements.num_of_generations):
            print("GP generation num:\n", generation_num)
            self.fitness = [round(metric_function_for_nodes(tree_root), 3) for tree_root in self.population]
            if not self.requirements.minimization:
                self.the_best_ind = self.population[np.argsort(self.fitness)[len(self.fitness) - 1]]
            else:
                self.the_best_ind = self.population[np.argsort(self.fitness)[0]]

            selected_indexes = tournament_selection(fitnesses=self.fitness,
                                                    minimization=self.requirements.minimization,
                                                    group_size=5)
            new_population = []
            for ind_num in range(self.requirements.pop_size - 1):
                new_population.append(standard_crossover(tree1=self.population[selected_indexes[ind_num][0]],
                                                         tree2=self.population[selected_indexes[ind_num][1]],
                                                         max_depth=self.requirements.max_depth, pair_num=ind_num,
                                                         pop_num=generation_num))

                new_population[ind_num] = standard_mutation(new_population[ind_num],
                                                            secondary_requirements=self.requirements.secondary_requirements,
                                                            primary_requirements=self.requirements.primary_requirements)

                self.population = deepcopy(new_population)
                self.population.append(self.the_best_ind)

        return self.the_best_ind

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
                    node_parent.get_depth_up() < self.requirements.max_depth and randint(
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
