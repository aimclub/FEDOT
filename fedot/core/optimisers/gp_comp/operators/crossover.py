from copy import deepcopy
from random import choice, random
from typing import Callable, List, Union, Iterable, Tuple

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_operators import equivalent_subtree, replace_subtrees
from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


class Crossover(Operator):
    def __init__(self, crossover_types: List[Union[CrossoverTypesEnum, Callable]],
                 requirements: PipelineComposerRequirements, graph_generation_params: GraphGenerationParams,
                 max_number_of_attempts: int = 100):
        self.crossover_types = crossover_types
        self.graph_generation_params = graph_generation_params
        self.requirements = requirements
        self.max_number_of_attempts = max_number_of_attempts
        self.log = default_log(self)

    def __call__(self, population: PopulationT) -> PopulationT:
        if len(population) == 1:
            new_population = population
        else:
            new_population = []
            for ind_1, ind_2 in Crossover.crossover_parents_selection(population):
                new_population += self._crossover(ind_1, ind_2)
        return new_population

    def update_requirements(self, new_requirements: PipelineComposerRequirements):
        self.requirements = new_requirements

    @staticmethod
    def crossover_parents_selection(population: PopulationT) -> Iterable[Tuple[Individual, Individual]]:
        return zip(population[::2], population[1::2])

    def _crossover(self, ind_first: Individual, ind_second: Individual) -> Tuple[Individual, Individual]:
        crossover_type = choice(self.crossover_types)

        if self._will_crossover_be_applied(ind_first.graph, ind_second.graph, crossover_type):
            crossover_func = self._obtain_crossover_function(crossover_type)
            for _ in range(self.max_number_of_attempts):
                new_graphs = self._adapt_and_apply_crossover(ind_first, ind_second, crossover_func)
                are_correct = all(self.graph_generation_params.verifier(new_graph) for new_graph in new_graphs)
                if are_correct:
                    parent_individuals = (ind_first, ind_second)
                    new_individuals = self._get_individuals(new_graphs, parent_individuals, crossover_type)
                    return new_individuals

            self.log.debug('Number of crossover attempts exceeded. '
                           'Please check composer requirements for correctness.')

        return ind_first, ind_second

    def _obtain_crossover_function(self, crossover_type: Union[CrossoverTypesEnum, Callable]) -> Callable:
        if isinstance(crossover_type, Callable):
            return crossover_type
        else:
            return self._crossover_by_type(crossover_type)

    def _crossover_by_type(self, crossover_type: CrossoverTypesEnum) \
            -> Callable[[OptGraph, OptGraph, int], Tuple[OptGraph, OptGraph]]:
        crossovers = {
            CrossoverTypesEnum.subtree: subtree_crossover,
            CrossoverTypesEnum.one_point: one_point_crossover,
        }
        if crossover_type in crossovers:
            return crossovers[crossover_type]
        else:
            raise ValueError(f'Required crossover type is not found: {crossover_type}')

    def _adapt_and_apply_crossover(self, first_individual: Individual, second_individual: Individual,
                                   crossover_function: Callable) -> Tuple[OptGraph, OptGraph]:
        is_custom_operator = isinstance(first_individual, OptGraph)
        first_object = deepcopy(first_individual.graph)
        second_object = deepcopy(second_individual.graph)

        if is_custom_operator:
            first_object = self.graph_generation_params.adapter.restore(first_object)
            second_object = self.graph_generation_params.adapter.restore(second_object)

        new_graphs = crossover_function(first_object, second_object, self.requirements.max_depth)

        if is_custom_operator:
            for graph_id, graph in enumerate(new_graphs):
                new_graphs[graph_id] = self.graph_generation_params.adapter.adapt(graph)
        return new_graphs

    def _get_individuals(self, new_graphs: Tuple[OptGraph, OptGraph], parent_individuals: Tuple[Individual, Individual],
                         crossover_type: Union[CrossoverTypesEnum, Callable]) -> Tuple[Individual, Individual]:
        operator = ParentOperator(type='crossover',
                                  operators=(str(crossover_type),),
                                  parent_individuals=parent_individuals)
        return tuple(Individual(graph, operator) for graph in new_graphs)

    def _will_crossover_be_applied(self, graph_first, graph_second, crossover_type) -> bool:
        return not (graph_first is graph_second or
                    random() > self.requirements.crossover_prob or
                    crossover_type is CrossoverTypesEnum.none)


def subtree_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth: int) -> Tuple[OptGraph, OptGraph]:
    """Performed by the replacement of random subtree
    in first selected parent to random subtree from the second parent"""
    random_layer_in_graph_first = choice(range(graph_first.depth))
    min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_second.depth > 1 else 0
    random_layer_in_graph_second = choice(range(min_second_layer, graph_second.depth))

    node_from_graph_first = choice(graph_first.nodes_from_layer(random_layer_in_graph_first))
    node_from_graph_second = choice(graph_second.nodes_from_layer(random_layer_in_graph_second))

    replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                     random_layer_in_graph_first, random_layer_in_graph_second, max_depth)

    return graph_first, graph_second


def one_point_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth: int) -> Tuple[OptGraph, OptGraph]:
    """Finds common structural parts between two trees, and after that randomly
    chooses the location of nodes, subtrees of which will be swapped"""
    pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
    if pairs_of_nodes:
        node_from_graph_first, node_from_graph_second = choice(pairs_of_nodes)

        layer_in_graph_first = \
            graph_first.root_node.distance_to_primary_level - node_from_graph_first.distance_to_primary_level
        layer_in_graph_second = \
            graph_second.root_node.distance_to_primary_level - node_from_graph_second.distance_to_primary_level

        replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                         layer_in_graph_first, layer_in_graph_second, max_depth)
    return graph_first, graph_second
