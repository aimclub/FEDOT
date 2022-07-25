from copy import deepcopy
from random import choice, random
from typing import TYPE_CHECKING, Any, Callable, List, Union

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_operators import equivalent_subtree, replace_subtrees
from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.selection import Selection
from fedot.core.optimisers.graph import OptGraph
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


class Crossover:
    def __init__(self, crossover_types: List[Union[CrossoverTypesEnum, Callable]],
                 requirements: PipelineComposerRequirements, graph_generation_params: 'GraphGenerationParams',
                 max_number_of_attempts: int = 100):
        self.crossover_types = crossover_types
        self.graph_generation_params = graph_generation_params
        self.requirements = requirements
        self.max_number_of_attempts = max_number_of_attempts
        self.log = default_log(prefix='crossover')

    def __call__(self, population: PopulationT):
        if len(population) == 1:
            new_population = population
        else:
            new_population = []
            for ind_1, ind_2 in Selection.crossover_parents_selection(population):
                new_population += self._crossover(ind_1, ind_2)
        return new_population

    def update_requirements(self, new_requirements: PipelineComposerRequirements):
        self.requirements = new_requirements

    def _crossover(self, ind_first: Individual, ind_second: Individual) -> Any:
        crossover_type = choice(self.crossover_types)
        try:
            if self._will_crossover_be_applied(ind_first.graph, ind_second.graph, crossover_type):
                crossover_func = self._obtain_crossover_function(crossover_type)
                for _ in range(self.max_number_of_attempts):
                    new_graphs = self._adapt_and_apply_crossover(ind_first, ind_second, crossover_func)
                    are_correct = all(self.graph_generation_params.verifier(new_graph) for new_graph in new_graphs)
                    if are_correct:
                        parent_individuals = (ind_first, ind_second)
                        new_individuals = self._get_new_individuals_with_proper_parent_operators(new_graphs,
                                                                                                 parent_individuals,
                                                                                                 crossover_type)
                        return new_individuals

                self.log.debug('Number of crossover attempts exceeded. '
                               'Please check composer requirements for correctness.')
        except Exception as ex:
            self.log.error(f'Crossover ex: {ex}')

        return ind_first, ind_second

    def _obtain_crossover_function(self, crossover_type: Union[CrossoverTypesEnum, Callable]):
        if isinstance(crossover_type, Callable):
            return crossover_type
        else:
            return self._crossover_by_type(crossover_type)

    def _crossover_by_type(self, crossover_type: CrossoverTypesEnum):
        crossovers = {
            CrossoverTypesEnum.subtree: self._subtree_crossover,
            CrossoverTypesEnum.one_point: self._one_point_crossover,
        }
        if crossover_type in crossovers:
            return crossovers[crossover_type]
        else:
            raise ValueError(f'Required crossover type is not found: {crossover_type}')

    def _adapt_and_apply_crossover(self, first_individual, second_individual, crossover_function):
        is_custom_operator = isinstance(first_individual, OptGraph)
        first_object = deepcopy(first_individual.graph)
        second_object = deepcopy(second_individual.graph)

        if is_custom_operator:
            first_object = self.graph_generation_params.adapter.restore(first_object)
            second_object = self.graph_generation_params.adapter.restore(second_object)

        new_graphs = crossover_function(first_object, second_object)

        if is_custom_operator:
            for graph_id, graph in enumerate(new_graphs):
                new_graphs[graph_id] = self.graph_generation_params.adapter.adapt(graph)
        return new_graphs

    def _get_new_individuals_with_proper_parent_operators(self, new_graphs, parent_individuals, crossover_type):
        operator = ParentOperator(operator_type='crossover',
                                  operator_name=str(crossover_type),
                                  parent_individuals=parent_individuals)
        new_individuals = []
        for graph in new_graphs:
            parent_operators = []
            for parent_individual in parent_individuals:
                parent_operators.extend(parent_individual.parent_operators)
            parent_operators.append(operator)
            new_ind = Individual(graph, tuple(parent_operators))
            new_individuals.append(new_ind)
        return new_individuals

    def _will_crossover_be_applied(self, graph_first, graph_second, crossover_type) -> bool:
        return not (graph_first is graph_second or
                    random() > self.requirements.crossover_prob or
                    crossover_type is CrossoverTypesEnum.none)

    def _subtree_crossover(self, graph_first: Any, graph_second: Any) -> Any:
        """Performed by the replacement of random subtree
        in first selected parent to random subtree from the second parent"""
        random_layer_in_graph_first = choice(range(graph_first.depth))
        min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_second.depth > 1 else 0
        random_layer_in_graph_second = choice(range(min_second_layer, graph_second.depth))

        node_from_graph_first = choice(graph_first.operator.nodes_from_layer(random_layer_in_graph_first))
        node_from_graph_second = choice(graph_second.operator.nodes_from_layer(random_layer_in_graph_second))

        replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                         random_layer_in_graph_first, random_layer_in_graph_second, self.requirements.max_depth)

        return graph_first, graph_second

    def _one_point_crossover(self, graph_first: Any, graph_second: Any) -> Any:
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
                             layer_in_graph_first, layer_in_graph_second, self.requirements.max_depth)
        return graph_first, graph_second
