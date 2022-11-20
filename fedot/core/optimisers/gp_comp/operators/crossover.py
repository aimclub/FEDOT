from copy import deepcopy
from random import choice, random
from typing import Callable, Union, Iterable, Tuple, TYPE_CHECKING

from fedot.core.adapter import register_native
from fedot.core.dag.graph_utils import nodes_from_layer, node_depth
from fedot.core.optimisers.gp_comp.gp_operators import equivalent_subtree, replace_subtrees
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


CrossoverCallable = Callable[[OptGraph, OptGraph, int], Tuple[OptGraph, OptGraph]]


class Crossover(Operator):
    def __init__(self,
                 parameters: 'GPGraphOptimizerParameters',
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams):
        super().__init__(parameters, requirements)
        self.graph_generation_params = graph_generation_params

    def __call__(self, population: PopulationT) -> PopulationT:
        if len(population) == 1:
            new_population = population
        else:
            new_population = []
            for ind_1, ind_2 in Crossover.crossover_parents_selection(population):
                new_population += self._crossover(ind_1, ind_2)
        return new_population

    @staticmethod
    def crossover_parents_selection(population: PopulationT) -> Iterable[Tuple[Individual, Individual]]:
        return zip(population[::2], population[1::2])

    def _crossover(self, ind_first: Individual, ind_second: Individual) -> Tuple[Individual, Individual]:
        crossover_type = choice(self.parameters.crossover_types)

        if self._will_crossover_be_applied(ind_first.graph, ind_second.graph, crossover_type):
            crossover_func = self._get_crossover_function(crossover_type)
            for _ in range(self.parameters.max_num_of_operator_attempts):
                first_object = deepcopy(ind_first.graph)
                second_object = deepcopy(ind_second.graph)
                new_graphs = crossover_func(first_object, second_object, max_depth=self.requirements.max_depth)
                are_correct = all(self.graph_generation_params.verifier(new_graph) for new_graph in new_graphs)
                if are_correct:
                    parent_individuals = (ind_first, ind_second)
                    new_individuals = self._get_individuals(new_graphs, parent_individuals, crossover_type)
                    return new_individuals

            self.log.debug('Number of crossover attempts exceeded. '
                           'Please check composer requirements for correctness.')

        return ind_first, ind_second

    def _get_crossover_function(self, crossover_type: Union[CrossoverTypesEnum, Callable]) -> Callable:
        if isinstance(crossover_type, Callable):
            crossover_func = crossover_type
        else:
            crossover_func = self._crossover_by_type(crossover_type)
        return self.graph_generation_params.adapter.adapt_func(crossover_func)

    def _crossover_by_type(self, crossover_type: CrossoverTypesEnum) -> CrossoverCallable:
        crossovers = {
            CrossoverTypesEnum.subtree: subtree_crossover,
            CrossoverTypesEnum.one_point: one_point_crossover,
        }
        if crossover_type in crossovers:
            return crossovers[crossover_type]
        else:
            raise ValueError(f'Required crossover type is not found: {crossover_type}')

    def _get_individuals(self, new_graphs: Tuple[OptGraph, OptGraph], parent_individuals: Tuple[Individual, Individual],
                         crossover_type: Union[CrossoverTypesEnum, Callable]) -> Tuple[Individual, Individual]:
        operator = ParentOperator(type_='crossover',
                                  operators=str(crossover_type),
                                  parent_individuals=parent_individuals)
        return tuple(Individual(graph, operator) for graph in new_graphs)

    def _will_crossover_be_applied(self, graph_first, graph_second, crossover_type) -> bool:
        return not (graph_first is graph_second or
                    random() > self.parameters.crossover_prob or
                    crossover_type is CrossoverTypesEnum.none)


@register_native
def subtree_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth: int) -> Tuple[OptGraph, OptGraph]:
    """Performed by the replacement of random subtree
    in first selected parent to random subtree from the second parent"""
    random_layer_in_graph_first = choice(range(graph_first.depth))
    min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_second.depth > 1 else 0
    random_layer_in_graph_second = choice(range(min_second_layer, graph_second.depth))

    node_from_graph_first = choice(nodes_from_layer(graph_first, random_layer_in_graph_first))
    node_from_graph_second = choice(nodes_from_layer(graph_second, random_layer_in_graph_second))

    if is_crossover_correct(graph_first, graph_second):
        replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                         random_layer_in_graph_first, random_layer_in_graph_second, max_depth)

    return graph_first, graph_second


@register_native
def one_point_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth: int) -> Tuple[OptGraph, OptGraph]:
    """Finds common structural parts between two trees, and after that randomly
    chooses the location of nodes, subtrees of which will be swapped"""
    if is_crossover_correct(graph_first, graph_second):
        pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
        if pairs_of_nodes:
            node_from_graph_first, node_from_graph_second = choice(pairs_of_nodes)

            layer_in_graph_first = graph_first.depth - node_depth(node_from_graph_first)
            layer_in_graph_second = graph_second.depth - node_depth(node_from_graph_second)

            replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                             layer_in_graph_first, layer_in_graph_second, max_depth)
    return graph_first, graph_second


def is_crossover_correct(graph_first: OptGraph, graph_second: OptGraph):
    # crossover with custom models and exog_ts is unsafe
    for node in graph_first.nodes:
        operation_id = node.content['name']
        if 'exog_ts' in operation_id or 'custom' in operation_id:
            return False

    for node in graph_second.nodes:
        operation_id = node.content['name']
        if 'exog_ts' in operation_id or 'custom' in operation_id:
            return False
    return True
