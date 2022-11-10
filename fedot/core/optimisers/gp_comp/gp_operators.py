from copy import deepcopy
from random import randint
from typing import Any, List, Optional, Tuple

from fedot.core.constants import MAXIMAL_ATTEMPTS_NUMBER
from fedot.core.dag.graph_utils import distance_to_root_level, distance_to_primary_level
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.opt_node_factory import OptNodeFactory
from fedot.core.optimisers.optimizer import GraphGenerationParams


def random_graph(graph_generation_params: GraphGenerationParams,
                 requirements: PipelineComposerRequirements,
                 max_depth: Optional[int] = None) -> OptGraph:
    max_depth = max_depth if max_depth else requirements.max_depth
    is_correct_graph = False
    graph = None
    n_iter = 0
    requirements = adjust_requirements(requirements)
    node_factory = graph_generation_params.node_factory

    while not is_correct_graph:
        graph = OptGraph()
        if requirements.max_depth == 1:
            graph_root = node_factory.get_node(is_primary=True)
            graph.add_node(graph_root)
        else:
            graph_root = node_factory.get_node(is_primary=False)
            graph.add_node(graph_root)
            graph_growth(graph, graph_root, node_factory, requirements, max_depth)

        is_correct_graph = graph_generation_params.verifier(graph)
        n_iter += 1
        if n_iter > MAXIMAL_ATTEMPTS_NUMBER:
            raise ValueError(f'Could not generate random graph for {n_iter} '
                             f'iterations with requirements {requirements}')
    return graph


def adjust_requirements(requirements: PipelineComposerRequirements) -> PipelineComposerRequirements:
    """Function returns modified copy of the requirements if necessary.
    Example: Graph with only one primary node should consist of only one primary node
    without duplication, because this causes errors. Therefore minimum and maximum arity
    become equal to one.
    """
    requirements = deepcopy(requirements)
    if len(requirements.primary) == 1 and requirements.max_arity > 1:
        requirements.min_arity = requirements.max_arity = 1
    return requirements


def graph_growth(graph: OptGraph,
                 node_parent: OptNode,
                 node_factory: OptNodeFactory,
                 requirements: PipelineComposerRequirements,
                 max_depth: int):
    """Function create a graph and links between nodes"""
    offspring_size = randint(requirements.min_arity, requirements.max_arity)

    for offspring_node in range(offspring_size):
        height = distance_to_root_level(graph, node_parent)
        is_max_depth_exceeded = height >= max_depth - 2
        is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
        if is_max_depth_exceeded or is_primary_node_selected:
            primary_node = node_factory.get_node(is_primary=True)
            node_parent.nodes_from.append(primary_node)
            graph.add_node(primary_node)
        else:
            secondary_node = node_factory.get_node(is_primary=False)
            graph.add_node(secondary_node)
            node_parent.nodes_from.append(secondary_node)
            graph_growth(graph, secondary_node, node_factory, requirements, max_depth)


def equivalent_subtree(graph_first: Any, graph_second: Any) -> List[Tuple[Any, Any]]:
    """Finds the similar subtree in two given trees"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        node_first_childs = node_first.nodes_from
        node_second_childs = node_second.nodes_from
        if is_same_type and (not node_first.nodes_from or
                             node_first_childs and node_second_childs and
                             len(node_first_childs) == len(node_second_childs)):
            nodes.append((node_first, node_second))
            if node_first.nodes_from:
                for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                    nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                    if nodes_set:
                        nodes += nodes_set
        return nodes

    pairs_set = structural_equivalent_nodes(graph_first.root_node, graph_second.root_node)
    return pairs_set


def replace_subtrees(graph_first: Any, graph_second: Any, node_from_first: Any, node_from_second: Any,
                     layer_in_first: int, layer_in_second: int, max_depth: int):
    node_from_graph_first_copy = deepcopy(node_from_first)

    summary_depth = layer_in_first + distance_to_primary_level(node_from_second) + 1
    if summary_depth <= max_depth and summary_depth != 0:
        graph_first.update_subtree(node_from_first, node_from_second)

    summary_depth = layer_in_second + distance_to_primary_level(node_from_first) + 1
    if summary_depth <= max_depth and summary_depth != 0:
        graph_second.update_subtree(node_from_second, node_from_graph_first_copy)


def num_of_parents_in_crossover(num_of_final_inds: int) -> int:
    return num_of_final_inds if not num_of_final_inds % 2 else num_of_final_inds + 1


def filter_duplicates(archive, population) -> List[Any]:
    filtered_archive = []
    for ind in archive.items:
        has_duplicate_in_pop = False
        for pop_ind in population:
            if ind.fitness == pop_ind.fitness:
                has_duplicate_in_pop = True
                break
        if not has_duplicate_in_pop:
            filtered_archive.append(ind)
    return filtered_archive
