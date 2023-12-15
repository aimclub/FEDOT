from functools import partial
from random import choice, randint, random, sample
from typing import TYPE_CHECKING, Optional

from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import distance_to_root_level, distance_to_primary_level, graph_has_cycle
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.operators.base_mutations import single_edge_mutation as golem_single_edge_mutation, \
    add_as_child, add_separate_parent_node, add_intermediate_node
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

# @register_native
def fedot_single_edge_mutation(graph: OptGraph,
                          requirements: GraphRequirements,
                          graph_gen_params: GraphGenerationParams,
                          parameters: 'GPAlgorithmParameters'
                          ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """

    def nodes_not_cycling(source_node: OptNode, target_node: OptNode):
        parents = source_node.nodes_from
        while parents:
            if target_node not in parents:
                grandparents = []
                for parent in parents:
                    grandparents.extend(parent.nodes_from)
                parents = grandparents
            else:
                return False
        return True

    for _ in range(parameters.max_num_of_operator_attempts):
        if len(graph.nodes) < 2 or graph.depth > requirements.max_depth:
            return graph

        source_node, target_node = sample(graph.nodes, 2)
        if source_node not in target_node.nodes_from:
            if graph_has_cycle(graph):
                graph.connect_nodes(source_node, target_node)
                break
            else:
                if nodes_not_cycling(source_node, target_node):
                    graph.connect_nodes(source_node, target_node)
                    break
    return graph


# @register_native
# def single_add_mutation(graph: OptGraph,
#                         requirements: GraphRequirements,
#                         graph_gen_params: GraphGenerationParams,
#                         parameters: AlgorithmParameters
#                         ) -> OptGraph:
#     """
#     Add new node between two sequential existing modes
#
#     :param graph: graph to mutate
#     """
#
#     if graph.depth >= requirements.max_depth:
#         # add mutation is not possible
#         return graph
#
#     node_to_mutate = choice(graph.nodes)
#
#     single_add_strategies = [add_as_child, add_separate_parent_node]
#     if node_to_mutate.nodes_from:
#         single_add_strategies.append(add_intermediate_node)
#     strategy = choice(single_add_strategies)
#
#     result = strategy(graph, node_to_mutate, graph_gen_params.node_factory)
#     return result
#
#
# @register_native
# def single_change_mutation(graph: OptGraph,
#                            requirements: GraphRequirements,
#                            graph_gen_params: GraphGenerationParams,
#                            parameters: AlgorithmParameters
#                            ) -> OptGraph:
#     """
#     Change node between two sequential existing modes.
#
#     :param graph: graph to mutate
#     """
#     node = choice(graph.nodes)
#     new_node = graph_gen_params.node_factory.exchange_node(node)
#     if not new_node:
#         return graph
#     graph.update_node(node, new_node)
#     return graph
#
#
# @register_native
# def single_drop_mutation(graph: OptGraph,
#                          requirements: GraphRequirements,
#                          graph_gen_params: GraphGenerationParams,
#                          parameters: AlgorithmParameters
#                          ) -> OptGraph:
#     """
#     Drop single node from graph.
#
#     :param graph: graph to mutate
#     """
#     if len(graph.nodes) < 2:
#         return graph
#     node_to_del = choice(graph.nodes)
#     node_name = node_to_del.name
#     removal_type = graph_gen_params.advisor.can_be_removed(node_to_del)
#     if removal_type == RemoveType.with_direct_children:
#         # TODO refactor workaround with data_source
#         graph.delete_node(node_to_del)
#         nodes_to_delete = \
#             [n for n in graph.nodes
#              if n.descriptive_id.count('data_source') == 1 and node_name in n.descriptive_id]
#         for child_node in nodes_to_delete:
#             graph.delete_node(child_node, reconnect=ReconnectType.all)
#     elif removal_type == RemoveType.with_parents:
#         graph.delete_subtree(node_to_del)
#     elif removal_type == RemoveType.node_rewire:
#         graph.delete_node(node_to_del, reconnect=ReconnectType.all)
#     elif removal_type == RemoveType.node_only:
#         graph.delete_node(node_to_del, reconnect=ReconnectType.none)
#     elif removal_type == RemoveType.forbidden:
#         pass
#     else:
#         raise ValueError("Unknown advice (RemoveType) returned by Advisor ")
#     return graph
