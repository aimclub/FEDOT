from typing import TYPE_CHECKING

import networkx as nx
from networkx import isolates, simple_cycles

from fedot.core.pipelines.convert import graph_structure_as_nx_graph

if TYPE_CHECKING:
    from fedot.core.dag.graph import Graph

ERROR_PREFIX = 'Invalid graph configuration:'


def has_one_root(graph: 'Graph'):
    if graph.root_node:
        return True


def has_no_cycle(graph: 'Graph'):
    nx_graph, _ = graph_structure_as_nx_graph(graph)
    cycled = list(simple_cycles(nx_graph))
    if len(cycled) > 0:
        raise ValueError(f'{ERROR_PREFIX} Graph has cycles')

    return True


def has_no_isolated_nodes(graph: 'Graph'):
    nx_graph, _ = graph_structure_as_nx_graph(graph)
    isolated = list(isolates(nx_graph))
    if len(isolated) > 0 and graph.length != 1:
        raise ValueError(f'{ERROR_PREFIX} Graph has isolated nodes')
    return True


def has_no_self_cycled_nodes(graph: 'Graph'):
    if any([node for node in graph.nodes if node.nodes_from and node in node.nodes_from]):
        raise ValueError(f'{ERROR_PREFIX} Graph has self-cycled nodes')
    return True


def has_no_isolated_components(graph: 'Graph'):
    nx_graph, _ = graph_structure_as_nx_graph(graph)
    ud_nx_graph = nx.Graph()
    ud_nx_graph.add_nodes_from(nx_graph)
    ud_nx_graph.add_edges_from(nx_graph.edges)
    if not nx.is_connected(ud_nx_graph):
        raise ValueError(f'{ERROR_PREFIX} Graph has isolated components')
    return True


DEFAULT_DAG_RULES = [has_one_root, has_no_cycle, has_no_isolated_components,
                     has_no_self_cycled_nodes, has_no_isolated_nodes]
