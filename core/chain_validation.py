import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isolate import isolates

from core.composer.chain import Chain, as_nx_graph
from core.composer.node import PrimaryNode, SecondaryNode

ERROR_PREFIX = 'Invalid chain configuration:'


def self_validation(chain: Chain):
    has_one_root(chain)
    has_no_cycle(chain)
    has_no_self_cycled_nodes(chain)
    has_no_isolated_nodes(chain)
    has_primary_nodes(chain)
    return True


def has_one_root(chain: Chain):
    if chain.root_node:
        return True


def has_no_cycle(chain: Chain):
    graph, _ = as_nx_graph(chain)
    cycled = list(simple_cycles(graph))
    if len(cycled) > 0:
        raise ValueError(f'{ERROR_PREFIX} Chain has cycles')
    return True


def has_no_isolated_nodes(chain: Chain):
    graph, _ = as_nx_graph(chain)
    isolated = list(isolates(graph))
    if len(isolated) > 0:
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated nodes')
    return True


def has_primary_nodes(chain: Chain):
    if not any(node for node in chain.nodes if isinstance(node, PrimaryNode)):
        raise ValueError(f'{ERROR_PREFIX} Chain does not have primary nodes')
    return True


def has_no_self_cycled_nodes(chain: Chain):
    if any([node for node in chain.nodes if isinstance(node, SecondaryNode) and node in node.nodes_from]):
        raise ValueError(f'{ERROR_PREFIX} Chain has self-cycled nodes')
    return True


def has_no_isolated_components(chain: Chain):
    graph, _ = as_nx_graph(chain)
    ud_graph = nx.Graph()
    ud_graph.add_nodes_from(graph)
    ud_graph.add_edges_from(graph.edges)
    if not nx.is_connected(ud_graph):
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated components')
    return True
