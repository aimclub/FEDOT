from typing import Optional

import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isolate import isolates

from fedot.core.chains.chain import Chain, chain_as_nx_graph
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.model import Model
from fedot.core.repository.tasks import Task

ERROR_PREFIX = 'Invalid chain configuration:'


def validate(chain: Chain, task: Optional[Task] = None):
    # TODO pass task to this function
    has_one_root(chain)
    has_no_cycle(chain)
    has_no_self_cycled_nodes(chain)
    has_no_isolated_nodes(chain)
    has_primary_nodes(chain)
    has_correct_operation_positions(chain, task)
    has_final_operation_as_model(chain)
    return True


def has_one_root(chain: Chain):
    if chain.root_node:
        return True


def has_no_cycle(chain: Chain):
    graph, _ = chain_as_nx_graph(chain)
    cycled = list(simple_cycles(graph))
    if len(cycled) > 0:
        raise ValueError(f'{ERROR_PREFIX} Chain has cycles')

    return True


def has_no_isolated_nodes(chain: Chain):
    graph, _ = chain_as_nx_graph(chain)
    isolated = list(isolates(graph))
    if len(isolated) > 0 and chain.length != 1:
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
    graph, _ = chain_as_nx_graph(chain)
    ud_graph = nx.Graph()
    ud_graph.add_nodes_from(graph)
    ud_graph.add_edges_from(graph.edges)
    if not nx.is_connected(ud_graph):
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated components')
    return True


def has_correct_operation_positions(chain: Chain, task: Optional[Task] = None):
    is_root_satisfy_task_type = True
    if task:
        is_root_satisfy_task_type = task.task_type in chain.root_node.operation.acceptable_task_types

    if not is_root_satisfy_task_type:
        raise ValueError(f'{ERROR_PREFIX} Chain has incorrect operations positions')

    return True


def has_final_operation_as_model(chain: Chain):
    """ Check if the operation in root node is model or not """
    root_node = chain.root_node

    if type(root_node.operation) is Model:
        pass
    else:
        raise ValueError(f'{ERROR_PREFIX} Root operation is not a model')

    return True
