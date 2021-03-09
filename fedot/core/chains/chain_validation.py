from typing import Optional

import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isolate import isolates

from fedot.core.chains.chain import Chain, chain_as_nx_graph
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.operation import Model
from fedot.core.repository.operation_types_repository import OperationTypesRepository
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
    has_at_least_one_model(chain)
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


def _is_data_merged(chain: Chain):
    root_node_merges_data = 'composition' in chain.root_node.operation_tags
    merging_is_required = any('decomposition' in node.operation_tags for node in chain.nodes)
    data_merged_or_merging_not_required = root_node_merges_data or not merging_is_required

    return data_merged_or_merging_not_required


def _is_root_not_datamodel(chain: Chain):
    return 'data_model' not in chain.root_node.operation_tags and \
           'decomposition' not in chain.root_node.operation_tags


def has_correct_operation_positions(chain: Chain, task: Optional[Task] = None):
    is_root_satisfy_task_type = True
    if task:
        is_root_satisfy_task_type = task.task_type in chain.root_node.operation.acceptable_task_types

    if not _is_root_not_datamodel(chain) or \
            not _is_data_merged(chain) or \
            not is_root_satisfy_task_type:
        raise ValueError(f'{ERROR_PREFIX} Chain has incorrect operations positions')

    return True


def has_at_least_one_model(chain: Chain):
    # Check is there at least one model in the chain
    models_amount = 0
    for node in chain.nodes:
        if type(node.operation) is Model:
            models_amount = models_amount + 1

    if models_amount == 0:
        raise ValueError(f'{ERROR_PREFIX} Chain consists only of data '
                         f'operations, at least one model required')
    return True


def has_final_operation_as_model(chain: Chain):
    root_node = chain.root_node
    root_operation = root_node.operation

    # Get available models
    operations_repo = OperationTypesRepository()
    models_ids = operations_repo.operations

    if any(str(root_operation) == model.id for model in models_ids):
        pass
    else:
        raise ValueError(f'{ERROR_PREFIX} Root operation is not a model')

    return True
