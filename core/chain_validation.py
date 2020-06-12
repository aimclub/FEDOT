from typing import Optional

import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isolate import isolates

from core.composer.chain import Chain, as_nx_graph
from core.composer.node import PrimaryNode, SecondaryNode
from core.repository.model_types_repository import ModelTypesRepository
from core.repository.tasks import Task

ERROR_PREFIX = 'Invalid chain configuration:'


def validate(chain: Chain, task: Optional[Task] = None):
    has_one_root(chain)
    has_no_cycle(chain)
    has_no_self_cycled_nodes(chain)
    has_no_isolated_nodes(chain)
    has_primary_nodes(chain)
    has_correct_models(chain, task)
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
    graph, _ = as_nx_graph(chain)
    ud_graph = nx.Graph()
    ud_graph.add_nodes_from(graph)
    ud_graph.add_edges_from(graph.edges)
    if not nx.is_connected(ud_graph):
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated components')
    return True


def has_correct_models(chain: Chain, task: Optional[Task] = None):
    # TODO pass task to this function

    models_repo = ModelTypesRepository()
    data_models_types, _ = ModelTypesRepository().models_with_tag(['data-models'])
    composition_data_models_types, _ = models_repo.models_with_tag(['composition'])

    is_root_not_datamodel = chain.root_node.model.model_type not in data_models_types or \
                            chain.root_node.model.model_type in composition_data_models_types

    is_primary_not_composition_datamodel = all([(node.model.model_type not in composition_data_models_types)
                                                for node in chain.nodes if isinstance(node, PrimaryNode)])

    is_primary_not_direct_datamodel = all([(node.model.model_type != 'direct_datamodel')
                                           for node in chain.nodes if isinstance(node, PrimaryNode)])
    if task:
        is_root_satisfy_task_type = task.task_type not in chain.root_node.model.acceptable_task_types
    else:
        is_root_satisfy_task_type = True

    if not (is_root_not_datamodel and
            is_root_satisfy_task_type and
            is_primary_not_composition_datamodel and
            is_primary_not_direct_datamodel):
        raise ValueError(f'{ERROR_PREFIX} Chain has incorrect models positions')

    return True
