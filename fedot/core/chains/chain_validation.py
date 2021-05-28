from typing import Optional

import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isolate import isolates

from fedot.core.chains.chain import Chain, nodes_with_operation
from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.model import Model
from fedot.core.repository.operation_types_repository import \
    OperationTypesRepository, get_ts_operations
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
    has_no_conflicts_with_data_flow(chain)
    has_no_conflicts_in_decompose(chain)
    has_correct_data_connections(chain)

    # TSForecasting specific task validations
    if is_chain_contains_ts_operations(chain) is True:
        only_ts_specific_operations_are_primary(chain)
        has_no_data_flow_conflicts_in_ts_chain(chain)

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

    if type(root_node.operation) is not Model:
        raise ValueError(f'{ERROR_PREFIX} Root operation is not a model')

    return True


def has_no_conflicts_with_data_flow(chain: Chain):
    """ Check if the chain contains incorrect connections between nodes """
    operation_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
    forbidden_parents_combination, _ = operation_repo.suitable_operation()
    forbidden_parents_combination = set(forbidden_parents_combination)

    for node in chain.nodes:
        parent_nodes = node.nodes_from

        if parent_nodes is not None and len(parent_nodes) > 1:
            # There are several parents
            operation_names = []
            for parent in parent_nodes:
                operation_names.append(parent.operation.operation_type)

            # If operations are identical
            if len(set(operation_names)) == 1:
                # And if it is forbidden to combine them
                if operation_names[0] in forbidden_parents_combination:
                    raise ValueError(f'{ERROR_PREFIX} Chain has incorrect subgraph with identical data operations')
    return True


def has_correct_data_connections(chain: Chain):
    """ Check if the chain contains incorrect connections between operation for different data types """
    operation_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
    models_repo = OperationTypesRepository(repository_name='model_repository.json')

    for node in chain.nodes:
        parent_nodes = node.nodes_from

        if parent_nodes is not None and len(parent_nodes) > 0:
            for parent_node in parent_nodes:
                current_nodes_supported_data_types = \
                    get_supported_data_types(node, operation_repo, models_repo)
                parent_node_supported_data_types = \
                    get_supported_data_types(parent_node, operation_repo, models_repo)

                node_dtypes = set(current_nodes_supported_data_types.input_types)
                parent_dtypes = set(parent_node_supported_data_types.output_types)
                if len(set.intersection(node_dtypes, parent_dtypes)) == 0:
                    raise ValueError(f'{ERROR_PREFIX} Chain has incorrect data connections')

    return True


def get_supported_data_types(node, operation_repo, models_repo):
    supported_data_types = operation_repo.operation_info_by_id(node.operation.operation_type)
    if supported_data_types is None:
        supported_data_types = models_repo.operation_info_by_id(node.operation.operation_type)
    return supported_data_types


def is_chain_contains_ts_operations(chain: Chain):
    """ Function checks is the model contains operations for time series
    forecasting """
    # Get time series specific operations with tag "ts_specific"
    ts_operations = get_ts_operations(tags=["ts_specific"], mode='all')

    # List with operations in considering chain
    operations_in_chain = []
    for node in chain.nodes:
        operations_in_chain.append(node.operation.operation_type)

    if len(set(ts_operations) & set(operations_in_chain)) > 0:
        return True
    else:
        return False


def has_no_data_flow_conflicts_in_ts_chain(chain: Chain):
    """ Function checks the correctness of connection between nodes """
    models = get_ts_operations(mode='models')
    # Preprocessing not only for time series
    non_ts_data_operations = get_ts_operations(mode='data_operations',
                                               forbidden_tags=["ts_specific"])
    ts_data_operations = get_ts_operations(mode='data_operations',
                                           tags=["ts_specific"])
    # Remove lagged transformation
    ts_data_operations.remove('lagged')
    ts_data_operations.remove('exog_ts_data_source')

    # Dictionary as {'current operation in the node': 'parent operations list'}
    wrong_connections = {'lagged': models + non_ts_data_operations + ['lagged'],
                         'ar': models + non_ts_data_operations + ['lagged'],
                         'arima': models + non_ts_data_operations + ['lagged'],
                         'ridge': ts_data_operations, 'linear': ts_data_operations,
                         'lasso': ts_data_operations, 'dtreg': ts_data_operations,
                         'knnreg': ts_data_operations, 'scaling': ts_data_operations,
                         'xgbreg': ts_data_operations, 'adareg': ts_data_operations,
                         'gbr': ts_data_operations, 'treg': ts_data_operations,
                         'rfr': ts_data_operations, 'svr': ts_data_operations,
                         'sgdr': ts_data_operations, 'normalization': ts_data_operations,
                         'simple_imputation': ts_data_operations, 'pca': ts_data_operations,
                         'kernel_pca': ts_data_operations, 'poly_features': ts_data_operations,
                         'ransac_lin_reg': ts_data_operations, 'ransac_non_lin_reg': ts_data_operations,
                         'rfe_lin_reg': ts_data_operations, 'rfe_non_lin_reg': ts_data_operations}

    for node in chain.nodes:
        # Operation name in the current node
        current_operation = node.operation.operation_type
        parent_nodes = node.nodes_from

        if parent_nodes is not None:
            # There are several parents for current node or at least 1
            for parent in parent_nodes:
                parent_operation = parent.operation.operation_type

                forbidden_parents = wrong_connections.get(current_operation)
                if forbidden_parents is not None:
                    __check_connection(parent_operation, forbidden_parents)

    return True


def only_ts_specific_operations_are_primary(chain: Chain):
    """ Only time series specific operations could be placed in primary nodes """
    ts_data_operations = get_ts_operations(mode='data_operations',
                                           tags=["ts_specific"])

    # Check only primary nodes
    for node in chain.nodes:
        if type(node) == PrimaryNode:
            if node.operation.operation_type not in ts_data_operations:
                raise ValueError(
                    f'{ERROR_PREFIX} Chain for forecasting has not ts_specific preprocessing in primary nodes')

    return True


def has_no_conflicts_in_decompose(chain: Chain):
    """ The function checks whether the 'class_decompose' or 'decompose'
    operation has two ancestors
    """

    for decomposer in ['decompose', 'class_decompose']:
        decompose_nodes = nodes_with_operation(chain,
                                               decomposer)
        if len(decompose_nodes) != 0:
            # Launch check decomposers
            __check_decomposer_has_two_parents(nodes_to_check=decompose_nodes)
            __check_decompose_parent_position(nodes_to_check=decompose_nodes)

    return True


def __check_connection(parent_operation, forbidden_parents):
    if parent_operation in forbidden_parents:
        raise ValueError(f'{ERROR_PREFIX} Chain has incorrect subgraph with wrong parent nodes combination')


def __check_decompose_parent_position(nodes_to_check: list):
    """ Function check if the data flow before decompose operation is correct
    or not

    :param nodes_to_check: list with decompose nodes in the chain
    """
    for decompose_node in nodes_to_check:
        parents = decompose_node.nodes_from
        model_parent = parents[0]

        if type(model_parent.operation) is not Model:
            raise ValueError(f'{ERROR_PREFIX} For decompose operation Model as first parent is required')


def __check_decomposer_has_two_parents(nodes_to_check: list):
    """ Function check if there are two parent nodes for decompose operation

    :param nodes_to_check: list with decompose nodes in the chain
    """

    for decompose_node in nodes_to_check:
        parents = decompose_node.nodes_from

        if len(parents) != 2:
            raise ValueError(f'{ERROR_PREFIX} Two parents for decompose node were'
                             f' expected, but {len(parents)} were given')
