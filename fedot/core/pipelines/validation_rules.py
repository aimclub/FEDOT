from typing import Optional

from fedot.core.operations.atomized_model import AtomizedModel
from fedot.core.operations.model import Model
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline, nodes_with_operation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum

ERROR_PREFIX = 'Invalid pipeline configuration:'


def has_correct_operation_positions(pipeline: 'Pipeline', task: Optional[Task] = None):
    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    is_root_satisfy_task_type = True
    if task:
        is_root_satisfy_task_type = task.task_type in pipeline.root_node.operation.acceptable_task_types

    if not is_root_satisfy_task_type:
        raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect operations positions')

    return True


def has_primary_nodes(pipeline: 'Pipeline'):
    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    if not any(node for node in pipeline.nodes if isinstance(node, PrimaryNode)):
        raise ValueError(f'{ERROR_PREFIX} Pipeline does not have primary nodes')
    return True


def has_final_operation_as_model(pipeline: 'Pipeline'):
    """ Check if the operation in root node is model or not """
    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    root_node = pipeline.root_node

    if type(root_node.operation) is not Model and type(root_node.operation) is not AtomizedModel:
        raise ValueError(f'{ERROR_PREFIX} Root operation is not a model')

    return True


def has_no_conflicts_with_data_flow(pipeline: 'Pipeline'):
    """ Check if the pipeline contains incorrect connections between nodes """
    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    operation_repo = OperationTypesRepository(operation_type='data_operation')
    forbidden_parents_combination, _ = operation_repo.suitable_operation()
    forbidden_parents_combination = set(forbidden_parents_combination)

    for node in pipeline.nodes:
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
                    raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect subgraph with identical data operations')
    return True


def has_correct_data_connections(pipeline: 'Pipeline'):
    """ Check if the pipeline contains incorrect connections between operation for different data types """
    _repo = OperationTypesRepository(operation_type='all')

    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)

    for node in pipeline.nodes:
        parent_nodes = node.nodes_from

        if parent_nodes is not None and len(parent_nodes) > 0:
            for parent_node in parent_nodes:
                if 'custom' in str(parent_node) or 'custom' in str(node):
                    return True

                current_nodes_supported_data_types = _repo.operation_info_by_id(node.operation.operation_type)
                parent_node_supported_data_types = _repo.operation_info_by_id(parent_node.operation.operation_type)

                if current_nodes_supported_data_types is None:
                    # case for atomic model
                    return True

                node_dtypes = set(current_nodes_supported_data_types.input_types)
                parent_dtypes = set(parent_node_supported_data_types.output_types) \
                    if parent_node_supported_data_types else node_dtypes
                if len(set.intersection(node_dtypes, parent_dtypes)) == 0:
                    raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect data connections')

    return True


def is_pipeline_contains_ts_operations(pipeline: 'Pipeline'):
    """ Function checks is the model contains operations for time series
    forecasting """
    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    # Get time series specific operations with tag "non_lagged"
    ts_operations = get_operations_for_task(task=Task(TaskTypesEnum.ts_forecasting),
                                            tags=["non_lagged"], mode='all')

    # List with operations in considering pipeline
    operations_in_pipeline = []
    for node in pipeline.nodes:
        operations_in_pipeline.append(node.operation.operation_type)

    if len(set(ts_operations) & set(operations_in_pipeline)) > 0:
        return True
    else:
        raise ValueError(f'{ERROR_PREFIX} pipeline not contains operations for time series processing')


def has_no_data_flow_conflicts_in_ts_pipeline(pipeline: 'Pipeline'):
    """ Function checks the correctness of connection between nodes """

    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)

    task = Task(TaskTypesEnum.ts_forecasting)
    models = get_operations_for_task(task=task, mode='model')
    # Preprocessing not only for time series
    non_ts_data_operations = get_operations_for_task(task=task,
                                                     mode='data_operation',
                                                     tags=["non_applicable_for_ts"])
    ts_data_operations = get_operations_for_task(task=task,
                                                 mode='data_operation',
                                                 tags=["non_lagged"])
    # Remove lagged and sparse lagged transformation
    ts_data_operations.remove('lagged')
    ts_data_operations.remove('sparse_lagged')
    ts_data_operations.remove('exog_ts')

    # Dictionary as {'current operation in the node': 'parent operations list'}
    # TODO refactor
    wrong_connections = {'lagged': models + non_ts_data_operations + ['lagged', 'sparse_lagged'],
                         'sparse_lagged': models + non_ts_data_operations + ['lagged', 'sparse_lagged'],
                         'ar': models + non_ts_data_operations + ['lagged', 'sparse_lagged'],
                         'arima': models + non_ts_data_operations + ['lagged', 'sparse_lagged'],
                         'ridge': ts_data_operations, 'linear': ts_data_operations,
                         'lasso': ts_data_operations, 'dtreg': ts_data_operations,
                         'knnreg': ts_data_operations, 'scaling': ts_data_operations,
                         'xgbreg': ts_data_operations, 'adareg': ts_data_operations,
                         'gbr': ts_data_operations, 'treg': ts_data_operations,
                         'rfr': ts_data_operations, 'svr': ts_data_operations,
                         'sgdr': ts_data_operations, 'normalization': ts_data_operations,
                         'kernel_pca': ts_data_operations, 'poly_features': ts_data_operations,
                         'ransac_lin_reg': ts_data_operations, 'ransac_non_lin_reg': ts_data_operations,
                         'rfe_lin_reg': ts_data_operations, 'rfe_non_lin_reg': ts_data_operations,
                         'pca': ts_data_operations}

    for node in pipeline.nodes:
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


def only_non_lagged_operations_are_primary(pipeline: 'Pipeline'):
    """ Only time series specific operations could be placed in primary nodes """
    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)

    # Check only primary nodes
    for node in pipeline.nodes:
        if type(node) == PrimaryNode and DataTypesEnum.ts not in node.operation.metadata.input_types:
            raise ValueError(
                f'{ERROR_PREFIX} Pipeline for forecasting has not non_lagged preprocessing in primary nodes')

    return True


def has_no_conflicts_in_decompose(pipeline: Pipeline):
    """ The function checks whether the 'class_decompose' or 'decompose'
    operation has two ancestors
    """

    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    for decomposer in ['decompose', 'class_decompose']:
        decompose_nodes = nodes_with_operation(pipeline,
                                               decomposer)
        if len(decompose_nodes) != 0:
            # Launch check decomposers
            __check_decomposer_has_two_parents(nodes_to_check=decompose_nodes)
            __check_decompose_parent_position(nodes_to_check=decompose_nodes)

    return True


def has_correct_data_sources(pipeline: Pipeline):
    """ Checks that data sources and other nodes are not mixed
    """

    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)
    is_data_source_in_names_conds = ['data_source' in str(n) for n in pipeline.nodes if isinstance(n, PrimaryNode)]

    if any(is_data_source_in_names_conds) and not all(is_data_source_in_names_conds):
        raise ValueError(f'{ERROR_PREFIX} Data sources are mixed with other primary nodes')
    return True


def has_parent_contain_single_resample(pipeline: Pipeline):
    """ 'Resample' should be single parent node for child operation.
    """

    if not isinstance(pipeline, Pipeline):
        pipeline = PipelineAdapter().restore(pipeline)

    for node in pipeline.nodes:
        if node.operation.operation_type == 'resample':
            children_nodes = pipeline.operator.node_children(node)
            for child_node in children_nodes:
                if len(child_node.nodes_from) > 1:
                    raise ValueError(f'{ERROR_PREFIX} Resample node is not single parent node for child operation')

    return True


def __check_connection(parent_operation, forbidden_parents):
    if parent_operation in forbidden_parents:
        raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')


def __check_decompose_parent_position(nodes_to_check: list):
    """ Function check if the data flow before decompose operation is correct
    or not

    :param nodes_to_check: list with decompose nodes in the pipeline
    """
    for decompose_node in nodes_to_check:
        parents = decompose_node.nodes_from
        model_parent = parents[0]

        if type(model_parent.operation) is not Model:
            raise ValueError(f'{ERROR_PREFIX} For decompose operation Model as first parent is required')


def __check_decomposer_has_two_parents(nodes_to_check: list):
    """ Function check if there are two parent nodes for decompose operation

    :param nodes_to_check: list with decompose nodes in the pipeline
    """

    for decompose_node in nodes_to_check:
        parents = decompose_node.nodes_from

        if parents is None:
            raise ValueError('Decompose operation has no parents')
        elif len(parents) != 2:
            raise ValueError(f'{ERROR_PREFIX} Two parents for decompose node were'
                             f' expected, but {len(parents)} were given')
