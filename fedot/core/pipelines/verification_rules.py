from typing import Optional

from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task, \
    atomized_model_type
from fedot.core.repository.tasks import Task, TaskTypesEnum

ERROR_PREFIX = 'Invalid pipeline configuration:'


def has_correct_operations_for_task(pipeline: Pipeline, task_type: Optional[TaskTypesEnum] = None):
    if task_type and not task_type in pipeline.root_node.operation.acceptable_task_types:
        raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect operations positions')
    return True


def has_primary_nodes(pipeline: Pipeline):
    if not any(node for node in pipeline.nodes if isinstance(node, PrimaryNode)):
        raise ValueError(f'{ERROR_PREFIX} Pipeline does not have primary nodes')
    return True


def has_final_operation_as_model(pipeline: Pipeline):
    """ Check if the operation in root node is model or not """
    root_node = pipeline.root_node
    # TODO @YamLyubov refactor check for AtomizedModel (fix circular import)
    if type(root_node.operation) is not Model and root_node.operation.operation_type != atomized_model_type():
        raise ValueError(f'{ERROR_PREFIX} Root operation is not a model')

    return True


def has_no_conflicts_with_data_flow(pipeline: Pipeline):
    """ Check if the pipeline contains incorrect connections between nodes """
    operation_repo = OperationTypesRepository(operation_type='data_operation')
    forbidden_parents_combination = operation_repo.suitable_operation()
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


def has_correct_data_connections(pipeline: Pipeline):
    """ Check if the pipeline contains incorrect connections between operation for different data types """
    _repo = OperationTypesRepository(operation_type='all')

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


def is_pipeline_contains_ts_operations(pipeline: Pipeline):
    """ Function checks is the model contains operations for time series
    forecasting """

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


def has_no_data_flow_conflicts_in_ts_pipeline(pipeline: Pipeline):
    """ Function checks the correctness of connection between nodes """

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
    wrong_connections = {'lagged': models + non_ts_data_operations + ['lagged', 'sparse_lagged', 'exog_ts'],
                         'sparse_lagged': models + non_ts_data_operations + ['lagged', 'sparse_lagged', 'exog_ts'],
                         'ar': models + non_ts_data_operations + ['lagged', 'sparse_lagged', 'exog_ts'],
                         'arima': models + non_ts_data_operations + ['lagged', 'sparse_lagged', 'exog_ts'],
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
                         'pca': ts_data_operations,
                         'gaussian_filter': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'diff_filter': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'smoothing': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'cut': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'ts_naive_average': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'locf': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'ets': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'polyfit': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'clstm': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'glm': ['lagged', 'sparse_lagged', 'exog_ts'],
                         'stl_arima': ['lagged', 'sparse_lagged', 'exog_ts'],
                         }

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


def only_non_lagged_operations_are_primary(pipeline: Pipeline):
    """ Only time series specific operations could be placed in primary nodes """

    # Check only primary nodes
    for node in pipeline.nodes:
        if type(node) == PrimaryNode and DataTypesEnum.ts not in node.operation.metadata.input_types:
            raise ValueError(
                f'{ERROR_PREFIX} Pipeline for forecasting has not non_lagged preprocessing in primary nodes')

    return True


def has_no_conflicts_in_decompose(pipeline: Pipeline):
    """ The function checks whether the 'class_decompose' or 'decompose' operation has two ancestors """

    for decomposer in ['decompose', 'class_decompose']:
        decompose_nodes = pipeline.get_nodes_by_name(decomposer)
        if len(decompose_nodes) != 0:
            # Launch check decomposers
            __check_decomposer_has_two_parents(nodes_to_check=decompose_nodes)
            __check_decompose_parent_position(nodes_to_check=decompose_nodes)

    return True


def has_correct_data_sources(pipeline: Pipeline):
    """ Checks that data sources and other nodes are not mixed """

    is_data_source_in_names_conds = ['data_source' in str(n) for n in pipeline.nodes if isinstance(n, PrimaryNode)]

    if any(is_data_source_in_names_conds) and not all(is_data_source_in_names_conds):
        raise ValueError(f'{ERROR_PREFIX} Data sources are mixed with other primary nodes')
    return True


def has_parent_contain_single_resample(pipeline: Pipeline):
    """ 'Resample' should be single parent node for child operation. """

    for node in pipeline.nodes:
        if node.operation.operation_type == 'resample':
            children_nodes = pipeline.node_children(node)
            for child_node in children_nodes:
                if len(child_node.nodes_from) > 1:
                    raise ValueError(f'{ERROR_PREFIX} Resample node is not single parent node for child operation')

    return True


def has_no_conflicts_during_multitask(pipeline: Pipeline):
    """
    Now if the classification task is solved, one part of the pipeline can solve
    the regression task if used after class_decompose. If class_decompose is followed
    by a classification operation, then this pipelining is incorrect.
    Validation perform only for classification pipelines.
    """

    classification_operations = get_operations_for_task(task=Task(TaskTypesEnum.classification), mode='all')
    pipeline_operations = [node.operation.operation_type for node in pipeline.nodes]
    pipeline_operations = set(pipeline_operations)

    number_of_unique_pipeline_operations = len(pipeline_operations)
    pipeline_operations_for_classification = set(classification_operations).intersection(pipeline_operations)

    if len(pipeline_operations_for_classification) == 0:
        return True

    if 'class_decompose' not in pipeline_operations:
        # There are no decompose operations in the pipeline
        if number_of_unique_pipeline_operations != len(pipeline_operations_for_classification):
            # There are operations in the pipeline that solve different tasks
            __check_multitask_operation_location(pipeline, classification_operations)

    return True


def has_no_conflicts_after_class_decompose(pipeline: Pipeline):
    """
    After the class_decompose operation, a regression model is required.
    Validation perform only for classification pipelines.
    """
    error_message = f'{ERROR_PREFIX} After classification decompose it is required to use regression model'
    pipeline_operations = [node.operation.operation_type for node in pipeline.nodes]
    if 'class_decompose' not in pipeline_operations:
        return True

    regression_operations = get_operations_for_task(task=Task(TaskTypesEnum.regression), mode='all')

    # Check for correct descendants after classification decompose
    for node in pipeline.nodes:
        parent_operations = [node.operation.operation_type for node in node.nodes_from]
        if 'class_decompose' in parent_operations:
            # Check is this model for regression task
            if node.operation.operation_type not in regression_operations:
                raise ValueError(error_message)

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


def __check_multitask_operation_location(pipeline: Pipeline, operations_for_classification: list):
    """
    Investigate paths for different tasks in the pipeline. If the pipeline solves
    several tasks simultaneously and there are no transitive operations in its
    structure (e.g. class_decompose), then the side branches must start from the
    primary node (nodes)
    """
    # TODO refactor to implement check via PipelineStructureExplorer
    primary_operations = []
    for node in pipeline.nodes:
        if isinstance(node, PrimaryNode):
            primary_operations.append(node.operation.operation_type)

    primary_operations = set(primary_operations)
    unique_primary_operations_number = len(primary_operations)

    primary_operations_for_classification = set(operations_for_classification).intersection(primary_operations)

    if unique_primary_operations_number != len(primary_operations_for_classification):
        # There are difference in tasks are in the primary nodes
        return True
    else:
        raise ValueError(f'{ERROR_PREFIX} Current pipeline can not solve multitask problem')
