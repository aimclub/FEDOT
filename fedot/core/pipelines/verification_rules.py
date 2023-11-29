from typing import Optional

from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task, \
    atomized_model_type
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import make_pipeline_generator

ERROR_PREFIX = 'Invalid pipeline configuration:'


def has_correct_operations_for_task(pipeline: Pipeline, task_type: Optional[TaskTypesEnum] = None):
    if task_type and task_type not in pipeline.root_node.operation.acceptable_task_types:
        raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect operations positions')
    return True


def has_primary_nodes(pipeline: Pipeline):
    if not any(node for node in pipeline.nodes if (isinstance(node, PipelineNode) and node.is_primary)):
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
    for node in make_pipeline_generator(pipeline):
        if not (node.is_primary or node.operation.operation_type == 'custom'):
            types = set(node.operation.metadata.input_types)
            for _node in node.nodes_from:
                if node.operation.operation_type != 'custom':
                    types &= set(_node.operation.metadata.output_types)
                if len(types) == 0:
                    raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')

    return True


def has_no_data_flow_conflicts_in_ts_pipeline(pipeline: Pipeline):
    """ Function checks the correctness of connection between nodes """
    task = Task(TaskTypesEnum.ts_forecasting)
    ts_models = get_operations_for_task(task=task, mode='model', tags=["non_lagged"])
    non_ts_models = sorted(list(set(get_operations_for_task(task=task, mode='model')).difference(set(ts_models))))

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

    ts_to_table_operations = ['lagged', 'sparse_lagged', 'exog_ts']

    # Dictionary as {'current operation in the node': 'parent operations list'}
    wrong_connections = get_wrong_links(ts_to_table_operations, ts_data_operations, non_ts_data_operations,
                                        ts_models, non_ts_models)

    limit_parents_count, need_to_have_parent = get_parent_limits(ts_to_table_operations, ts_data_operations,
                                                                 non_ts_data_operations,
                                                                 ts_models)

    for node in make_pipeline_generator(pipeline):
        raise_error = False
        if node.is_primary:
            # if node is primary then it should use time series
            raise_error = DataTypesEnum.ts not in node.operation.metadata.input_types
            raise_error |= node.operation.operation_type in need_to_have_parent
        else:
            # Operation name in the current node
            current_operation = node.operation.operation_type
            if current_operation in limit_parents_count:
                raise_error = limit_parents_count[current_operation] < len(node.nodes_from)

            # There are several parents for current node or at least 1
            if not raise_error:
                forbidden_parents = wrong_connections.get(current_operation)
                if forbidden_parents:
                    parents = set(parent.operation.operation_type for parent in node.nodes_from)
                    raise_error = set(forbidden_parents) & parents
        if raise_error:
            raise ValueError(f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')
    return True


def has_correct_location_of_resample(pipeline: Pipeline):
    """
    Pipeline can have only one resample operation located in start of the pipeline

    :param pipeline: pipeline for checking
    """
    is_resample_primary = False
    is_not_resample_primary = False
    for node in pipeline.nodes:
        if node.is_primary:
            if node.name == 'resample':
                is_resample_primary = True
            else:
                is_not_resample_primary = True
        else:
            if node.name == 'resample':
                raise ValueError(
                    f'{ERROR_PREFIX} Pipeline can have only one resample operation located in start of the pipeline')
    if is_resample_primary and is_not_resample_primary:
        raise ValueError(
            f'{ERROR_PREFIX} Pipeline can have only one resample operation located in start of the pipeline')
    return True


def get_wrong_links(ts_to_table_operations: list, ts_data_operations: list, non_ts_data_operations: list,
                    ts_models: list, non_ts_models: list) -> dict:
    """
    Function that return wrong ts connections like op_A : [op_B, op_C] that means op_B and op_C
    can't be a parent for op_A.

    :param ts_to_table_operations: list of ts_to_table operations
    :param ts_data_operations: list of ts data operations
    :param non_ts_data_operations: list of non ts data operations
    :param ts_models: list of ts models
    :param non_ts_models: list of non ts models
    :return: dict with wrong connections
    """
    limit_lagged_parents = {lagged_op: ts_models + non_ts_models + non_ts_data_operations + ts_to_table_operations
                            for lagged_op in ts_to_table_operations}

    limit_ts_models_parents = {ts_model: ts_models + non_ts_models + non_ts_data_operations + ts_to_table_operations
                               for ts_model in ts_models}
    limit_non_ts_models_parents = {non_ts_model: ts_data_operations
                                   for non_ts_model in non_ts_models}

    limit_ts_data_operations_parents = {
        ts_data_op: ts_models + non_ts_models + non_ts_data_operations + ts_to_table_operations
        for ts_data_op in ts_data_operations}
    limit_non_ts_data_operations_parents = {non_ts_data_op: ts_data_operations
                                            for non_ts_data_op in non_ts_data_operations}

    wrong_connections = {**limit_non_ts_data_operations_parents,
                         **limit_ts_data_operations_parents,
                         **limit_non_ts_models_parents,
                         **limit_ts_models_parents,
                         **limit_lagged_parents}
    return wrong_connections


def get_parent_limits(ts_to_table_operations: list, ts_data_operations: list, non_ts_data_operations: list,
                      ts_models: list) -> (dict, list):
    """
    Function that return some constraints on number of parents for time series forecasting graphs

    :param ts_to_table_operations: list of ts_to_table operations
    :param ts_data_operations: list of ts data operations
    :param non_ts_data_operations: list of non ts data operations
    :param ts_models: list of ts models
    :return: dict with parent limits and list with operations that must have a parent
    """
    limit_ts_model_data_op_parents_count = {ts_model_op: 1
                                            for ts_model_op in ts_models + ts_data_operations + ts_to_table_operations}

    limit_decompose_parents_count = {'decompose': 1}

    limit_parents_count = {**limit_ts_model_data_op_parents_count, **limit_decompose_parents_count}
    need_to_have_parent = [op for op in non_ts_data_operations]
    return limit_parents_count, need_to_have_parent


def only_non_lagged_operations_are_primary(pipeline: Pipeline):
    """ Only time series specific operations could be placed in primary nodes """

    # Check only primary nodes
    for node in pipeline.nodes:
        if isinstance(node, PipelineNode) and node.is_primary and \
                DataTypesEnum.ts not in node.operation.metadata.input_types:
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

    is_data_source_in_names_conds = ['data_source' in str(n) for n in pipeline.nodes if
                                     (isinstance(n, PipelineNode) and n.is_primary)]

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
        if isinstance(node, PipelineNode) and node.is_primary:
            primary_operations.append(node.operation.operation_type)

    primary_operations = set(primary_operations)
    unique_primary_operations_number = len(primary_operations)

    primary_operations_for_classification = set(operations_for_classification).intersection(primary_operations)

    if unique_primary_operations_number != len(primary_operations_for_classification):
        # There are difference in tasks are in the primary nodes
        return True
    else:
        raise ValueError(f'{ERROR_PREFIX} Current pipeline can not solve multitask problem')
