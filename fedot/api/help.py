from typing import Optional, List

from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


def print_models_info(task_name):
    """ Function display models and information about it for considered task

    :param task_name: name of available task type
    """

    task = _get_task_by_name(task_name)

    repository = OperationTypesRepository(operation_type='model')

    # Filter operations
    repository_operations_list = _filter_operations_by_type(repository, task)
    search_space = PipelineSearchSpace()
    for model in repository_operations_list:
        if model.id != 'custom':
            hyperparameters = search_space.get_operation_parameter_range(str(model.id))
            implementation_info = model.current_strategy(task)(model.id).implementation_info
            info_lst = [
                f"Model name - '{model.id}'",
                f"Available hyperparameters to optimize with tuner - {hyperparameters}",
                f"Strategy implementation - {model.current_strategy(task)}",
                f"Model implementation - {implementation_info}\n"
            ]
            print('\n'.join(info_lst))


def print_data_operations_info(task_name):
    """ Function display data operations and information about it for considered task

    :param task_name: name of available task type
    """

    task = _get_task_by_name(task_name)

    repository = OperationTypesRepository(operation_type='data_operation')
    repository_operations_list = _filter_operations_by_type(repository, task)
    search_space = PipelineSearchSpace()
    for operation in repository_operations_list:
        hyperparameters = search_space.get_operation_parameter_range(str(operation.id))
        implementation_info = operation.current_strategy(task)(operation.id).implementation_info
        info_lst = [
            f"Data operation name - '{operation.id}'",
            f"Available hyperparameters to optimize with tuner - {hyperparameters}",
            f"Strategy implementation - {operation.current_strategy(task)}",
            f"Operation implementation - {implementation_info}\n"
        ]
        print('\n'.join(info_lst))


def _filter_operations_by_type(repository, task):
    """ Function filter operations in repository by task

    :param repository: repository with operations to filter
    :param task: task type, if None, return all models

    :return repository_operations_list: list with filtered operations
    """

    if task is None:
        # Return all operations
        repository_operations_list = repository.operations
    else:
        repository_operations_list = []
        for operation in repository.operations:
            # Every operation can have several compatible task types
            task_types = operation.task_type
            if any(task == task_type for task_type in task_types):
                repository_operations_list.append(operation)

    return repository_operations_list


def operations_for_task(task_name: str):
    """ Function filter operations by task and returns dictionary with names of
    models and data operations

    :param task_name: name of available task type

    :return dict_with_operations: dictionary with operations
        - models: appropriate models for task
        - data operations: appropriate data operations for task
    """

    task = _get_task_by_name(task_name)

    # Get models and data operations
    models_repo = OperationTypesRepository()
    data_operations_repo = OperationTypesRepository(operation_type='data_operation')

    appropriate_models = models_repo.suitable_operation(task_type=task)
    appropriate_data_operations = data_operations_repo.suitable_operation(task_type=task)

    dict_with_operations = {'model': appropriate_models,
                            'data operation': appropriate_data_operations}

    return dict_with_operations


def _get_task_by_name(task_name):
    """ Function return task by its name

    :param task_name: name of available task type
    :return task: TaskTypesEnum.<appropriate task>
    """
    task_by_name = {'regression': TaskTypesEnum.regression,
                    'classification': TaskTypesEnum.classification,
                    'clustering': TaskTypesEnum.clustering,
                    'ts_forecasting': TaskTypesEnum.ts_forecasting}
    try:
        task = task_by_name[task_name]
    except KeyError:
        raise NotImplementedError(f'Task with name "{task_name}" not available, use one of '
                                  f'the following task names: {list(task_by_name.keys())}')

    return task
