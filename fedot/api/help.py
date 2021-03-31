from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.chains.tuning.hyperparams import get_operation_parameter_range
from fedot.core.repository.tasks import TaskTypesEnum


def print_models_info(task_name):
    """ Function display models and information about it for considered task

    :param task_name: name of available task type
    """

    task = _get_task_by_name(task_name)

    repository = OperationTypesRepository(repository_name='model_repository.json')

    # Filter operations
    repository_operations_list = _filter_operations_by_type(repository, task)
    for model in repository_operations_list:
        hyperparameters = get_operation_parameter_range(str(model.id))
        implementation_info = model.current_strategy(task)(model.id).implementation_info
        print(f"Model name - '{model.id}'")
        print(f"Available hyperparameters to optimize with tuner - {hyperparameters}")
        print(f"Strategy implementation - {model.current_strategy(task)}")
        print(f"Model implementation - {implementation_info}\n")


def print_data_operations_info(task_name):
    """ Function display data operations and information about it for considered task

    :param task_name: name of available task type
    """

    task = _get_task_by_name(task_name)

    repository = OperationTypesRepository(repository_name='data_operation_repository.json')
    # Filter operations
    repository_operations_list = _filter_operations_by_type(repository, task)
    for operation in repository_operations_list:
        hyperparameters = get_operation_parameter_range(str(operation.id))
        implementation_info = operation.current_strategy(task)(operation.id).implementation_info
        print(f"Data operation name - '{operation.id}'")
        print(f"Available hyperparameters to optimize with tuner - {hyperparameters}")
        print(f"Strategy implementation - {operation.current_strategy(task)}")
        print(f"Operation implementation - {implementation_info}\n")


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
    data_operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')

    appropriate_models, _ = models_repo.suitable_operation(task_type=task)
    appropriate_data_operations, _ = data_operations_repo.suitable_operation(task_type=task)

    dict_with_operations = {'models': appropriate_models,
                            'data operations': appropriate_data_operations}

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
