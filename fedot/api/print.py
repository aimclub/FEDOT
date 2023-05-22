from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.api.help import get_task_by_name

def print_models_info(task_name):
    """ Function display models and information about it for considered task

    :param task_name: name of available task type
    """

    task = get_task_by_name(task_name)

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

    task = get_task_by_name(task_name)

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