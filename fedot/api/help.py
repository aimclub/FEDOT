from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.operations.tuning.hyperopt_tune.hp_hyperparams import params_by_operation


def print_models_info(task=None):
    """ Function display models and information about it for considered task

    :param task: task type, if None, return all models
    """

    repository = OperationTypesRepository()
    # Filter operations
    repository_operations_list = _filter_operations_by_type(repository, task)
    for model in repository_operations_list:
        hyperparameters = params_by_operation.get(str(model.id))
        implementation_info = model.current_strategy(task)(model.id).implementation_info
        print(f"Model name - '{model.id}'")
        print(f"Available hyperparameters to optimize with tuner - {hyperparameters}")
        print(f"Strategy implementation - {model.current_strategy(task)}")
        print(f"Model implementation - {implementation_info}\n")


def print_data_operations_info(task=None):
    """ Function display data operations and information about it for considered task

    :param task: task type, if None, return all models
    """

    repository = OperationTypesRepository(repository_name='data_operation_repository.json')
    # Filter operations
    repository_operations_list = _filter_operations_by_type(repository, task)
    for operation in repository_operations_list:
        hyperparameters = params_by_operation.get(str(operation.id))
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


if __name__ == '__main__':
    print('======================== Models ===========================')
    print_models_info()
    print('======================== Data operations ===========================')
    print_data_operations_info()
