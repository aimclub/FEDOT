from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


def operations_for_task(task_name: str):
    """ Function filter operations by task and returns dictionary with names of
    models and data operations

    :param task_name: name of available task type

    :return dict_with_operations: dictionary with operations
        - models: appropriate models for task
        - data operations: appropriate data operations for task
    """

    task = get_task_by_name(task_name)

    # Get models and data operations
    models_repo = OperationTypesRepository()
    data_operations_repo = OperationTypesRepository(operation_type='data_operation')

    appropriate_models = models_repo.suitable_operation(task_type=task)
    appropriate_data_operations = data_operations_repo.suitable_operation(task_type=task)

    dict_with_operations = {'model': appropriate_models,
                            'data operation': appropriate_data_operations}

    return dict_with_operations


def get_task_by_name(task_name):
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
