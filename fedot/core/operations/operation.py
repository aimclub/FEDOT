from abc import abstractmethod

from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.operation_types_repository import \
    OperationMetaInfo, OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, compatible_task_types

DEFAULT_PARAMS_STUB = 'default_params'


class Operation:
    """
    Base class for operators in nodes. Operators could be machine learning
    (or statistical) models or data operations

    """

    def __init__(self, operation_type: str, log: Log = None):
        self.operation_type = operation_type
        self.log = log

        self._eval_strategy = None
        self.params = DEFAULT_PARAMS_STUB

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @property
    def description(self):
        operation_type = self.operation_type
        operation_params = self.params
        return f'n_{operation_type}_{operation_params}'

    @abstractmethod
    def fit(self, data: InputData, is_fit_chain_stage: bool):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, fitted_operation, data: InputData,
                is_fit_chain_stage: bool, output_mode: str = 'default'):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.operation_type}'


def _eval_strategy_for_task(operation_type: str, current_task_type: TaskTypesEnum,
                            operations_repo):
    """
    The function returns the strategy for the selected operation and task type.
    And if it is necessary, found acceptable strategy for operation

    :param operation_type: name of operation, for example, 'ridge'
    :param current_task_type: task to solve
    :param operations_repo: repository with operations

    :return strategy: EvaluationStrategy class for this operation
    """

    # Get acceptable task types for operation
    operation_info = operations_repo.operation_info_by_id(operation_type)
    acceptable_task_types = operation_info.task_type

    # If the operation can't be used directly for the task type from data
    set_acceptable_types = set(acceptable_task_types)
    if current_task_type not in acceptable_task_types:

        # Search the supplementary task types, that can be included in chain
        # which solves main task
        globally_compatible_task_types = compatible_task_types(current_task_type)
        globally_set = set(globally_compatible_task_types)

        comp_types_acceptable_for_operation = list(set_acceptable_types.intersection(globally_set))
        if len(comp_types_acceptable_for_operation) == 0:
            raise ValueError(f'Operation {operation_type} can not be used as a part of {current_task_type}.')
        current_task_type = comp_types_acceptable_for_operation[0]

    strategy = operations_repo.operation_info_by_id(operation_type).current_strategy(current_task_type)
    return strategy
