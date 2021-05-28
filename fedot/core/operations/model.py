from fedot.core.log import Log
from fedot.core.operations.operation import Operation
from fedot.core.repository.operation_types_repository import \
    OperationTypesRepository


class Model(Operation):
    """
    Class with fit/predict methods defining the evaluation strategy for the task

    :param operation_type: name of the model
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type=operation_type, log=log)
        self.operations_repo = OperationTypesRepository(repository_name='model_repository.json')
