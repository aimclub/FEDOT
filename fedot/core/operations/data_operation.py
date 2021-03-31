from fedot.core.log import Log
from fedot.core.repository.operation_types_repository import \
    OperationMetaInfo, OperationTypesRepository
from fedot.core.operations.operation import Operation


class DataOperation(Operation):
    """
    Class with fit/predict methods defining the evaluation strategy for the task

    :param operation_type: name of the data operation
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type, log)
        self.operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')

    @property
    def metadata(self) -> OperationMetaInfo:
        operation_info = self.operations_repo.operation_info_by_id(self.operation_type)
        if not operation_info:
            raise ValueError(f'Data operation {self.operation_type} not found')
        return operation_info
