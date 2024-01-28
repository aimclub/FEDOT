from fedot.core.operations.automl import AutoML
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.operations.operation import Operation
from fedot.core.repository.operation_types_repository import OperationReposEnum, OperationTypesRepository, \
    get_operation_type_from_id


class OperationFactory:
    """ Base class for determining what type of operations should be defined in the node """

    def __init__(self, operation_name: str):
        self.operation_name = get_operation_type_from_id(operation_name)
        operation = OperationTypesRepository(OperationReposEnum.ALL).operation_info_by_id(self.operation_name)
        if operation is None:
            raise ValueError(f"Unknown operation {self.operation_name}")
        self.operation_type = operation.operation_types_repository

    def get_operation(self) -> Operation:
        """ Factory method returns the desired object which depends on model_type variable """
        if self.operation_type in (OperationReposEnum.MODEL, OperationReposEnum.GPU):
            operation = Model(operation_type=self.operation_name)
        elif self.operation_type is OperationReposEnum.DATA_OPERATION:
            operation = DataOperation(operation_type=self.operation_name)
        elif self.operation_type is OperationReposEnum.AUTOML:
            operation = AutoML(operation_type=self.operation_name)
        else:
            raise ValueError(f'Operation type {self.operation_type} is not supported')

        return operation
