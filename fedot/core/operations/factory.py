from fedot.core.operations.atomized import Atomized
from fedot.core.operations.automl import AutoML
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.operations.operation import Operation
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operation_type_from_id


class OperationFactory:
    """
    Base class for determining what type of operations should be defined
    in the node. Possible operations are models (ML models with fit and predict
    methods) and data operations (e.g. scaling) with fit and transform methods

    """

    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.operation_type = (OperationTypesRepository('all')
                               .operation_info_by_id(self.operation_name)
                               .repository_name)

    def get_operation(self) -> Operation:
        """
        Factory method returns the desired object of the 'Data_operation' or
        'Model' class which depends on model_type variable

        """

        if self.operation_type == 'model':
            operation = Model(operation_type=self.operation_name)
        elif self.operation_type == 'data_operation':
            operation = DataOperation(operation_type=self.operation_name)
        elif self.operation_type == 'automl':
            operation = AutoML(operation_type=self.operation_name)
        elif self.operation_type == 'atomized':
            operation = Atomized(operation_type=self.operation_name)
        else:
            raise ValueError(f'Operation type {self.operation_type} is not supported')

        return operation
