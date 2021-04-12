from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.operations.operation import Operation
from fedot.core.operations.model import Model
from fedot.core.operations.data_operation import DataOperation


class OperationFactory:
    """
    Base class for determining what type of operations should be defined
    in the node. Possible operations are models (ML models with fit and predict
    methods) and data operations (e.g. scaling) with fit and transform methods

    """

    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.operation_type = self._define_operation_type()

    def get_operation(self) -> Operation:
        """
        Factory method returns the desired object of the 'Data_operation' or
        'Model' class which depends on model_type variable

        """

        if self.operation_type == 'model':
            operation = Model(operation_type=self.operation_name)
        elif self.operation_type == 'data_operation':
            operation = DataOperation(operation_type=self.operation_name)
        else:
            raise ValueError(f'Operation type {self.operation_type} is not supported')

        return operation

    @property
    def operation_type_name(self):
        return self.operation_type

    def _define_operation_type(self) -> str:
        """
        The method determines what type of operations is set for this node

        :return : operations type 'model' or 'data_operation'
        """

        # Get available models from model_repository.json file
        operations_repo = OperationTypesRepository()
        models = operations_repo.operations

        # If there is a such model in the list
        if any(self.operation_name == model.id for model in models):
            operation_type = 'model'
        # Otherwise - it is preprocessing operations
        else:
            operation_type = 'data_operation'
        return operation_type
