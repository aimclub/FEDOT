from fedot.core.repository.operation_types_repository import ModelTypesRepository
from fedot.core.operations.operation import Operation, Model, DataOperation


class StrategyOperator:
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

        return operation

    @property
    def operation_type_name(self):
        return self.operation_type

    def _define_operation_type(self) -> str:
        """
        The method determines what type of operations is set for this node

        :return : operations type 'model' or 'data_operation'
        TODO need to add a flag for whether preprocessing is used in the node
         or not
        """

        # Get available models
        operations_repo = ModelTypesRepository()
        operations = operations_repo.operations

        # If there is a such model in the list
        if any(self.operation_name == model.id for model in operations):
            operation_type = 'model'
        # Overwise - it is preprocessing operations
        else:
            operation_type = 'data_operation'
        return operation_type
