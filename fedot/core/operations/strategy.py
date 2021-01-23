from fedot.core.repository.operation_types_repository import ModelTypesRepository
from fedot.core.operations.operation import Model, DataOperation


class StrategyOperator:
    """
    Base class for determining what type of operations should be defined
    in the node

    """

    def __init__(self, model_type):
        self.model_type = model_type
        #self.operation_type = self._define_operation_type()

    def get_operation(self, operation):
        """
        Factory method returns the desired object of the 'Preprocessing' or
        'Model' class which depends on model_type variable

        """

        if operation == 'model':
            operator = Model(model_type=self.model_type)
        else:
            operator = DataOperation(model_type=self.model_type)

        return operator

    def _define_operation_type(self) -> str:
        """
        The method determines what type of operations is set for this node

        :return : operations type 'model' or 'preprocessing'
        TODO need to add a flag for whether preprocessing is used in the node
         or not
        """

        # Get available models
        models_repo = ModelTypesRepository()
        models = models_repo.models

        # If there is a such model in the list
        if any(self.model_type == model.id for model in models):
            operation_type = 'model'
        # Overwise - it is preprocessing operations
        else:
            operation_type = 'preprocessing'
        return operation_type