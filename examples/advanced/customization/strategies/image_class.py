import warnings
from typing import Optional

from examples.advanced.customization.implementations.cnn_impls import MyCNNImplementation
from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.classification import FedotClassificationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class ImageClassificationStrategy(FedotClassificationStrategy):
    _operations_by_types = {
        'cnn_1': MyCNNImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation
