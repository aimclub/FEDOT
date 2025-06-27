from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class DummyOperationImplementation(DataOperationImplementation):
    """ Dummy class for data forwarding """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)

    def fit(self, input_data: InputData):
        """
        Dummy operation doesn't support fit method
        """
        pass

    def transform(self, input_data: InputData) -> OutputData:
        """
        Method for forwarding input_data's features
        :param input_data: data with features, target and ids

        :return input_data: data with the same features
        """
        features = input_data.features
        output_data = self._convert_to_output(input_data, features)
        return output_data
