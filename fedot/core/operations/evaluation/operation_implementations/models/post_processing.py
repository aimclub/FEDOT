import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class CropToDataRangeImplementation(ModelImplementation):
    def __init__(self, params: OperationParameters):
        super().__init__(params)

    def fit(self, input_data: InputData):
        """ Class doesn't support fit operation

        Args:
            input_data: data with features, target and ids to process
        """

        pass

    def predict(self, input_data: InputData) -> OutputData:
        """ Method for cutting data by range for predict stage

        Args:
            input_data: data with features, target and ids to process

        Returns:
            output data with cut data
        """

        source_features = np.array(input_data.features)

        sigma_min = self.params.get('sigma_min')
        sigma_max = self.params.get('sigma_max')

        min_value = np.nanmin(np.array(input_data.target))
        min_value = min_value + min_value * sigma_min

        max_value = np.nanmax(np.array(input_data.target))
        max_value = max_value + max_value * sigma_max

        source_features[source_features > max_value] = max_value
        source_features[source_features < min_value] = min_value

        output_data = self._convert_to_output(input_data,
                                              source_features,
                                              data_type=input_data.data_type)

        return output_data
