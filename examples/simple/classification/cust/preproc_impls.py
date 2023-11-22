from copy import deepcopy
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class GammaFiltImplementation(DataOperationImplementation):
    """ Class for application of :obj:`PolynomialFeatures` operation on data,
    where only not encoded features (were not converted from categorical using
    ``OneHot encoding``) are used

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        if not self.params:
            # Default parameters
            pass
        else:
            # Checking the appropriate params are using or not
            pass

    def fit(self, input_data: InputData):
        return None

    def transform(self, input_data: InputData) -> OutputData:
        # example of custom data pre-processing for predict state
        transformed_features = deepcopy(input_data.features)
        for i in range(transformed_features.shape[0]):
            transformed_features[i, :, :] = transformed_features[i, :, :] + np.random.normal(0, 30)

        output_data = self._convert_to_output(input_data,
                                              transformed_features, data_type=DataTypesEnum.image)

        return output_data