import numpy as np
from typing import Optional

from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation


class SimpleDecomposeImplementation(DataOperationImplementation):
    """ Implementation of the simple decomposer for regression task

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.pca = None
        self.params = None

    def fit(self, input_data):
        """
        The operation doesn't support fit method
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for modifying input_data

        :param input_data: data with features, target and ids
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        raise NotImplementedError()

    def get_params(self):
        return None


class TimeSeriesDecomposeImplementation(DataOperationImplementation):
    """ Operation decompose

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.pca = None
        self.params = None

    def fit(self, input_data):
        """
        The operation doesn't support fit method
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for modifying input_data

        :param input_data: data with features, target and ids
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        if is_fit_chain_stage:
            # Calculate difference between prediction and target
            diff = input_data.target - input_data.features
            # Update target
            input_data.target = diff
        else:
            # For predict stage don't perform any operations
            pass

        # Create OutputData
        output_data = self._convert_to_output(input_data, input_data.prev_features)
        # We decompose the target, so in the future we need to combine it again
        output_data.combine_target = 'sum'
        return output_data

    def get_params(self):
        return {}
