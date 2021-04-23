import numpy as np
from typing import Optional

from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation


class SimpleDecomposeImplementation(DataOperationImplementation):
    """

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.pca = None
        self.params = None

    def fit(self, input_data):
        """
        The method trains the PCA model

        :param input_data: data with features, target and ids for PCA training
        :return pca: trained PCA model (optional output)
        """

        raise NotImplementedError()

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for transformation tabular data using PCA

        :param input_data: data with features, target and ids for PCA applying
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        raise NotImplementedError()

    def get_params(self):
        return None
