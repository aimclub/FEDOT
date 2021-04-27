import numpy as np

from copy import copy
from typing import Optional

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation


class DecomposerImplementation(DataOperationImplementation):
    """ Class for decomposing target """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = None

    def fit(self, input_data):
        """
        The decompose operation doesn't support fit method
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
            features = np.array(input_data.features)
            # Array with masks
            masked_features = np.array(input_data.masked_features)

            # Firts parent is "Data parent"
            prev_prediction_id = np.ravel(np.argwhere(masked_features == 0))
            prev_prediction = features[:, prev_prediction_id]

            # Calculate difference between prediction and target
            diff = input_data.target - input_data.features
            # Update target
            input_data.target = diff

    def get_params(self):
        return {}
