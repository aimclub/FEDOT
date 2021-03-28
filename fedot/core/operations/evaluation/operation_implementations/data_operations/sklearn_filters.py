from copy import copy
from typing import Optional

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation


class FilterImplementation(DataOperationImplementation):
    """ Base class for applying filtering operations on tabular data """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = None
        self.operation = None

    def fit(self, input_data):
        """ Method for fit filter

        :param input_data: data with features, target and ids to process
        :return operation: trained operation (optional output)
        """

        self.operation.fit(input_data.features, input_data.target)

        return self.operation

    def transform(self, input_data, is_fit_chain_stage: bool):
        """ Method for making prediction

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: filtered input data by rows
        """

        features = input_data.features
        if is_fit_chain_stage:
            # For fit stage - filter data
            mask = self.operation.inlier_mask_
            inner_features = features[mask]

            # Update data
            modified_input_data = self._update_data(input_data, mask)

        else:
            # For predict stage there is a need to safe all the data
            inner_features = features
            modified_input_data = copy(input_data)

        # Convert it to OutputData
        output_data = self._convert_to_output(modified_input_data,
                                              inner_features)
        return output_data

    def get_params(self):
        return self.operation.get_params()

    def _update_data(self, input_data, mask):
        """ Method for updating target and features"""

        modified_input_data = copy(input_data)
        old_features = modified_input_data.features
        old_target = modified_input_data.target
        old_idx = modified_input_data.idx

        modified_input_data.features = old_features[mask]
        modified_input_data.target = old_target[mask]
        modified_input_data.idx = old_idx[mask]

        return modified_input_data


class LinearRegRANSACImplementation(FilterImplementation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with LinearRegression as core
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = LinearRegression(normalize=True)

        if not params:
            # Default parameters
            self.operation = RANSACRegressor(base_estimator=self.inner_model)
        else:
            self.operation = RANSACRegressor(base_estimator=self.inner_model,
                                             **params)
        self.params = params


class NonLinearRegRANSACImplementation(FilterImplementation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with DecisionTreeRegressor as core
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = DecisionTreeRegressor()

        if not params:
            # Default parameters
            self.operation = RANSACRegressor(base_estimator=self.inner_model)
        else:
            self.operation = RANSACRegressor(base_estimator=self.inner_model,
                                             **params)
        self.params = params
