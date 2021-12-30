from copy import copy
from typing import Optional

import numpy as np
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    DataOperationImplementation
)
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class FilterImplementation(DataOperationImplementation):
    """ Base class for applying filtering operations on tabular data """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = None
        self.operation = None

        self.log = default_log(__name__)

    def fit(self, input_data):
        """ Method for fit filter

        :param input_data: data with features, target and ids to process
        :return operation: trained operation (optional output)
        """

        self.operation.fit(input_data.features, input_data.target)

        return self.operation

    def transform(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for making prediction

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: filtered input data by rows
        """

        features = input_data.features
        if is_fit_pipeline_stage:
            # For fit stage - filter data
            mask = self.operation.inlier_mask_
            if mask is not None:
                inner_features = features[mask]
                # Update data
                modified_input_data = self._update_data(input_data, mask)
            else:
                self.log.info("Filtering Algorithm: didn't fit correctly. Return all objects")
                inner_features = features
                modified_input_data = copy(input_data)

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
        modified_input_data.idx = np.array(old_idx)[mask]

        return modified_input_data


class RegRANSACImplementation(FilterImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.max_iter = 10
        self.parameters_changed = False

    def get_params(self):
        params_dict = self.params
        if self.parameters_changed is True:
            return tuple([params_dict, ['residual_threshold']])
        else:
            return params_dict

    def fit(self, input_data):
        iter_ = 0

        while iter_ < self.max_iter:
            try:
                self.operation.inlier_mask_ = None
                self.operation.fit(input_data.features, input_data.target)
                return self.operation
            except ValueError:
                self.log.info("RANSAC: multiplied residual_threshold on 2")
                self.params["residual_threshold"] *= 2
                self.parameters_changed = True
                iter_ += 1

        return self.operation


class LinearRegRANSACImplementation(RegRANSACImplementation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with LinearRegression as core
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())

        if not params:
            # Default parameters
            self.operation = RANSACRegressor(base_estimator=self.inner_model)
        else:
            self.operation = RANSACRegressor(base_estimator=self.inner_model,
                                             **params)
        self.params = params


class NonLinearRegRANSACImplementation(RegRANSACImplementation):
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
