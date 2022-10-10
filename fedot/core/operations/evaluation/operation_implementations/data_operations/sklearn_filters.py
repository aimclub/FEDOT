from copy import copy
from typing import Optional, Dict, Any

import numpy as np
import sklearn
from pkg_resources import parse_version
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    DataOperationImplementation
)
from fedot.core.operations.operation_parameters import OperationParameters


class FilterImplementation(DataOperationImplementation):
    """ Base class for applying filtering operations on tabular data """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = None
        self.operation = None

        self.log = default_log(self)

    def fit(self, input_data: InputData):
        """ Method for fit filter

        :param input_data: data with features, target and ids to process
        :return operation: trained operation (optional output)
        """

        self.operation.fit(input_data.features, input_data.target)

        return self.operation

    def transform(self, input_data: InputData) -> OutputData:
        """ Method for making prediction for predict stage

        :param input_data: data with features, target and ids to process
        :return output_data: filtered input data by rows
        """
        output_data = self._convert_to_output(input_data,
                                              input_data.features)
        return output_data

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """ Method for making prediction for fit stage

        :param input_data: data with features, target and ids to process
        :return output_data: filtered input data by rows
        """
        # For fit stage - filter data
        mask = self.operation.inlier_mask_
        if mask is not None:
            input_data = update_data(input_data, mask)
        else:
            self.log.info("Filtering Algorithm: didn't fit correctly. Return all objects")
        output_data = self._convert_to_output(input_data,
                                              input_data.features)
        return output_data


class RegRANSACImplementation(FilterImplementation):
    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.max_iter = 10

    def fit(self, input_data: InputData):
        iter_ = 0

        while iter_ < self.max_iter:
            try:
                self.operation.inlier_mask_ = None
                self.operation.fit(input_data.features, input_data.target)
                return self.operation
            except ValueError:
                self.log.info("RANSAC: multiplied residual_threshold on 2")
                residual_threshold = self.params.get('residual_threshold')
                self.params.update(residual_threshold=residual_threshold * 2)
                iter_ += 1

        return self.operation


class LinearRegRANSACImplementation(RegRANSACImplementation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with LinearRegression as core
    Task type - regression
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())

        # TODO valer1435 | Delete this after removing compatibility with sklearn<1.1
        if parse_version(sklearn.__version__) < parse_version('1.1.0'):
            self.operation = RANSACRegressor(base_estimator=self.inner_model, **self.params.to_dict())
        else:
            self.operation = RANSACRegressor(estimator=self.inner_model, **self.params.to_dict())


class NonLinearRegRANSACImplementation(RegRANSACImplementation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with DecisionTreeRegressor as core
    Task type - regression
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = DecisionTreeRegressor()

        # TODO valer1435 | Delete this after removing compatibility with sklearn<1.1
        if parse_version(sklearn.__version__) < parse_version('1.1.0'):
            self.operation = RANSACRegressor(base_estimator=self.inner_model, **self.params.to_dict())
        else:
            self.operation = RANSACRegressor(estimator=self.inner_model, **self.params.to_dict())


class IsolationForestRegImplementation(DataOperationImplementation):
    """
    Isolation Forest algorithm based on ExtraTreeRegressor
    Task type - regression
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.log = default_log(self)
        self.operation = IsolationForest(**self.params.to_dict())

    def fit(self, input_data: InputData) -> 'IsolationForest':
        """ Method for fit filter

        :param input_data: data with features, target and ids to process
        :return operation: trained operation (optional output)
        """

        self.operation.fit(input_data.features, input_data.target)

        return self.operation

    def _get_inlier_mask(self, input_data: InputData) -> np.ndarray:
        """ Method for making boolean mask of inliers classified as False

        :param input_data: data with features, target and ids to process
        :return mask: boolean mask of inliers classified as False
        """

        predictions = self.operation.predict(input_data.features)
        mask = predictions == 1
        return mask

    def transform(self, input_data: InputData) -> OutputData:
        """ Method for making prediction for predict stage

        :param input_data: data with features, target and ids to process
        :return output_data: filtered input data by rows
        """
        output_data = self._convert_to_output(input_data,
                                              input_data.features)
        return output_data

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """ Method for making prediction for fit stage

        :param input_data: data with features, target and ids to process
        :return output_data: filtered input data by rows
        """
        # For fit stage - filter data
        mask = self._get_inlier_mask(input_data)
        input_data = update_data(input_data, mask)

        output_data = self._convert_to_output(input_data,
                                              input_data.features)
        return output_data


class IsolationForestClassImplementation(IsolationForestRegImplementation):
    """
    Isolation Forest algorithm based on ExtraTreeRegressor
    Task type - classification
    """

    @staticmethod
    def _is_inlier_mask_correct(input_targets: np.ndarray, modified_targets: np.ndarray) -> bool:
        """ Method for checking if inlier mask is correct
        Inlier mask considered correct if after its application no class is completely or partially
        removed from the dataset by more than 80 percent

        :param input_targets: targets from input_data
        :param modified_targets: targets from modified_targets
        :return True if mask is correct otherwise False
        """

        if np.unique(input_targets).shape != np.unique(modified_targets).shape:
            return False
        for counts in zip(np.unique(input_targets, return_counts=True)[1],
                          np.unique(modified_targets, return_counts=True)[1]):
            if counts[0] > 5 * counts[1]:
                return False
        return True

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """ Method for making prediction for fit stage

        :param input_data: data with features, target and ids to process
        :return output_data: filtered input data by rows
        """
        # For fit stage - filter data
        mask = self._get_inlier_mask(input_data)
        modified_input_data = update_data(input_data, mask)
        if self._is_inlier_mask_correct(input_data.target, modified_input_data.target):
            input_data = modified_input_data

        output_data = self._convert_to_output(input_data,
                                              input_data.features)
        return output_data


def update_data(input_data: InputData, mask: np.ndarray) -> InputData:
    """ Method for updating target and features"""

    modified_input_data = copy(input_data)
    old_features = modified_input_data.features
    old_target = modified_input_data.target
    old_idx = modified_input_data.idx

    modified_input_data.features = old_features[mask]
    modified_input_data.target = old_target[mask]
    modified_input_data.idx = np.array(old_idx)[mask]

    return modified_input_data
