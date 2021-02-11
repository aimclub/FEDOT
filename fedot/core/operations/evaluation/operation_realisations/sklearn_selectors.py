from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import RFE

from fedot.core.operations.evaluation.\
    operation_realisations.abs_interfaces import EncodedInvariantOperation


class FeatureSelection(EncodedInvariantOperation):
    """ Class for applying feature selection operations on tabular data """

    def __init__(self):
        super().__init__()
        self.inner_model = None
        self.operation = None
        self.ids_to_process = None
        self.bool_ids = None

    def fit(self, input_data):
        """ Method for fit feature selection

        :param input_data: data with features, target and ids to process
        :return operation: trained operation (optional output)
        """
        features = input_data.features
        target = input_data.target

        bool_ids, ids_to_process = self._reasonability_check(features)
        self.ids_to_process = ids_to_process
        self.bool_ids = bool_ids

        if len(ids_to_process) > 0:
            features_to_process = np.array(features[:, ids_to_process])
            self.operation.fit(features_to_process, target)
        else:
            pass

        return self.operation

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """ Method for making prediction

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: filtered input data by columns
        """
        features = input_data.features
        if len(self.ids_to_process) > 0:
            transformed_features = self._make_new_table(features)
        else:
            transformed_features = features

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        return output_data

    def get_params(self):
        return self.operation.get_params()

    def _make_new_table(self, features):
        """
        The method creates a table based on transformed data and source boolean
        features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """

        features_to_process = np.array(features[:, self.ids_to_process])
        # Bool vector - mask for columns
        mask = self.operation.support_
        transformed_part = features_to_process[:, mask]

        # If there are no binary features in the dataset
        if len(self.bool_ids) == 0:
            transformed_features = transformed_part
        else:
            # Stack transformed features and bool features
            bool_features = np.array(features[:, self.bool_ids])
            frames = (bool_features, transformed_part)
            transformed_features = np.hstack(frames)

        return transformed_features


class LinearRegFS(FeatureSelection):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    LinearRegression as core model
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = LinearRegression()
        self.operation = RFE(estimator=self.inner_model)
        self.params = params


class NonLinearRegFS(FeatureSelection):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    DecisionTreeRegressor as core model
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = DecisionTreeRegressor()
        self.operation = RFE(estimator=self.inner_model)
        self.params = params
