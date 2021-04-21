from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import RFE

from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import EncodedInvariantImplementation


class FeatureSelectionImplementation(EncodedInvariantImplementation):
    """ Class for applying feature selection operations on tabular data """

    def __init__(self, **params: Optional[dict]):
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
            try:
                self.operation.fit(features_to_process, target)
            except ValueError:
                # For time series forecasting not available multi-targets
                self.operation.fit(features_to_process, target[:, 0])
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


class LinearRegFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    LinearRegression as core model
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = LinearRegression(normalize=True)

        if not params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: params[k] for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)
        self.params = params


class NonLinearRegFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    DecisionTreeRegressor as core model
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = DecisionTreeRegressor()

        if not params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: params[k] for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)
        self.params = params


class LinearClassFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    LogisticRegression as core model
    Task type - classification
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = LogisticRegression()

        if not params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: params[k] for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)
        self.params = params


class NonLinearClassFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    DecisionTreeClassifier as core model
    Task type - classification
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = DecisionTreeClassifier()

        if not params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: params[k] for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)
        self.params = params
