from abc import abstractmethod
from typing import Optional

import numpy as np
from sklearn.linear_model import RANSACRegressor, \
    LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import RFE

from fedot.core.operations.evaluation.\
    operation_realisations.abs_interfaces import reasonability_check


class FeatureSelection:
    """ Base class for applying feature selection operations on tabular data """

    def __init__(self):
        self.inner_model = None
        self.operation = None

    def fit(self, features, target):
        """ Method for fit feature selection

        :param features: tabular data for operation training
        :param target: target output
        :return operation: trained operation (optional output)
        """

        bool_ids, ids_to_process = reasonability_check(features)
        self.ids_to_process = ids_to_process
        self.bool_ids = bool_ids

        if len(ids_to_process) > 0:
            features_to_process = np.array(features[:, ids_to_process])
            self.operation.fit(features_to_process, target)
        else:
            pass

        return self.operation

    def transform(self, features, is_fit_chain_stage: bool):
        """ Method for making prediction

        :param features: tabular data for filtering
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return inner_features: filtered rows
        """
        if len(self.ids_to_process) > 0:
            transformed_features = self._make_new_table(features)
        else:
            transformed_features = features

        return transformed_features

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
