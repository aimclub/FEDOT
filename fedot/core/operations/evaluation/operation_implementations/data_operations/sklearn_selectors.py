from typing import Optional

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fedot.core.data.data import OutputData, InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FeatureSelectionImplementation(DataOperationImplementation):
    """ Class for applying feature selection operations on tabular data """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = None
        self.operation = None
        self.is_not_fitted = None

        # Number of columns in features table
        self.features_columns_number = None

        # Bool mask where True - remain column and False - drop it
        self.remain_features_mask = None

    def fit(self, input_data: InputData):
        """ Method for fit feature selection

        :param input_data: data with features, target and ids to process
        :return operation: trained operation (optional output)
        """
        features = input_data.features
        target = input_data.target

        # Define number of columns in the features table
        if len(features.shape) == 1:
            self.features_columns_number = 1
        else:
            self.features_columns_number = features.shape[1]

        if self.features_columns_number > 1:
            if self._is_input_data_one_dimensional(features):
                self.is_not_fitted = True
                return self.operation
            try:
                self.operation.fit(features, target)
            except ValueError:
                # For time series forecasting not available multi-targets
                self.operation.fit(features, target[:, 0])
        else:
            self.is_not_fitted = True
        return self.operation

    def transform(self, input_data: InputData) -> OutputData:
        """ Method for making prediction for prediction stage

        :param input_data: data with features, target and ids to process
        :return output_data: filtered input data by columns
        """
        if self.is_not_fitted:
            return self._convert_to_output(input_data, input_data.features)

        features = input_data.features
        source_features_shape = features.shape
        transformed_features = self._make_new_table(features)

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        self._update_column_types(source_features_shape, output_data)
        return output_data

    def _update_column_types(self, source_features_shape, output_data: OutputData):
        """ Update column types after applying feature selection operations """
        if len(source_features_shape) < 2:
            return output_data
        else:
            if self.features_columns_number > 1:
                cols_number_removed = source_features_shape[1] - output_data.predict.shape[1]
                if cols_number_removed > 0:
                    # There are several columns, which were dropped
                    col_types = output_data.supplementary_data.column_types['features']

                    # Calculate
                    remained_column_types = np.array(col_types)[self.remain_features_mask]
                    output_data.supplementary_data.column_types['features'] = list(remained_column_types)

    def _make_new_table(self, features):
        """
        The method creates a table based on transformed data and source boolean
        features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """

        # Bool vector - mask for columns
        self.remain_features_mask = self.operation.support_
        transformed_features = features[:, self.remain_features_mask]
        return transformed_features

    @staticmethod
    def _is_input_data_one_dimensional(features_to_process: np.array):
        """ Check if features table contain only one column """
        return features_to_process.shape[1] == 1


class LinearRegFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    LinearRegression as core model
    Task type - regression
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = LinearRegression()

        if not self.params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: self.params.get(k) for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)


class NonLinearRegFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    DecisionTreeRegressor as core model
    Task type - regression
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = DecisionTreeRegressor()

        if not self.params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: self.params.get(k) for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)


class LinearClassFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    LogisticRegression as core model
    Task type - classification
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = LogisticRegression()

        if not self.params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: self.params.get(k) for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)


class NonLinearClassFSImplementation(FeatureSelectionImplementation):
    """
    Class for feature selection based on Recursive Feature Elimination (RFE) and
    DecisionTreeClassifier as core model
    Task type - classification
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.inner_model = DecisionTreeClassifier()

        if not self.params:
            # Default parameters
            self.operation = RFE(estimator=self.inner_model)
        else:
            # Checking the appropriate params are using or not
            rfe_params = {k: self.params.get(k) for k in
                          ['n_features_to_select', 'step']}
            self.operation = RFE(estimator=self.inner_model, **rfe_params)
