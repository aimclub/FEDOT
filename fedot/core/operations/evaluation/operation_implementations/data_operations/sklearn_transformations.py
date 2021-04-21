from typing import Optional

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, \
    StandardScaler, MinMaxScaler

from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation, EncodedInvariantImplementation


class ComponentAnalysisImplementation(DataOperationImplementation):
    """ Class for applying PCA and kernel PCA models form sklearn

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.pca = None
        self.params = None
        self.amount_of_features = None

    def fit(self, input_data):
        """
        The method trains the PCA model

        :param input_data: data with features, target and ids for PCA training
        :return pca: trained PCA model (optional output)
        """
        self.amount_of_features = np.array(input_data.features).shape[1]

        if self.amount_of_features > 1:
            self.check_and_correct_params()
            self.pca.fit(input_data.features)

        return self.pca

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for transformation tabular data using PCA

        :param input_data: data with features, target and ids for PCA applying
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        if self.amount_of_features > 1:
            transformed_features = self.pca.transform(input_data.features)
        else:
            transformed_features = input_data.features

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        return output_data

    def check_and_correct_params(self) -> None:
        """ Method check if amount of features in data enough for n_components
        parameter in PCA or not. And if not enough - fixes it
        """
        current_parameters = self.pca.get_params()

        if type(current_parameters['n_components']) == int:
            if current_parameters['n_components'] > self.amount_of_features:
                current_parameters['n_components'] = self.amount_of_features

        self.pca.set_params(**current_parameters)
        self.params = current_parameters

    def get_params(self):
        return self.pca.get_params()


class PCAImplementation(ComponentAnalysisImplementation):
    """ Class for applying PCA from sklearn

    :param params: optional, dictionary with the hyperparameters
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.pca = PCA(svd_solver='full', n_components='mle')
        else:
            self.pca = PCA(**params)
        self.params = params
        self.amount_of_features = None


class KernelPCAImplementation(ComponentAnalysisImplementation):
    """ Class for applying kernel PCA from sklearn

    :param params: optional, dictionary with the hyperparameters
    """
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.pca = KernelPCA()
        else:
            self.pca = KernelPCA(**params)
        self.params = params


class OneHotEncodingImplementation(DataOperationImplementation):
    """ Class for automatic categorical data detection and one hot encoding """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.encoder = OneHotEncoder()
        else:
            self.encoder = OneHotEncoder(**params)
        self.categorical_ids = None
        self.non_categorical_ids = None

    def fit(self, input_data):
        """ Method for fit encoder with automatic determination of categorical
        features

        :param input_data: data with features, target and ids for encoder training
        :return encoder: trained encoder (optional output)
        """
        features = input_data.features
        categorical_ids, non_categorical_ids = self.str_columns_check(features)

        # Indices of columns with categorical and non-categorical features
        self.categorical_ids = categorical_ids
        self.non_categorical_ids = non_categorical_ids

        if len(categorical_ids) == 0:
            pass
        else:
            categorical_features = np.array(features[:, categorical_ids])
            self.encoder.fit(categorical_features)

        return self.encoder

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        The method that transforms the categorical features in the original
        dataset, but does not affect the rest features

        :param input_data: data with features, target and ids for transformation
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with transformed features table
        """

        features = input_data.features
        if len(self.categorical_ids) == 0:
            # If there are no categorical features in the table
            transformed_features = features
        else:
            # If categorical features are exists
            transformed_features = self._make_new_table(features)

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        return output_data

    def _make_new_table(self, features):
        """
        The method creates a table based on categorical and real features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """

        categorical_features = np.array(features[:, self.categorical_ids])
        transformed_categorical = self.encoder.transform(categorical_features).toarray()

        # If there are non-categorical features in the data
        if len(self.non_categorical_ids) == 0:
            transformed_features = transformed_categorical
        else:
            # Stack transformed categorical and non-categorical data
            non_categorical_features = np.array(features[:, self.non_categorical_ids])
            frames = (non_categorical_features, transformed_categorical)
            transformed_features = np.hstack(frames)

        return transformed_features

    def get_params(self):
        return self.encoder.get_params()

    @staticmethod
    def str_columns_check(features):
        """
        Method for checking which columns contain categorical (text) data

        :param features: tabular data for check
        :return categorical_ids: indices of categorical columns in table
        :return non_categorical_ids: indices of non categorical columns in table
        """
        source_shape = features.shape
        columns_amount = source_shape[1]

        categorical_ids = []
        non_categorical_ids = []
        # For every column in table make check for first element
        for column_id in range(0, columns_amount):
            column = features[:, column_id]
            if type(column[0]) == str:
                categorical_ids.append(column_id)
            else:
                non_categorical_ids.append(column_id)

        return categorical_ids, non_categorical_ids


class PolyFeaturesImplementation(EncodedInvariantImplementation):
    """ Class for application of PolynomialFeatures operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.operation = PolynomialFeatures(include_bias=False)
        else:
            # Checking the appropriate params are using or not
            poly_params = {k: params[k] for k in
                           ['degree', 'interaction_only']}
            self.operation = PolynomialFeatures(include_bias=False,
                                                **poly_params)
        self.params = params

    def get_params(self):
        return self.operation.get_params()


class ScalingImplementation(EncodedInvariantImplementation):
    """ Class for application of Scaling operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.operation = StandardScaler()
        else:
            self.operation = StandardScaler(**params)
        self.params = params

    def get_params(self):
        return self.operation.get_params()


class NormalizationImplementation(EncodedInvariantImplementation):
    """ Class for application of MinMax normalization operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.operation = MinMaxScaler()
        else:
            self.operation = MinMaxScaler(**params)
        self.params = params

    def get_params(self):
        return self.operation.get_params()


class ImputationImplementation(DataOperationImplementation):
    """ Class for applying imputation on tabular data

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            # Default parameters
            self.imputer = SimpleImputer()
        else:
            self.imputer = SimpleImputer(**params)
        self.params = params

    def fit(self, input_data):
        """
        The method trains SimpleImputer

        :param input_data: data with features
        :return imputer: trained SimpleImputer model
        """

        self.imputer.fit(input_data.features)
        return self.imputer

    def transform(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """
        Method for transformation tabular data using SimpleImputer

        :param input_data: data with features
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """
        transformed_features = self.imputer.transform(input_data.features)

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        return output_data

    def get_params(self):
        return self.imputer.get_params()
