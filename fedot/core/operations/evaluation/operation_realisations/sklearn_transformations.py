from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, \
    StandardScaler, MinMaxScaler
from fedot.core.operations.evaluation.operation_realisations.\
    abs_interfaces import OperationRealisation, EncodedInvariantOperation

DEFAULT_EXPLAINED_VARIANCE_THR = 0.8
DEFAULT_MIN_EXPLAINED_VARIANCE = 0.01


class PCAOperation(OperationRealisation):
    """ Adapter class for automatically determining the number of components
    for PCA

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.pca = PCA(svd_solver='randomized', iterated_power='auto')
        else:
            # Checking the appropriate params are using or not
            pca_params = {k: params[k] for k in
                          ['svd_solver', 'iterated_power']}
            self.pca = PCA(**pca_params)
        self.params = params

    def fit(self, input_data):
        """
        The method trains the PCA model and selects only those features in
        which the ratio of the total explained variance reaches the desired
        threshold
        TODO make comparison with PCA model from
         /operations/evaluation/data_evaluation.py

        :param input_data: data with features, target and ids for PCA training
        :return pca: trained PCA model (optional output)
        """
        global DEFAULT_EXPLAINED_VARIANCE_THR
        global DEFAULT_MIN_EXPLAINED_VARIANCE

        self.pca.fit(input_data.features)

        # The proportion of the explained variance in the data
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        explained_variance_thr = DEFAULT_EXPLAINED_VARIANCE_THR
        min_explained_variance = DEFAULT_MIN_EXPLAINED_VARIANCE
        if self.params:
            explained_variance_thr = self.params.get('explained_variance_thr')
            min_explained_variance = self.params.get('min_explained_variance')

        # Column ids with attributes that explain the desired ratio of variance
        significant_ids = np.argwhere(cumulative_variance < explained_variance_thr)
        significant_ids = np.ravel(significant_ids)

        print(len(significant_ids))
        if len(significant_ids) > 1:
            # Update amounts of components
            setattr(self.pca, 'n_components', len(significant_ids))
            self.pca.fit(input_data.features)
        else:
            pass
        return self.pca

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for transformation tabular data using PCA

        :param input_data: data with features, target and ids for PCA applying
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        transformed_features = self.pca.transform(input_data.features)

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        return output_data

    def get_params(self):
        return self.pca.get_params()


class OneHotEncodingOperation(OperationRealisation):
    """ Class for automatic categorical data detection and one hot encoding """

    def __init__(self):
        super().__init__()
        self.encoder = OneHotEncoder()
        self.categorical_ids = None
        self.non_categorical_ids = None

    def fit(self, input_data):
        """ Method for fit encoder with automatic determination of categorical
        features

        :param input_data: data with features, target and ids for encoder training
        :return encoder: trained encoder (optional output)
        """
        features = input_data.features
        categorical_ids, non_categorical_ids = self._str_columns_check(features)

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
        dataset, but does not affect the rest

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
    def _str_columns_check(features):
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


class PolyFeaturesOperation(EncodedInvariantOperation):
    """ Adapter class for application of PolynomialFeatures operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.operation = PolynomialFeatures(include_bias=False)
        else:
            # Checking the appropriate params are using ow not
            poly_params = {k: params[k] for k in
                           ['degree', 'interaction_only']}
            self.operation = PolynomialFeatures(include_bias=False,
                                                **poly_params)
        self.params = params


class ScalingOperation(EncodedInvariantOperation):
    """ Adapter class for application of Scaling operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.operation = StandardScaler()
        else:
            # TODO implement it - need help and advises
            scaling_params = {}
            self.operation = StandardScaler(**scaling_params)
        self.params = params


class NormalizationOperation(EncodedInvariantOperation):
    """ Adapter class for application of MinMax normalization operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.operation = MinMaxScaler()
        else:
            # TODO implement it - need help and advises
            normalization_params = {}
            self.operation = MinMaxScaler(**normalization_params)
        self.params = params
