from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

DEFAULT_EXPLAINED_VARIANCE_THR = 0.8
DEFAULT_MIN_EXPLAINED_VARIANCE = 0.01


class PCAOperation:
    """ Adapter class for automatically determining the number of components
    for PCA

    :param params: optional, dictionary with the arguments
    """

    def __init__(self, **params: Optional[dict]):
        if not params:
            self.pca = PCA(svd_solver='randomized', iterated_power='auto')
        else:
            pca_params = {k: params[k] for k in
                          ['svd_solver', 'iterated_power']}
            self.pca = PCA(**pca_params)
        self.params = params

    def fit(self, features):
        """
        The method trains the PCA model and selects only those features in
        which the ratio of the total explained variance reaches the desired
        threshold
        TODO make comparison with PCA model from
         /operations/evaluation/data_evaluation.py
        """
        global DEFAULT_EXPLAINED_VARIANCE_THR
        global DEFAULT_MIN_EXPLAINED_VARIANCE

        self.pca.fit(features)

        # The proportion of the explained variance in the data
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        explained_variance_thr = DEFAULT_EXPLAINED_VARIANCE_THR
        min_explained_variance = DEFAULT_MIN_EXPLAINED_VARIANCE
        if self.params:
            explained_variance_thr = self.params.get('explained_variance')
            min_explained_variance = self.params.get('min_explained_variance')

        # Column ids with attributes that explain the desired ratio of variance
        significant_ids = np.argwhere(cumulative_variance < explained_variance_thr)
        significant_ids = np.ravel(significant_ids)

        if len(significant_ids) > 1:
            # Update amounts of components
            setattr(self.pca, 'n_components', len(significant_ids))
            self.pca.fit(features)
        else:
            pass
        return self.pca

    def transform(self, features):
        transformed_features = self.pca.transform(features)

        return transformed_features


class PolyFeaturesOperation:
    """ Adapter class for application of PolynomialFeatures operation on data,
    where only not encoded features (were not converted from categorical using
    OneHot encoding) are used

    :param params: optional, dictionary with the arguments"""

    def __init__(self, **params: Optional[dict]):
        if not params:
            self.poly_transform = PolynomialFeatures()
        else:
            # TODO implement it - need help and advises
            poly_params = {}
            self.poly_transform = PolynomialFeatures(**poly_params)
        self.params = params

    def fit(self, features):
        """ Method for fit PolyFeatures transformer with automatic determination
        of boolean features, with which there is no need to make transformation

        :param features: tabular data for operation training
        :return encoder: trained transformer (optional output)
        """

        bool_ids, ids_to_process = self._reasonability_check(features)
        self.ids_to_process = ids_to_process
        self.bool_ids = bool_ids

        if len(ids_to_process) > 0:
            features_to_process = np.array(features[:, ids_to_process])
            self.poly_transform.fit(features_to_process)
        else:
            pass

        return self.poly_transform

    def transform(self, features):
        """
        The method that transforms the source features using PolynomialFeatures

        :param features: tabular data for transformation
        :return transformed_features: transformed features table
        """

        if len(self.ids_to_process) > 0:
            transformed_features = self._make_new_table(features)
        else:
            transformed_features = features

        return transformed_features

    def _make_new_table(self, features):
        """
        The method creates a table based on transformed data and source boolean
        features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """

        features_to_process = np.array(features[:, self.ids_to_process])
        transformed_part = self.poly_transform.transform(features_to_process)

        # If there are no binary features in the dataset
        if len(self.bool_ids) == 0:
            transformed_features = transformed_part
        else:
            # Stack transformed features and bool features
            bool_features = np.array(features[:, self.bool_ids])
            frames = (bool_features, transformed_part)
            transformed_features = np.hstack(frames)

        return transformed_features

    @staticmethod
    def _reasonability_check(features):
        """
        Method for checking which columns contain boolean data

        :param features: tabular data for check
        :return bool_ids: indices of boolean columns in table
        :return non_bool_ids: indices of non boolean columns in table
        """
        # TODO perhaps there is a more effective way to do this
        source_shape = features.shape
        columns_amount = source_shape[1]

        bool_ids = []
        non_bool_ids = []
        # For every column in table make check
        for column_id in range(0, columns_amount):
            column = features[:, column_id]
            if len(np.unique(column)) > 2:
                non_bool_ids.append(column_id)
            else:
                bool_ids.append(column_id)

        return bool_ids, non_bool_ids


class OneHotEncodingOperation:
    """ Class for automatic categorical data detection and one hot encoding """

    def __init__(self):
        self.encoder = OneHotEncoder()

    def fit(self, features):
        """ Method for fit encoder with automatic determination of categorical
        features

        :param features: tabular data for operation training
        :return encoder: trained encoder (optional output)
        """

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

    def transform(self, features):
        """
        The method that transforms the categorical features in the original
        dataset, but does not affect the rest

        :param features: tabular data for transformation
        :return transformed_features: transformed features table
        """

        if len(self.categorical_ids) == 0:
            # If there are no categorical features in the table
            transformed_features = features
        else:
            # If categorical features are exists
            transformed_features = self._make_new_table(features)

        return transformed_features

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
