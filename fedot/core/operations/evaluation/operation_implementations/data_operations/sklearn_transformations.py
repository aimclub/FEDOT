from typing import Optional

import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler

from fedot.core.data.data import InputData
from fedot.core.data.data import data_has_categorical_features, divide_data_categorical_numerical, str_columns_check
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
        self.number_of_features = None
        self.parameters_changed = False

    def fit(self, input_data):
        """
        The method trains the PCA model

        :param input_data: data with features, target and ids for PCA training
        :return pca: trained PCA model (optional output)
        """
        self.number_of_features = np.array(input_data.features).shape[1]

        if self.number_of_features > 1:
            self.parameters_changed = self.check_and_correct_params()
            self.pca.fit(input_data.features)

        return self.pca

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        """
        Method for transformation tabular data using PCA

        :param input_data: data with features, target and ids for PCA applying
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return input_data: data with transformed features attribute
        """

        if self.number_of_features > 1:
            transformed_features = self.pca.transform(input_data.features)
        else:
            transformed_features = input_data.features

        # Update features
        output_data = self._convert_to_output(input_data,
                                              transformed_features)
        return output_data

    def check_and_correct_params(self) -> bool:
        """ Method check if amount of features in data enough for n_components
        parameter in PCA or not. And if not enough - fixes it
        """
        was_changed = False
        current_parameters = self.pca.get_params()

        if type(current_parameters['n_components']) == int:
            if current_parameters['n_components'] > self.number_of_features:
                current_parameters['n_components'] = self.number_of_features
                was_changed = True

        self.pca.set_params(**current_parameters)
        self.params = current_parameters

        return was_changed

    def get_params(self):
        if self.parameters_changed is True:
            params_dict = self.pca.get_params()
            return tuple([params_dict, ['n_components']])
        else:
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
        default_params = {
            'drop': 'if_binary'
        }
        if not params:
            # Default parameters
            self.encoder = OneHotEncoder(**default_params)
        else:
            self.encoder = OneHotEncoder(**{**params, **default_params})
        self.categorical_ids = None
        self.non_categorical_ids = None

    def fit(self, input_data: InputData):
        """ Method for fit encoder with automatic determination of categorical features

        :param input_data: data with features, target and ids for encoder training
        :return encoder: trained encoder (optional output)
        """
        features = input_data.features
        categorical_ids, non_categorical_ids = str_columns_check(features)

        # Indices of columns with categorical and non-categorical features
        self.categorical_ids = categorical_ids
        self.non_categorical_ids = non_categorical_ids

        if len(categorical_ids) == 0:
            pass
        else:
            categorical_features = np.array(features[:, categorical_ids])
            self.encoder.fit(categorical_features)

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        """
        The method that transforms the categorical features in the original
        dataset, but does not affect the rest features

        :param input_data: data with features, target and ids for transformation
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
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
        self._check_same_categories(categorical_features)
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

    def _check_same_categories(self, categorical_features):
        encoder_unique_categories = sorted(list(np.hstack(self.encoder.categories_)))
        features_unique_categories = sorted(np.unique(np.array(categorical_features)))

        if encoder_unique_categories != features_unique_categories:
            raise ValueError('Category in test data was not exist in train.')

    def get_params(self):
        return self.encoder.get_params()


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
        default_params_categorical = {'strategy': 'most_frequent'}
        self.params_cat = {**params, **default_params_categorical}
        self.params_num = params

        if not params:
            # Default parameters
            self.imputer_cat = SimpleImputer(**default_params_categorical)
            self.imputer_num = SimpleImputer()
        else:
            self.imputer_cat = SimpleImputer(**self.params_cat)
            self.imputer_num = SimpleImputer(**self.params_num)

    def fit(self, input_data: InputData):
        """
        The method trains SimpleImputer

        :param input_data: data with features
        :return imputer: trained SimpleImputer model
        """

        features_with_replaced_inf = np.where(np.isin(input_data.features,
                                                      [np.inf, -np.inf]),
                                              np.nan,
                                              input_data.features)
        input_data.features = features_with_replaced_inf

        if data_has_categorical_features(input_data):
            numerical, categorical = divide_data_categorical_numerical(input_data)
            if len(categorical.features.shape) == 1:
                self.imputer_cat.fit(categorical.features.reshape(-1, 1))
            else:
                self.imputer_cat.fit(categorical.features)
            if len(numerical.features.shape) == 1:
                self.imputer_num.fit(numerical.features.reshape(-1, 1))
            else:
                self.imputer_num.fit(numerical.features)
        else:
            if len(input_data.features.shape) == 1:
                self.imputer_num.fit(input_data.features.reshape(-1, 1))
            else:
                self.imputer_num.fit(input_data.features)

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool] = None):
        """
        Method for transformation tabular data using SimpleImputer

        :param input_data: data with features
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return input_data: data with transformed features attribute
        """
        features_with_replaced_inf = np.where(np.isin(input_data.features,
                                                      [np.inf, -np.inf]),
                                              np.nan,
                                              input_data.features)
        input_data.features = features_with_replaced_inf

        if data_has_categorical_features(input_data):
            numerical, categorical = divide_data_categorical_numerical(input_data)
            if len(categorical.features.shape) == 1:
                categorical_features = self.imputer_cat.transform(categorical.features.reshape(-1, 1))
            else:
                categorical_features = self.imputer_cat.transform(categorical.features)
            if len(numerical.features.shape) == 1:
                numerical_features = self.imputer_num.transform(numerical.features.reshape(-1, 1))
            else:
                numerical_features = self.imputer_num.transform(numerical.features)
            transformed_features = np.hstack((categorical_features, numerical_features))
        else:
            if len(input_data.features.shape) == 1:
                transformed_features = self.imputer_num.transform(input_data.features.reshape(-1, 1))
            else:
                transformed_features = self.imputer_num.transform(input_data.features)

        output_data = self._convert_to_output(input_data, transformed_features, data_type=input_data.data_type)
        return output_data

    def fit_transform(self, input_data, is_fit_pipeline_stage: Optional[bool] = None):
        """
        Method for training and transformation tabular data using SimpleImputer

        :param input_data: data with features
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return input_data: data with transformed features attribute
        """
        self.fit(input_data)
        output_data = self.transform(input_data)
        return output_data

    def get_params(self) -> dict:
        dictionary = {'imputer_categorical': self.params_cat, 'imputer_numerical': self.params_num}
        return dictionary
