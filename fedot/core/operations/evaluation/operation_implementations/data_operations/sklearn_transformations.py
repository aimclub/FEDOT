import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from fedot.core.constants import PCA_MIN_THRESHOLD_TS
from fedot.core.data.data import InputData, OutputData, data_type_is_table
from fedot.core.data.data_preprocessing import convert_into_column, divide_data_categorical_numerical, \
    replace_inf_with_nans
from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation, EncodedInvariantImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID


class ComponentAnalysisImplementation(DataOperationImplementation):
    """
    Class for applying PCA and kernel PCA models from sklearn

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.pca = None
        self.number_of_features = None
        self.number_of_samples = None

    def fit(self, input_data: InputData) -> PCA:
        """
        The method trains the PCA model

        Args:
            input_data: data with features, target and ids for PCA training

        Returns:
            trained PCA model (optional output)
        """

        self.number_of_samples, self.number_of_features = np.array(input_data.features).shape

        if self.number_of_features > 1:
            self.check_and_correct_params(is_ts_data=input_data.data_type is DataTypesEnum.ts)
            # TODO: remove a workaround by refactoring other operations in troubled pipelines (e.g. topo)
            # workaround for NaN-containing arrays during pca fitting, especially for fast_ica
            # fast_ica cannot fit with features represented by a rather sparse matrix
            try:
                self.pca.fit(input_data.features)
            except Exception as e:
                self.log.info(f'Switched from {type(self.pca).__name__} to default PCA on fit stage due to {e}')
                self.pca = PCA()
                self.pca.fit(input_data.features)

        return self.pca

    def transform(self, input_data: InputData) -> OutputData:
        """
        Method for transformation tabular data using PCA

        Args:
            input_data: data with features, target and ids for PCA applying

        Returns:
            data with transformed features attribute
        """

        if self.number_of_features > 1:
            transformed_features = self.pca.transform(input_data.features)
        else:
            transformed_features = input_data.features

        # Update features
        output_data = self._convert_to_output(input_data, transformed_features)
        self.update_column_types(output_data)
        return output_data

    def check_and_correct_params(self, is_ts_data: bool = False):
        """
        Method check if number of features in data enough for ``n_components``
        parameter in PCA or not. And if not enough - fixes it
        """
        n_components = self.params.get('n_components')
        if isinstance(n_components, int):
            if n_components > self.number_of_features:
                self.params.update(n_components=self.number_of_features)
        elif n_components == 'mle':
            # Check that n_samples correctly map with n_features
            if self.number_of_samples < self.number_of_features:
                self.params.update(n_components=0.5)
        if is_ts_data and (n_components * self.number_of_features) < PCA_MIN_THRESHOLD_TS:
            self.params.update(n_components=PCA_MIN_THRESHOLD_TS / self.number_of_features)

        self.pca.set_params(**self.params.to_dict())

    @staticmethod
    def update_column_types(output_data: OutputData) -> OutputData:
        """
        Update column types after applying PCA operations
        """

        _, n_cols = output_data.predict.shape
        output_data.supplementary_data.col_type_ids['features'] = np.array([TYPE_TO_ID[float]] * n_cols)
        return output_data


class PCAImplementation(ComponentAnalysisImplementation):
    """
    Class for applying PCA from sklearn

    Args:
        params: OperationParameters with the hyperparameters
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        if not self.params:
            # Default parameters
            default_params = {'svd_solver': 'full', 'n_components': 'mle'}
            self.params.update(**default_params)
        self.pca = PCA(**self.params.to_dict())
        self.number_of_features = None


class KernelPCAImplementation(ComponentAnalysisImplementation):
    """
    Class for applying kernel PCA from sklearn

    Args:
        params: OperationParameters with the hyperparameters
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.pca = KernelPCA(**self.params.to_dict())


class FastICAImplementation(ComponentAnalysisImplementation):
    """
    Class for applying FastICA from sklearn

    Args:
        params: OperationParameters with the hyperparameters
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.pca = FastICA(**self.params.to_dict())


class PolyFeaturesImplementation(EncodedInvariantImplementation):
    """
    Class for application of :obj:`PolynomialFeatures` operation on data,
    where only not encoded features (were not converted from categorical using
    ``OneHot encoding``) are used

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.th_columns = 10
        if not self.params:
            # Default parameters
            self.operation = PolynomialFeatures(include_bias=False)
        else:
            # Checking the appropriate params are using or not
            poly_params = {k: self.params.get(k) for k in
                           ['degree', 'interaction_only']}
            self.operation = PolynomialFeatures(include_bias=False,
                                                **poly_params)
        self.columns_to_take = None

    def fit(self, input_data: InputData):
        """
        Method for fit Poly features operation
        """
        # Check the number of columns in source dataset
        n_rows, n_cols = input_data.features.shape
        if n_cols > self.th_columns:
            # Randomly choose subsample of features columns - 10 features
            column_indices = np.arange(n_cols)
            self.columns_to_take = np.array(random.sample(list(column_indices), self.th_columns))
            input_data = input_data.subset_features(self.columns_to_take)

        return super().fit(input_data)

    def transform(self, input_data: InputData) -> OutputData:
        """
        Firstly perform filtration of columns
        """

        clipped_input_data = input_data
        if self.columns_to_take is not None:
            clipped_input_data = input_data.subset_features(self.columns_to_take)
        output_data = super().transform(clipped_input_data)

        if self.columns_to_take is not None:
            # Get generated features from poly function
            generated_features = output_data.predict[:, self.th_columns:]
            # Concat source features with generated one
            all_features = np.hstack((input_data.features, generated_features))
            output_data.predict = all_features
        return output_data

    def _update_column_types(self, source_features_shape, output_data: OutputData):
        """Update column types after applying operations. If new columns added, new type for them are defined
        """

        if len(source_features_shape) < 2:
            return output_data
        else:
            cols_number_added = output_data.predict.shape[1] - source_features_shape[1]
            if cols_number_added > 0:
                # There are new columns in the table
                feature_type_ids = output_data.supplementary_data.col_type_ids['features']
                new_types = [TYPE_TO_ID[float]] * cols_number_added
                output_data.supplementary_data.col_type_ids['features'] = np.append(feature_type_ids, new_types)


class ScalingImplementation(EncodedInvariantImplementation):
    """Class for application of ``Scaling operation`` on data,
    where only not encoded features (were not converted from categorical using
    ``OneHot encoding``) are used

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.operation = StandardScaler(**self.params.to_dict())


class NormalizationImplementation(EncodedInvariantImplementation):
    """Class for application of ``MinMax normalization`` operation on data,
    where only not encoded features (were not converted from categorical using
    ``OneHot encoding``) are used

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.operation = MinMaxScaler(**self.params.to_dict())


class ImputationImplementation(DataOperationImplementation):
    """Class for applying imputation on tabular data

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        default_params_categorical = {'strategy': 'most_frequent'}
        self.params_cat = {**self.params.to_dict(), **default_params_categorical}
        self.params_num = self.params.to_dict()
        self.categorical_or_encoded_ids = None
        self.non_categorical_ids = None
        self.ids_binary_integer_features = {}

        self.imputer_cat = SimpleImputer(**self.params_cat)
        self.imputer_num = SimpleImputer(**self.params_num)

    def fit(self, input_data: InputData):
        """The method trains ``SimpleImputer``

        Args:
            input_data: data with features
        """

        replace_inf_with_nans(input_data)
        self._try_remove_empty_columns(input_data)

        if data_type_is_table(input_data):
            self.non_categorical_ids = input_data.numerical_idx

            # The data may have arrived here before categorical data encoding was called.
            if input_data.categorical_idx is not None and input_data.encoded_idx is None:
                self.categorical_or_encoded_ids = input_data.categorical_idx

            # Otherwise, it may have arrived here after categorical data encoding
            elif input_data.encoded_idx is not None:
                self.categorical_or_encoded_ids = input_data.encoded_idx

            # Tabular data contains categorical features
            numerical, categorical = divide_data_categorical_numerical(
                input_data, self.categorical_or_encoded_ids, self.non_categorical_ids
            )

            if categorical is not None and categorical.features.size > 0:
                categorical.features = convert_into_column(categorical.features)
                # Imputing for categorical values
                self.imputer_cat.fit(categorical.features)

            if numerical is not None and numerical.features.size > 0:
                numerical.features = convert_into_column(numerical.features)
                # Imputing for numerical values
                self.imputer_num.fit(numerical.features)
        else:
            # Time series or other type of non-tabular data
            input_data.features = convert_into_column(input_data.features)
            self.imputer_num.fit(input_data.features)

    def transform(self, input_data: InputData) -> OutputData:
        """Method for transformation tabular data using ``SimpleImputer``

        Args:
            input_data: data with features

        Returns:
            data with transformed features attribute
        """

        replace_inf_with_nans(input_data)

        categorical_features, numerical_features, transformed_features = None, None, None

        if data_type_is_table(input_data):
            numerical, categorical = divide_data_categorical_numerical(
                input_data, self.categorical_or_encoded_ids, self.non_categorical_ids
            )

            if categorical is not None:
                categorical_features = convert_into_column(categorical.features)
                categorical_features = self.imputer_cat.transform(categorical_features)

            if numerical is not None:
                numerical_features = convert_into_column(numerical.features)

                # Features with only two unique values must be filled in a specific way
                self._find_binary_features(numerical_features)
                numerical_features = self.imputer_num.transform(numerical_features)
                numerical_features = self._correct_binary_ids_features(numerical_features)

            if categorical_features is not None and numerical_features is not None:
                # Stack both categorical and numerical features
                transformed_features = self._categorical_numerical_union(categorical_features,
                                                                         numerical_features)
            elif categorical_features is not None and numerical_features is None:
                # Dataset contain only categorical features
                transformed_features = categorical_features

            elif categorical is None and numerical is not None:
                # Dataset contain only numerical features
                transformed_features = numerical_features

        else:
            input_data.features = convert_into_column(input_data.features)
            transformed_features = self.imputer_num.transform(input_data.features)

        output_data = self._convert_to_output(input_data, transformed_features, data_type=input_data.data_type)
        return output_data

    def fit_transform(self, input_data: InputData) -> OutputData:
        """Method for training and transformation tabular data using ``SimpleImputer``

        Args:
            input_data: data with features

        Returns:
            data with transformed features attribute
        """

        self.fit(input_data)
        output_data = self.transform_for_fit(input_data)
        return output_data

    def get_params(self) -> OperationParameters:
        features_imputers = {'imputer_categorical': self.params_cat,
                             'imputer_numerical': self.params_num}
        return OperationParameters(**features_imputers)

    def _categorical_numerical_union(self, categorical_features: np.array, numerical_features: np.array) -> np.array:
        """Merge numerical and categorical features in right order (as it was in source table)
        """

        categorical_df = pd.DataFrame(categorical_features, columns=self.categorical_or_encoded_ids)
        numerical_df = pd.DataFrame(numerical_features, columns=self.non_categorical_ids)
        all_features_df = pd.concat([numerical_df, categorical_df], axis=1)

        # Sort column names
        all_features_df = all_features_df.sort_index(axis=1)
        return np.array(all_features_df)

    def _find_binary_features(self, numerical_features: np.array):
        """Find indices of features with only two unique values in column

        Notes:
            All features in table are numerical
        """

        df = pd.DataFrame(numerical_features)

        # Calculate unique values per column (excluding nans)
        for column_id, col in enumerate(df):
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2:
                # Current numerical column has only two values
                column_info = {column_id: {'min': min(unique_values),
                                           'max': max(unique_values)}}
                self.ids_binary_integer_features.update(column_info)

    def _correct_binary_ids_features(self, filled_numerical_features: np.array) -> np.array:
        """ Correct filled features if previously it was binary. Discretization is performed
        for the reconstructed values

        Tip:
            [1, 1, 0.75, 0] will be transformed to [1, 1, 1, 0]
        """

        list_binary_ids = list(self.ids_binary_integer_features.keys())
        if len(list_binary_ids) == 0:
            # Return source array
            return filled_numerical_features

        for bin_id in list_binary_ids:
            # Correct values inplace
            filled_column = filled_numerical_features[:, bin_id]
            min_value = self.ids_binary_integer_features[bin_id]['min']
            max_value = self.ids_binary_integer_features[bin_id]['max']
            mean_value = (max_value - min_value) / 2

            filled_column[filled_column > mean_value] = max_value
            filled_column[filled_column < mean_value] = min_value

        return filled_numerical_features

    def _try_remove_empty_columns(self, input_data: InputData) -> bool:
        """
        Args:
            input_data: data with features

        Modifies:
            - Removes empty columns from input_data.features
            - Updates numerical and categorical indices

        Raises:
            ValueError: InputData is None
        """
        if input_data is None:
            raise ValueError("InputData is None")

        # Drop empty columns
        df_features = pd.DataFrame(input_data.features)
        original_columns = set(df_features.columns)  # get original columns

        df_features = df_features.dropna(axis=1, how='all')
        remaining_columns = set(df_features.columns)  # get columns after removing

        # Get removed elements
        removed_elements = tuple(original_columns - remaining_columns)

        if removed_elements:
            # Update values
            input_data.features = df_features.values

            # Update self and input_data indices
            self._update_indices(input_data, removed_elements)

            return True
        return False

    def _update_indices(self, input_data: InputData, removed_elements: Sequence):
        """
        Args:
            input_data: data with features
            removed_elements: indices of empty columns

        Modifies:
            - Updates indices of input_data
            - Updates indices of self.non_categorical_ids and self.categorical_or_encoded_ids

        Example:
            >>> array_of_indices = [1, 2, 3, 5, 10, 12, 15, 20]
            >>> idxs_of_empty_col = (10, )  # removed element
            >>> array_without_empty_cols = [1, 2, 3, 5, 12, 15, 20]
            >>> self._update_indices(input_data, idxs_of_empty_col)
            >>> # Goal of this function
            >>> updated_array_indices = [1, 2, 3, 5, 11, 14, 19]  # because of decreasing input_data.features shape
        """
        # Shift and update indices
        for element in removed_elements:
            for arr_ref in [("numerical_idx", input_data.numerical_idx),
                            ("categorical_idx", input_data.categorical_idx)]:
                arr_name, arr = arr_ref
                if element in arr:
                    arr = arr[arr != element]  # remove element
                    arr[arr > element] -= 1  # shift the remaining indices

                # Update array
                if arr_name == "numerical_idx":
                    input_data.numerical_idx = arr
                else:
                    input_data.categorical_idx = arr

        # Update self indices
        self.non_categorical_ids = input_data.numerical_idx
        self.categorical_or_encoded_ids = input_data.categorical_idx
