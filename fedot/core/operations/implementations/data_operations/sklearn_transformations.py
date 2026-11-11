import random
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from fedot.core.data.input_data.data import InputData, OutputData, data_type_is_table
from fedot.core.operations.implementations.base import DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.preprocessing.data_preprocessing import (
    convert_into_column,
    divide_data_categorical_numerical,
    replace_inf_with_nans,
)
from fedot.preprocessing.data_types import TYPE_TO_ID


class EncodedInvariantImplementation(DataOperationImplementation):
    """ Class for processing data without transforming encoded features.
    Encoded features - features after OneHot encoding operation, when one
    feature (with categorical values) can be represented as several boolean
    vectors
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.operation = None
        self.ids_to_process = None
        self.bool_ids = None

    def fit(self, input_data: InputData):
        """ Method for fit transformer with automatic determination
        of boolean features, with which there is no need to make transformation

        :param input_data: data with features, target and ids to process
        :return operation: trained transformer (optional output)
        """

        if input_data.task.task_type.name == 'ts_forecasting' and input_data.features.ndim == 2:
            features = input_data.features.ravel()
        else:
            features = input_data.features

        # Find boolean columns in features table
        bool_ids, ids_to_process = self._reasonability_check(features)
        self.ids_to_process = ids_to_process
        self.bool_ids = bool_ids
        if len(ids_to_process) > 0:
            if isinstance(features, np.ndarray):
                if input_data.task.task_type.name == 'ts_forecasting' and input_data.features.ndim == 2:
                    features = features.reshape(-1, 1)

                features_to_process = np.array(
                    features[:, ids_to_process]) if features.ndim > 1 else features
            else:
                features_to_process = np.array(
                    features.iloc[:, ids_to_process]) if features.ndim > 1 else features
            self.operation.fit(features_to_process)
        return self.operation

    def transform(self, input_data: InputData) -> OutputData:
        """
        The method that transforms the source features using "operation" for predict stage

        :param input_data: tabular data with features, target and ids to process
        :return output_data: output data with transformed features table
        """
        source_features_shape = input_data.features.shape
        features = input_data.features
        if len(self.ids_to_process) > 0:
            transformed_features = self._make_new_table(features)
        else:
            transformed_features = features

        transformed_features = np.nan_to_num(
            transformed_features, copy=False, nan=0, posinf=0, neginf=0)

        # Update features and column types
        output_data = self._convert_to_output(input_data, transformed_features)
        self._update_column_types(source_features_shape, output_data)
        return output_data

    def _make_new_table(self, features):
        """
        The method creates a table based on transformed data and source boolean
        features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """
        if isinstance(features, np.ndarray):
            features_to_process = np.array(
                features[:, self.ids_to_process]) if features.ndim > 1 else features.copy()
        else:
            features_to_process = np.array(
                features.iloc[:, self.ids_to_process]
            ) if features.ndim > 1 else features.copy()

        transformed_part = self.operation.transform(features_to_process)

        # If there are no binary features in the dataset
        if len(self.bool_ids) == 0:
            transformed_features = transformed_part
        else:
            # Stack transformed features and bool features
            if isinstance(features, np.ndarray):
                bool_features = np.array(features[:, self.bool_ids])
            else:
                bool_features = np.array(features[self.bool_ids])

            frames = (bool_features, transformed_part)
            transformed_features = np.hstack(frames)

        return transformed_features

    def _update_column_types(self, source_features_shape, output_data: OutputData) -> OutputData:
        """
        Update column types after applying operations.
        If new columns added, new type for them are defined
        """
        return output_data

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

        columns_amount = source_shape[1] if len(source_shape) > 1 else 1

        # Indices of boolean columns in features table
        bool_ids = []
        non_bool_ids = []

        # For every column in table make check
        for column_id in range(columns_amount):
            if isinstance(features, np.ndarray):
                column = features[:,
                                  column_id] if columns_amount > 1 else features.copy()
            else:
                column = features.iloc[:, column_id] if columns_amount > 1 else features.copy(
                )

            if (isinstance(column, pd.Series) and len(set(column)) > 2) or \
               (isinstance(column, np.ndarray) and len(np.unique(column)) > 2):
                non_bool_ids.append(column_id)
            else:
                bool_ids.append(column_id)

        return bool_ids, non_bool_ids


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
            self.columns_to_take = np.array(random.sample(
                list(column_indices), self.th_columns))
            input_data = input_data.subset_features(self.columns_to_take)

        return super().fit(input_data)

    def transform(self, input_data: InputData) -> OutputData:
        """
        Firstly perform filtration of columns
        """

        clipped_input_data = input_data
        if self.columns_to_take is not None:
            clipped_input_data = input_data.subset_features(
                self.columns_to_take)
        try:
            output_data = super().transform(clipped_input_data)
        except Exception:
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
            cols_number_added = output_data.predict.shape[1] - \
                source_features_shape[1]
            if cols_number_added > 0:
                # There are new columns in the table
                if output_data.supplementary_data.col_type_ids is None:
                    output_data.supplementary_data.col_type_ids = {}

                feature_type_ids = output_data.supplementary_data.col_type_ids.get(
                    'features')
                if feature_type_ids is None:
                    feature_type_ids = np.array(
                        [TYPE_TO_ID[float]] * source_features_shape[1])
                new_types = [TYPE_TO_ID[float]] * cols_number_added
                output_data.supplementary_data.col_type_ids['features'] = np.append(
                    feature_type_ids, new_types)


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
        self.params_cat = {
            **self.params.to_dict(), **default_params_categorical}
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
                categorical.features = convert_into_column(
                    categorical.features)
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

        categorical_features, numerical_features = None, None

        if data_type_is_table(input_data):
            numerical, categorical = divide_data_categorical_numerical(
                input_data, self.categorical_or_encoded_ids, self.non_categorical_ids
            )

            if categorical is not None:
                categorical_features = convert_into_column(
                    categorical.features)
                categorical_features = self.imputer_cat.transform(
                    categorical_features)

            if numerical is not None:
                numerical_features = convert_into_column(numerical.features)

                # Features with only two unique values must be filled in a specific way
                self._find_binary_features(numerical_features)
                numerical_features = self.imputer_num.transform(
                    numerical_features)
                numerical_features = self._correct_binary_ids_features(
                    numerical_features)

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
            transformed_features = self.imputer_num.transform(
                input_data.features)

        output_data = self._convert_to_output(
            input_data, transformed_features, data_type=input_data.data_type)
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

    def _categorical_numerical_union(self, categorical_features: np.array, numerical_features: np.array) -> np.array:
        """Merge numerical and categorical features in right order (as it was in source table)
        """

        categorical_df = pd.DataFrame(
            categorical_features, columns=self.categorical_or_encoded_ids)
        numerical_df = pd.DataFrame(
            numerical_features, columns=self.non_categorical_ids)
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
            mean_value = (max_value + min_value) / 2

            filled_column[filled_column > mean_value] = max_value
            filled_column[filled_column < mean_value] = min_value

        return filled_numerical_features

    def get_params(self) -> OperationParameters:
        features_imputers = {'imputer_categorical': self.params_cat,
                             'imputer_numerical': self.params_num}
        return OperationParameters(**features_imputers)
