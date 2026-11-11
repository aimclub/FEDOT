from dataclasses import replace
from itertools import combinations, combinations_with_replacement
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

from fedot.core.data.input_data.data import InputData, OutputData, data_type_is_table
from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import DataOperationImplementation, TensorDataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.preprocessing.data_preprocessing import (
    convert_into_column, divide_data_categorical_numerical, replace_inf_with_nans,
)
from fedot.preprocessing.methods.scaling_normalization import StandartScaling, MinMaxNormalization


def non_binary_columns(features: torch.Tensor) -> torch.Tensor:
    """Fit the historical <=2 unique-values exclusion, counting NaN once."""
    values = features.sort(dim=0).values
    different = values[1:] != values[:-1]
    both_nan = torch.isnan(values[1:]) & torch.isnan(values[:-1])
    counts = (different & ~both_nan).sum(dim=0) + 1
    return torch.nonzero(counts > 2, as_tuple=True)[0]


class ScalingImplementation(TensorDataOperationImplementation):
    """Torch standard scaling, retaining binary columns and their positions."""

    handler_type = StandartScaling

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        options = self.params.to_dict()
        # TensorData operations own their transformed features, regardless of sklearn's copy flag.
        options.pop('copy', None)
        self.operation = self.handler_type(**options)
        self.ids_to_process = None

    def fit(self, data: TensorData):
        self.ids_to_process = non_binary_columns(data.features)
        if self.ids_to_process.numel():
            features = data.features if data.features.is_floating_point() else data.features.to(torch.float64)
            self.operation.fit(PreparedData(features=features), self.ids_to_process)
        return self

    def transform(self, data: TensorData) -> TensorData:
        features = data.features.clone()
        if self.ids_to_process.numel():
            if not features.is_floating_point():
                features = features.to(torch.float64)
            features = self.operation.transform(PreparedData(features=features)).features
        features = torch.nan_to_num(features, nan=0, posinf=0, neginf=0)
        return replace(data, features=features, predict=None, fingerprint=None)


class NormalizationImplementation(ScalingImplementation):
    """Torch min-max normalization with the same fitted binary-column policy."""

    handler_type = MinMaxNormalization


class PolyFeaturesImplementation(TensorDataOperationImplementation):
    """Native Torch polynomial features, preserving the original column order.

    At most ten source columns are sampled, as in the historical implementation.
    Binary columns are retained but do not participate in generated monomials.
    Only small combinatorial index metadata is built in Python.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.th_columns = 10
        self.columns_to_take = None
        self.retained_columns = None
        self.terms = None
        self.term_groups = None
        self.has_non_binary_columns = False

    def fit(self, data: TensorData):
        width = data.features.shape[1]
        self.columns_to_take = random.sample(range(width), self.th_columns) if width > self.th_columns else None
        columns = self.columns_to_take if self.columns_to_take is not None else list(range(width))
        selected = data.features[:, columns]
        indices = non_binary_columns(selected)
        # Column numbers are metadata; feature values never leave the input device.
        active = [columns[int(index)] for index in indices]
        self.has_non_binary_columns = bool(active)
        binary = [column for column in columns if column not in active]
        if active and not torch.isfinite(data.features[:, active]).all():
            raise ValueError('Polynomial features require finite non-binary columns.')
        degree = self.params.get('degree', 2)
        low, high = degree if isinstance(degree, tuple) else (0, degree)
        if low < 0 or high < low:
            raise ValueError('Polynomial degree must satisfy 0 <= min_degree <= max_degree.')
        combine = combinations if self.params.get('interaction_only', False) else combinations_with_replacement
        terms = [(column,) for column in binary]
        terms.extend(term for power in range(max(1, low), high + 1) for term in combine(active, power))
        if self.columns_to_take is not None:
            self.retained_columns = list(range(width))
            self.terms = terms[len(columns):]
        else:
            self.retained_columns = sorted(term[0] for term in terms if len(term) == 1)
            self.terms = [term for term in terms if len(term) > 1]
        if not self.retained_columns and not self.terms:
            raise ValueError('Polynomial features with include_bias=False produce an empty feature set.')
        self.term_groups = [torch.tensor([term for term in self.terms if len(term) == power],
                                        dtype=torch.long, device=data.device)
                            for power in sorted({len(term) for term in self.terms})]
        return self

    def transform(self, data: TensorData) -> TensorData:
        source = data.features
        if self.has_non_binary_columns and not source.is_floating_point():
            source = source.to(torch.float64)
        blocks = [source[:, self.retained_columns]]
        blocks.extend(source[:, group.to(data.device)].prod(dim=-1) for group in self.term_groups)
        features = torch.nan_to_num(torch.cat(blocks, dim=1), nan=0, posinf=0, neginf=0)
        retained = self.retained_columns
        mapping = {pos: data.idx_mapping.get(column, column) for pos, column in enumerate(retained)}
        first_derived = max(data.idx_mapping.values(), default=data.features.shape[1] - 1) + 1
        mapping.update({len(retained) + pos: first_derived + pos for pos in range(len(self.terms))})
        categorical = [pos for pos, column in enumerate(retained) if column in data.categorical_idx]
        names = None
        if data.features_names is not None:
            names = [data.features_names[column] for column in retained]
            names += [' '.join(str(data.features_names[column]) for column in term) for term in self.terms]
        return replace(data, features=features, predict=None, idx_mapping=mapping,
                       categorical_idx=categorical,
                       numerical_idx=[pos for pos in range(features.shape[1]) if pos not in categorical],
                       features_names=names, fingerprint=None)


class ImputationImplementation(DataOperationImplementation):
    """Legacy InputData imputer retained for deferred TS/Industrial runtime

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
