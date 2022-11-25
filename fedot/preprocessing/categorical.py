from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.preprocessing.data_types import NAME_CLASS_INT, FEDOT_STR_NAN


class BinaryCategoricalPreprocessor:
    """ Class for categories features preprocessing: converting binary string features into integers. """

    def __init__(self):
        self.binary_encoders = {}
        self.binary_ids_to_convert = []

        # List with binary categorical features ids which contain Nans
        self.binary_features_with_nans = []

    def fit(self, input_data: InputData):
        """
        Find indices of columns which are contains categorical values. Binary features and at the same time
        has str objects. If there are such features - convert it into int
        """
        features_types = input_data.supplementary_data.column_types['features']
        categorical_ids, _ = find_categorical_columns(table=input_data.features,
                                                      column_types=features_types)
        if len(categorical_ids) == 0:
            # There is no need to process categorical features
            return self

        binary_ids_to_convert = []
        number_of_columns = input_data.features.shape[-1]
        for column_id in range(number_of_columns):
            pd_column = pd.Series(input_data.features[:, column_id], copy=True)
            has_nan = pd_column.isna()
            if has_nan.sum() and column_id in categorical_ids:
                # This categorical column has nans
                replaced_column, _ = replace_nans_with_fedot_nans(pd_column, has_nan)
                column_uniques = replaced_column.unique()

                if len(column_uniques) <= 3:
                    # There is column with binary categories and gaps
                    self.binary_features_with_nans.append(column_id)
                    binary_ids_to_convert.append(column_id)
                    self._train_encoder(replaced_column, column_id)
            else:
                column_uniques = pd_column.unique()
                if len(column_uniques) <= 2 and column_id in categorical_ids:
                    # Column contains binary string feature
                    binary_ids_to_convert.append(column_id)

                    # Train encoder for current column
                    self._train_encoder(pd_column, column_id)

        self.binary_ids_to_convert = binary_ids_to_convert
        return self

    def transform(self, input_data: InputData) -> InputData:
        """
        Apply transformation (converting str into integers) for selected (while training) features.
        """
        if len(self.binary_ids_to_convert) == 0:
            # There are no binary categorical features
            return input_data

        converted_features = []
        number_of_columns = input_data.features.shape[-1]
        for column_id in range(number_of_columns):
            if column_id in self.binary_ids_to_convert:
                # If column contains nans - replace them with fedot nans special string
                pd_column = pd.Series(input_data.features[:, column_id])
                has_nan = pd_column.isna()
                replaced_column, gap_ids = replace_nans_with_fedot_nans(pd_column, has_nan)

                # Convert into integers
                converted_column = self._apply_encoder(replaced_column, column_id, gap_ids)
            else:
                # Stay column the same
                converted_column = input_data.features[:, column_id]

            converted_features.append(converted_column.reshape((-1, 1)))

        # Store transformed features
        copied_data = deepcopy(input_data)
        copied_data.features = np.hstack(converted_features)

        # Update features types
        features_types = copied_data.supplementary_data.column_types['features']
        for converted_column_id in self.binary_ids_to_convert:
            features_types[converted_column_id] = NAME_CLASS_INT
        return copied_data

    def fit_transform(self, input_data: InputData) -> InputData:
        """
        Applies firstly :method:`fit` and then :method:`transform` methods of the class

        Args:
            input_data: to be trained on and transformed

        Returns:
            transformed ``input_data``
        """
        self.fit(input_data)
        return self.transform(input_data)

    def _train_encoder(self, column: pd.Series, column_id: int):
        """ Convert labels in the column from string into int via Label encoding.
        So, Label encoder is fitted to do such transformation.
        """
        encoder = LabelEncoder()
        encoder.fit(column)

        # Store fitted label encoder for transform method
        self.binary_encoders.update({column_id: encoder})

    def _apply_encoder(self, column: pd.Series, column_id: int, gap_ids: pd.Series) -> np.ndarray:
        """ Apply already fitted encoders """
        encoder = self.binary_encoders[column_id]
        # Extend encoder classes if the column contains categories not previously encountered
        encoder.classes_ = np.unique(np.concatenate((encoder.classes_, column)))

        converted = encoder.transform(column)
        if len(gap_ids) > 0:
            # Column has nans in its structure - after conversion replace it
            converted = converted.astype(float)
            converted[gap_ids] = np.nan

        return converted


def replace_nans_with_fedot_nans(column: pd.Series, has_nan: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Add new category - 'fedot_nan' after converting it will be replaced by nans
    column[has_nan] = FEDOT_STR_NAN
    return column, has_nan
