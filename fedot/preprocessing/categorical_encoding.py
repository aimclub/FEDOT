from fedot.core.backend.backend import Backend
from fedot.core.data.complex_types import ArrayType, IndexType
from fedot.core.data.prepared_data import PreparedData

from fedot.preprocessing.preprocessor_types import PreprocessingStep


"""
How to add a new categorical encoder
--------------------------------------
1. Implement the encoder class in this module (e.g. `class MyEncoder:`).
   It must use `backend.xp` for array operations so it works on both CPU (NumPy) and
   GPU (CuPy).
2. Implement the methods expected by the preprocessing pipeline:
   `fit(data, categorical_idx)`, `transform(data)`, and (optionally) `fit_transform(data, categorical_idx)`.
3. Add a new enum value to `EncodingStrategyEnum` in `fedot/preprocessing/preprocessor_types.py`.
   The value must be a unique string, for example `my_strategy = "my_strategy"`.
4. Add a mapping entry in `fedot/core/repository/preprocessor_mapping.py`:
   extend `ENCODER_MAPPING` with `EncodingStrategyEnum.my_strategy: MyEncoder`.
"""


class LabelEncoder:
    """
    Label-encode categorical feature columns.

    During :meth:`fit`, the encoder learns unique categories for each categorical
    column. During :meth:`transform`, it converts category values into integer
    IDs. Missing values are preserved as `NaN`.

    The encoded output shape is `(n_samples, n_categorical_columns)`.
    """

    def __init__(self):
        self.categories_ = {}
        self.categorical_idx_ = None

    def fit(self, features: ArrayType, categorical_idx: IndexType):
        """
        Learn category sets for each categorical column.

        Args:
            data (ArrayType): Feature matrix of shape `(n_samples, n_features)`.
            categorical_idx (IndexType): Indices of categorical columns to encode.

        Returns:
            LabelEncoder: Fitted encoder instance.
        """
        xp = Backend().xp

        self.categorical_idx_ = list(categorical_idx)
        self.categories_ = {}

        for idx in self.categorical_idx_:
            column = features[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

        return self

    def transform(self, features: ArrayType):
        """
        Transform categorical values to label-encoded numeric IDs.

        Args:
            data (ArrayType): Feature matrix of shape `(n_samples, n_features)`.

        Returns:
            ArrayType: Encoded array of shape `(n_samples, n_categorical_columns)`.
        """
        xp = Backend().xp

        n_rows = features.shape[0]
        n_cat = len(self.categorical_idx_)

        encoded = xp.full((n_rows, n_cat), xp.nan, dtype=float)

        for j, idx in enumerate(self.categorical_idx_):
            column = features[:, idx]
            categories = self.categories_[idx]

            nan_mask = column != column

            if categories.size > 0:
                matches = column.reshape(-1, 1) == categories.reshape(1, -1)

                matched_rows = matches.any(axis=1)

                encoded[matched_rows, j] = matches.argmax(axis=1)[matched_rows].astype(float)

            encoded[nan_mask, j] = xp.nan
        
        features[:, self.categorical_idx_] = encoded

        return features

    def fit_transform(self, data: PreparedData, step: PreprocessingStep):
        """
        Fit the encoder and immediately transform the data.

        Args:
            data (ArrayType): Feature matrix.
            categorical_idx (IndexType): Indices of categorical columns.

        Returns:
            ArrayType: Encoded output.
        """
        data.features = self.fit(data.features, step.features_idx).transform(data.features)
        return data


class OneHotEncoder:
    """
    One-hot encode categorical feature columns.

    During :meth:`fit`, the encoder learns unique categories for each categorical
    column and precomputes output slices. During :meth:`transform`, it produces
    a concatenated one-hot representation. Missing values are preserved as `NaN`.

    The encoded output shape is `(n_samples, n_output_features_)`.
    """

    def __init__(self):
        self.categories_ = {}
        self.categorical_idx_ = None
        self.feature_slices_ = None
        self.n_output_features_ = None
        self.new_cols_dict = {}

    def fit(self, features: ArrayType, categorical_idx: IndexType):
        """
        Learn categories and output slices for each categorical column.

        Args:
            data (ArrayType): Feature matrix of shape `(n_samples, n_features)`.
            categorical_idx (IndexType): Indices of categorical columns to encode.

        Returns:
            OneHotEncoder: Fitted encoder instance.
        """

        xp = Backend().xp

        self.categorical_idx_ = list(categorical_idx)
        self.categories_ = {}
        self.feature_slices_ = {}

        start = 0
        for idx in self.categorical_idx_:
            column = features[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

            self.new_cols_dict[idx] = int(categories.size)

            end = start + int(categories.size)
            self.feature_slices_[idx] = slice(start, end)
            start = end

        self.n_output_features_ = start
        return self

    def transform(self, features: ArrayType):
        """
        Transform categorical values to one-hot encoded features.

        Args:
            data (ArrayType): Feature matrix of shape `(n_samples, n_features)`.

        Returns:
            ArrayType: One-hot encoded array of shape
                `(n_samples, self.n_output_features_)`.
        """
        xp = Backend().xp

        n_rows = features.shape[0]
        encoded = xp.full((n_rows, self.n_output_features_), xp.nan, dtype=float)

        for idx in self.categorical_idx_:
            column = features[:, idx]
            categories = self.categories_[idx]
            feature_slice = self.feature_slices_[idx]

            nan_mask = column != column

            if categories.size == 0:
                continue

            block = (column.reshape(-1, 1) == categories.reshape(1, -1)).astype(float)
            block[nan_mask, :] = xp.nan

            encoded[:, feature_slice] = block
        
        features = xp.delete(features, self.categorical_idx_, axis=1)
        features = xp.hstack((features, encoded))

        return features

    def fit_transform(self, data: PreparedData, step: PreprocessingStep):
        """
        Fit the encoder and immediately transform the data.

        Args:
            data (ArrayType): Feature matrix.
            categorical_idx (IndexType): Indices of categorical columns.

        Returns:
            ArrayType: One-hot encoded output.
        """
        data.features = self.fit(data.features, step.features_idx).transform(data.features)
        data.new_cols_dict = self.new_cols_dict
        return data
