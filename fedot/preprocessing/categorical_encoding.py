from golem.utilities.data_structures import ComparableEnum as Enum
from dataclasses import dataclass
from fedot.core.backend.backend import Backend
from typing import Optional, Any
from fedot.core.data.tensordata import IndexType


class EncodingStrategyEnum(Enum):
    label = "label"
    ohe = "ohe"


@dataclass
class CategoricalEncodingDecision:
    categorical_columns: IndexType
    strategy: Optional[EncodingStrategyEnum] = None
    encoder: Any = None


class LabelEncoder:

    categories_ = {}
    categorical_idx_ = None

    def fit(self, data, categorical_idx):
        xp = Backend.xp

        self.categorical_idx_ = list(categorical_idx)
        self.categories_ = {}

        for idx in self.categorical_idx_:
            column = data[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

        return self

    def transform(self, data):
        xp = Backend.xp

        n_rows = data.shape[0]
        n_cat = len(self.categorical_idx_)

        encoded = xp.full((n_rows, n_cat), xp.nan, dtype=float)

        for j, idx in enumerate(self.categorical_idx_):
            column = data[:, idx]
            categories = self.categories_[idx]

            nan_mask = column != column

            if categories.size > 0:
                matches = column.reshape(-1, 1) == categories.reshape(1, -1)

                matched_rows = matches.any(axis=1)

                encoded[matched_rows, j] = matches.argmax(axis=1)[matched_rows].astype(float)

            encoded[nan_mask, j] = xp.nan

        return encoded

    def fit_transform(self, data, categorical_idx):
        return self.fit(data, categorical_idx).transform(data)


class OneHotEncoder:

    categories_ = {}
    categorical_idx_ = None
    feature_slices_ = None
    n_output_features_ = None

    def fit(self, data, categorical_idx):
        xp = Backend.xp

        self.categorical_idx_ = list(categorical_idx)
        self.categories_ = {}
        self.feature_slices_ = {}

        start = 0
        for idx in self.categorical_idx_:
            column = data[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

            end = start + int(categories.size)
            self.feature_slices_[idx] = slice(start, end)
            start = end

        self.n_output_features_ = start
        return self

    def transform(self, data):
        xp = Backend.xp

        n_rows = data.shape[0]
        encoded = xp.full((n_rows, self.n_output_features_), xp.nan, dtype=float)

        for idx in self.categorical_idx_:
            column = data[:, idx]
            categories = self.categories_[idx]
            feature_slice = self.feature_slices_[idx]

            nan_mask = column != column

            if categories.size == 0:
                continue

            block = (column.reshape(-1, 1) == categories.reshape(1, -1)).astype(float)
            block[nan_mask, :] = xp.nan

            encoded[:, feature_slice] = block

        return encoded

    def fit_transform(self, data, categorical_idx):
        return self.fit(data, categorical_idx).transform(data)




    # target = xp.asarray(target)

    # if target.dtype.kind in {"U", "S"}:
    #     target_flat = target.flatten()
    #     _, codes = xp.unique(target_flat, return_inverse=True)
    #     codes = codes.astype(xp.int64)
    #     return codes.reshape(-1, 1)

    # if target.dtype == object:
    #     if isinstance(target.flat[0], str):
    #         target_flat = target.flatten()
    #         _, codes = xp.unique(target_flat, return_inverse=True)
    #         return codes.astype(xp.int64).reshape(-1, 1)

    #     try:
    #         return target.astype(xp.int64)
    #     except Exception:
    #         return target.astype(xp.float32)

    # if target.dtype.kind in {"i", "u"}:
    #     return target.astype(xp.int64)

    # if target.dtype.kind == "f":
    #     return target.astype(xp.float32)

    # return target
