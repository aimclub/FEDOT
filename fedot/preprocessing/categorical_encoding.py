from golem.utilities.data_structures import ComparableEnum as Enum
from dataclasses import dataclass
from fedot.core.backend.backend import Backend
from typing import Optional, Any, Dict, List
from fedot.core.data.tensordata import IndexType
from fedot.core.data.data_tools import get_idx_from_features_names, convert_idx_to_array


xp = Backend.xp

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


def force_categorical_determination(table):
    """Find string columns using a unified approach for CPU/GPU backends."""
    pd_backend = Backend.pd

    categorical_ids = []

    for column_id, column in enumerate(table.T):
        series = pd_backend.Series(column)
        if str(series.dtype) in ("object", "string"):
            categorical_ids.append(column_id)

    if len(categorical_ids) == 0:
        return None

    categorical_ids = convert_idx_to_array(categorical_ids)
    return categorical_ids


def process_user_stratedy_encoding(strategy: Dict, features_names):
    strategy_list = []

    for strategy_name, idx in strategy.items():
        idx = convert_idx_to_array(idx)
        idx = get_idx_from_features_names(idx, features_names)
        strategy_list.append(
            CategoricalEncodingDecision(idx, EncodingStrategyEnum(strategy_name))
        )

    return strategy_list


def choose_categorical_encoding(
    data,
    categorical_idx=None,
    user_stategy=None,
    features_names=None,
) -> List[CategoricalEncodingDecision]:
    xp = Backend.xp

    if isinstance(user_stategy, Dict):
        decisions = process_user_stratedy_encoding(user_stategy, features_names)
        categorical_idx = xp.array(
            [col for dec in decisions for col in dec.categorical_columns]
        )
        non_categorical_idx = xp.setdiff1d(xp.arange(data.shape[1]), categorical_idx)
        return decisions, non_categorical_idx

    if categorical_idx is not None:
        categorical_idx = get_idx_from_features_names(categorical_idx, features_names)
    else:
        categorical_idx = force_categorical_determination(data)

    if categorical_idx is None:
        return None, xp.arange(data.shape[1])

    non_categorical_idx = xp.setdiff1d(xp.arange(data.shape[1]), categorical_idx)
    strategy = (
        EncodingStrategyEnum(user_stategy)
        if user_stategy is not None
        else EncodingStrategyEnum.label
    )
    return [CategoricalEncodingDecision(categorical_idx, strategy)], non_categorical_idx


def apply_categorical_encoding(data, decision):
    xp = Backend.xp

    if decision.categorical_columns is None:
        return data, decision

    if decision.strategy == EncodingStrategyEnum.label:
        decision.encoder = LabelEncoder(xp=xp)
        features = decision.encoder.fit_transform(data, decision.categorical_columns)

    elif decision.strategy == EncodingStrategyEnum.ohe:
        decision.encoder = OneHotEncoder(xp=xp)
        features = decision.encoder.fit_transform(data, decision.categorical_columns)

    return features, decision


def encode_categorical_features(data, decisions, non_categorical_idx):
    xp = Backend.xp

    if decisions is None:
        return data, None

    result_data = data[:, non_categorical_idx].copy()
    new_decisions = []

    for dec in decisions:
        encoded_features, dec_new = apply_categorical_encoding(data, dec)
        result_data = xp.hstack((result_data, encoded_features))
        new_decisions.append(dec_new)

    return result_data, new_decisions
