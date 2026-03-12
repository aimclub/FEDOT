import os
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple
from fedot.core.utils import fedot_project_root

from scipy.io.arff import loadarff
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer

import logging

logger = logging.getLogger(__name__)

PathType = Union[os.PathLike, str]

PROJECT_PATH = fedot_project_root()


def is_existed_csv_path(path: str) -> bool:
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return True
    return False


def get_device_from_str(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class TextToEmbedding:
    def __init__(self, model_name: str, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = SentenceTransformer(model_name, device=str(device))

    def __call__(self, sentences: list[str]) -> Tensor:
        
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=False,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings

def encode_text_columns_np(
    X: np.ndarray,
    model_name: str = None,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    if model_name is None:
        model_name = 'all-distilroberta-v1'
    text_embedder = TextToEmbedding(model_name, device=device)

    n_samples = X.shape[0]
    num_cols = X.shape[1]
    embedding_parts = []

    for col_idx in range(num_cols):
        texts = X[:, col_idx].astype(str)
        all_embeddings = []
        for i in range(0, n_samples, batch_size):
            batch = texts[i : i + batch_size].tolist()
            embeddings = text_embedder(batch)
            all_embeddings.append(embeddings)

        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        embedding_parts.append(embeddings_tensor)

    try:
        embeddings_all = torch.cat(embedding_parts, dim=1)
    except Exception as e:
        raise ValueError(f"Failed to get embeddings: {e}")
    
    embeddings_all = embeddings_all.to(torch.float32)
    return embeddings_all


def convert_bytes(x):
    # Conversion of target values to float or str
    try:
        x = np.char.decode(x, encoding='utf-8')
    except:
        pass
    try:
        x = x.astype('float')
    except ValueError:
        x = x.astype(str)
    return x


class BackendDefiner:
    """
    Unified backend for CPU/GPU array + dataframe operations.
    """

    def __init__(self, name: str = "cpu"):
        self.name = name



        if self.name == "gpu" and torch.cuda.is_available():
            import cupy as xp
            import cudf as pd

            self.device = torch.device("cuda")

            self.xp = xp
            self.pd = pd
            self.name = "gpu"

        else:
            import numpy as xp
            import pandas as pd

            self.device = torch.device("cpu")

            self.xp = xp
            self.pd = pd

    def get_df_from_csv(self, file_path: PathType, delimiter: str, index_col: Optional[Union[str, int]] = None,
                        possible_idx_keywords: Optional[List[str]] = None, *,
                        columns_to_drop: Optional[List[Union[str, int]]] = None,
                        columns_to_use: Optional[List[Union[str, int]]] = None,
                        nrows: Optional[int] = None) -> pd.DataFrame:
        def define_index_column(candidate_columns: List[str]) -> Optional[str]:
            for column_name in candidate_columns:
                if is_column_name_suitable_for_index(column_name):
                    return column_name

        def is_column_name_suitable_for_index(column_name: str) -> bool:
            return any(key in column_name.lower() for key in possible_idx_keywords)

        columns_to_drop = columns_to_drop or []
        columns_to_use = columns_to_use or []
        possible_idx_keywords = possible_idx_keywords or []

        columns = self.pd.read_csv(file_path, sep=delimiter, index_col=False, nrows=1).columns

        if columns_to_drop and columns_to_use:
            raise ValueError('Incompatible arguments are used: columns_to_drop and columns_to_use. '
                            'Only one of them can be specified simultaneously.')

        if columns_to_drop:
            columns_to_use = [col for col in columns if col not in columns_to_drop]
        elif not columns_to_use:
            columns_to_use = list(columns)
        # else:
        #     columns_to_use = []

        candidate_idx_cols = [columns_to_use[0], columns[0]]
        if index_col is None:
            defined_index = define_index_column(candidate_idx_cols)
            if defined_index is not None:
                index_col = defined_index

        if (index_col is not None) and (index_col not in columns_to_use):
            columns_to_use.append(index_col)

        return self.pd.read_csv(file_path,
                                sep=delimiter,
                                index_col=index_col,
                                usecols=columns_to_use,
                                nrows=nrows)

    def read_arff_file(self, file_path: str, target_idx: Optional[Union[int, str]] = None):
        data, meta = loadarff(file_path)

        data_array = np.asarray([data[name] for name in meta.names()])

        if target_idx is not None:
            if isinstance(target_idx, str):
                target_idx = meta.names().index(target_idx)
        else:
            if isinstance(convert_bytes(data_array[-1])[0], str):
                target_idx = -1
            elif isinstance(convert_bytes(data_array[0])[0], str):
                target_idx = 0
            else:
                target_idx = None

        if target_idx is not None:
            target = data_array[target_idx]
            features = np.delete(data_array, target_idx, axis=0)
            target = convert_bytes(target)
        else:
            target = None
            features = data_array
        
        features = convert_bytes(features)

        if self.name == "gpu":
            features = self.xp.asarray(features)
            if target is not None:
                target = self.xp.asarray(target)

        return features, target
    
    def get_values_from_df(self, df):
        try:
            features = df.values
            return features
        except:
            raise ValueError(f"Fedot preprocessing doesn't support categorical data in gpu mode")

    def replace_missing_with_nan(self, arr):
        """
        """
        if self.name == "gpu":
            return arr

        try:
            xp_arr = self.xp.asarray(arr)
            if xp_arr.dtype.kind in ("i", "u", "f", "b"):
                return xp_arr.astype(self.xp.float32, copy=False)
        except Exception:
            pass

        arr_obj = self.xp.asarray(arr, dtype=object)

        has_string = False

        def _normalize(x):
            nonlocal has_string

            if x is None or self.pd.isna(x):
                return self.xp.nan

            if isinstance(x, str):
                has_string = True
                return x

            try:
                return float(x)
            except (TypeError, ValueError):
                return self.xp.nan

        out = self.xp.vectorize(_normalize, otypes=[object])(arr_obj)

        if not has_string:
            return self.xp.asarray(out, dtype=self.xp.float32)

        return out

    def _drop_rows_with_nan_in_target(self, features, target):
        """
        Drop rows where target contains NaN in any target column.
        Works for both numpy and cupy backends.
        """
        if target is None:
            return features, target

        target = self.xp.asarray(target)

        nan_mask = self.xp.isnan(target)
        number_nans_per_rows = nan_mask.sum(axis=1)
        non_nan_row_ids = self.xp.ravel(
            self.xp.argwhere(number_nans_per_rows == 0)
        )

        if non_nan_row_ids.size == 0:
            raise ValueError('Data contains too much nans in the target column(s)')

        features = features[non_nan_row_ids, :]
        target = target[non_nan_row_ids, :]

        return features, target

    def atleast_n_dimensions(self, data, ndim):
        """
        Returns a view of the ``data` with at least ``ndim`` dimensions

        :param data: ndarray which dimensional size should be set to at least ``ndim``
        :param ndim: number of required axes to have in ``data``

        :return: ``data`` expanded from the last axis to the provided ``ndim`` size if it doesn't satisfy it
        """
        while data.ndim < ndim:
            data = self.xp.expand_dims(data, axis=-1)
        return data

    def encode_target(self, target):
        """
        Encode categorical target values and ensure numeric dtype.
        """

        if target is None or target.shape[0] == 0:
            return target

        target = self.xp.asarray(target)

        # строковые типы numpy
        if target.dtype.kind in {"U", "S"}:
            target_flat = target.flatten()
            _, codes = self.xp.unique(target_flat, return_inverse=True)
            codes = codes.astype(self.xp.int64)
            return codes.reshape(-1, 1)

        # object dtype
        if target.dtype == object:
            if isinstance(target.flat[0], str):
                target_flat = target.flatten()
                _, codes = self.xp.unique(target_flat, return_inverse=True)
                return codes.astype(self.xp.int64).reshape(-1, 1)

            try:
                return target.astype(self.xp.int64)
            except Exception:
                return target.astype(self.xp.float32)

        # integers
        if target.dtype.kind in {"i", "u"}:
            return target.astype(self.xp.int64)

        # floats
        if target.dtype.kind == "f":
            return target.astype(self.xp.float32)

        return target

    def convert_idx_to_array(self, idx):
        if isinstance(idx, self.xp.ndarray) or idx is None:
            return idx
        if isinstance(idx, int) or isinstance(idx, str):
            return self.xp.array([idx])
        else:
            return self.xp.array(idx)

    def get_idx_from_features_names(self, idx, features_names):
        if isinstance(idx[0], self.xp.int_):
            return idx
        
        if features_names is None:
            raise ValueError(
                    'Impossible to specify categorical features by name when the features_names are not specified'
                )
        
        try:   
            if isinstance(idx[0], str):
                return self.xp.array([self.xp.where(features_names == name)[0][0] for name in idx])
        except:
            raise ValueError(f"Failed to get index from features names: {idx}")

    def get_target_and_features(self, features,
                            features_names,
                            target_idx: Optional[Union[int, np.ndarray]],
                            ):
        """Function for getting target and features from numpy array"""
        if target_idx is not None:
            target_idx = self.get_idx_from_features_names(target_idx, features_names)
            target = features[:, target_idx].copy()
        else:
            target = features[:, -1].copy()
            target_idx = self.xp.array([-1])

        target = self.atleast_n_dimensions(target, 2)
        target = self.encode_target(target)
        target = self.replace_missing_with_nan(target)

        features = self.xp.delete(features, target_idx, axis=1)

        return features, target, target_idx

    # TODO: could be optimised if needed
    def get_text_column_indices(
        self,
        arr,
        text_idx=None,
        min_avg_length: int = 30,
        sample_size: int = 1000,
    ) -> Union[np.ndarray, List[int]]:
        """
        Returns indices of columns with text data.
        """
        if text_idx is not None:
            return text_idx

        text_columns = []

        for col_idx in range(arr.shape[1]):
            col_data = arr[:, col_idx]
            non_nan_mask = ~self.pd.isna(col_data)
            non_nan_data = col_data[non_nan_mask].astype(str)

            if len(non_nan_data) == 0:
                continue

            # create sample
            if len(non_nan_data) > sample_size:
                sample_indices = self.xp.random.choice(
                    len(non_nan_data),
                    size=sample_size,
                    replace=False
                )
                sample = non_nan_data[sample_indices]
            else:
                sample = non_nan_data

            avg_length = self.xp.mean([len(text) for text in sample])

            if avg_length >= min_avg_length:
                text_columns.append(col_idx)

        return self.convert_idx_to_array(text_columns) if \
                                            len(text_columns) > 0 else None

    def encode_categorical_features(self, array, 
                               categorical_idx = None,
                               text_idx = None):
        if categorical_idx is None:
            categorical_idx = self.force_categorical_determination(array)

            if text_idx is not None:
                categorical_idx = self.xp.setdiff1d(categorical_idx, text_idx)

        if categorical_idx is None:
            return array, None

        # Encode each categorical column
        for idx in categorical_idx:
            column = array[:, idx]
            _, codes = self.xp.unique(column, return_inverse=True)
            codes = codes.astype(float)
            codes[codes == -1] = self.xp.nan
            array[:, idx] = codes

        return array, categorical_idx

    def force_categorical_determination(self, table):
        """Find string columns using a unified approach for CPU/GPU backends."""
        categorical_ids = []

        for column_id, column in enumerate(table.T):
            series = self.pd.Series(column)
            if str(series.dtype) in ("object", "string"):
                categorical_ids.append(column_id)
        
        if len(categorical_ids) == 0:
            return None

        categorical_ids = self.convert_idx_to_array(categorical_ids)
        return categorical_ids

    def to_tensor(self, array, dtype = None):
        if array is None:
            return None
        array = array.astype(self.xp.float64)
        return torch.tensor(array, dtype=dtype, device=self.device)


class TSDataConverter(BackendDefiner):
    def __init__(self, xp):
        super().__init__(xp)
        self.xp = xp
    
    def split_long_array(self, features, features_names, terms_idx = None):
        if terms_idx is None:
            terms_idx = self.xp.array(range(features.shape[1] - 1))
        terms_idx = self.get_idx_from_features_names(terms_idx, features_names)[0]

        unique_labels = self.xp.unique(features[:, terms_idx])
        split_arrays = self.xp.array([features[features[:, terms_idx] == label, :terms_idx] for label in unique_labels])
        return split_arrays, terms_idx

    def check_multichannel_ts(self, features: np.ndarray):
        if features.ndim == 1:
            features = self.xp.expand_dims(features, axis=0)
            init_shape = (1, features.shape[1])
        elif features.ndim == 2:
            B, T = features.shape
            init_shape = (B, T)
        elif features.ndim == 3:
            B, C, T = features.shape
            features = features.reshape(B * C, T)
            init_shape = (B, C, T)
        elif features.ndim > 3:
            raise ValueError('Multichannel time series must not have more than 3 dimensions')
        return features, init_shape

    def process_ts_data(self, features, target=None, features_names=None, state='fit', 
                        ts_orientation: Optional[str] = None, terms_idx: int = None,
                        forecast_horizon: int = None):
        features, init_shape = self.check_multichannel_ts(features)

        if ts_orientation is None:
            ts_orientation = 'wide'
        elif ts_orientation == 'long':
            features, terms_idx = self.split_long_array(features, features_names,terms_idx)

        if state == 'fit' and forecast_horizon is not None:
            target = features[features.shape[1] - forecast_horizon:, :]
            features = features[:-forecast_horizon, :]

        return features, target, init_shape, terms_idx
