import os
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.array_utilities import atleast_2d

import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer


def replace_missing_with_nan(arr: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Replace all missing or non-numeric values in a numpy array with np.nan
    and return a numeric array compatible with torch.tensor.

    Args:
        arr: Input numpy array (any dtype, including object).
        dtype: Output numeric dtype (np.float32 or np.float64).

    Returns:
        Numeric numpy array with np.nan, safe for torch.tensor.
    """
    arr_obj = np.asarray(arr, dtype=object)

    out = np.empty(arr_obj.shape, dtype=dtype)

    it = np.nditer(arr_obj, flags=["multi_index", "refs_ok"])
    for x in it:
        val = x.item()
        if val is None or val is pd.NA:
            out[it.multi_index] = np.nan
        else:
            try:
                out[it.multi_index] = float(val)
            except (TypeError, ValueError):
                out[it.multi_index] = np.nan

    return out


# TODO: add dependecies to task if neccessary (e.g. case with clustering)
def get_target_and_features(features: np.ndarray,
                            target_idx: Optional[Union[int, np.ndarray]]
                            ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Function for getting target and features from numpy array"""

    if target_idx is None:
        target = features[:, -1]
        target_idx = -1

    target = atleast_2d(features[:, target_idx])
    features = np.delete(features, target_idx, axis=1)

    return features, target


def _drop_rows_with_nan_in_target(features: np.ndarray, 
                                  target: np.ndarray) -> pd.DataFrame:
    """
    Drops rows with nans in target column

    Args:
        data: to be modified

    Returns:
        modified ``data``
    """
    if target is None:
        return features, target

    bool_target = np.array(pd.isna(target))
    number_nans_per_rows = bool_target.sum(axis=1)
    non_nan_row_ids = np.ravel(np.argwhere(number_nans_per_rows == 0))

    if len(non_nan_row_ids) == 0:
        raise ValueError('Data contains too much nans in the target column(s)')

    features = features[non_nan_row_ids, :]
    target = target[non_nan_row_ids, :]

    return features, target


def encode_categorical_features(array: np.ndarray, 
                               categorical_idx: Optional[np.ndarray] = None,
                               text_idx: Union[np.ndarray, List[int], None] = None
                            ) -> np.ndarray:
    if categorical_idx is None:
        categorical_idx, _ = find_categorical_columns(array)
    
    if text_idx is not None:
        categorical_idx = np.setdiff1d(categorical_idx, text_idx).tolist()

    if len(categorical_idx) == 0:
        return array, None

    # Encode each categorical column
    for idx in categorical_idx:
        column = array[:, idx]
        _, codes = np.unique(column, return_inverse=True)
        codes = codes.astype(float)
        codes[codes == -1] = np.nan
        array[:, idx] = codes

    return array, categorical_idx


def encode_target(target: np.ndarray) -> np.ndarray:
    """
    Encodes categorical target values using factorization.

    Args:
        target: Input 2D numpy array of shape [N, 1] with categorical target values.

    Returns:
        Encoded target array of the same shape [N, 1].
    """
    if target is None or len(target) == 0 or not isinstance(target[0, 0], str):
        return target
    
    target_flat = target.flatten()
    _, codes = np.unique(target_flat, return_inverse=True)
    codes = codes.astype(float)
    codes[codes == -1] = np.nan
    encoded_target = codes.reshape(-1, 1)
    return encoded_target


def is_existed_csv_path(path: str) -> bool:
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return True
    return False


def get_text_column_indices(
    arr: np.ndarray,
    min_avg_length: int = 30,
    sample_size: int = 1000,
) -> list[int]:
    """
    Returns indices of columns with text data.
    """
    text_columns = []
    n_samples = arr.shape[0]

    for col_idx in range(arr.shape[1]):
        col_data = arr[:, col_idx]
        non_nan_mask = ~pd.isna(col_data)
        non_nan_data = col_data[non_nan_mask].astype(str)

        if len(non_nan_data) == 0:
            continue

        # create sample
        if len(non_nan_data) > sample_size:
            sample_indices = np.random.choice(
                len(non_nan_data),
                size=sample_size,
                replace=False
            )
            sample = non_nan_data[sample_indices]
        else:
            sample = non_nan_data

        avg_length = np.mean([len(text) for text in sample])

        if avg_length >= min_avg_length:
            text_columns.append(col_idx)

    return text_columns if len(text_columns) > 0 else None


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
        return embeddings.cpu()


def encode_text_columns_np(
    X: np.ndarray,
    model_name: str = None,
    batch_size: int = 32,
    device: torch.device | None = None,
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
    return embeddings_all


def get_target_idx(target_columns: Optional[Union[List[str], List[int], str, int]],
                   features_names: np.ndarray) -> Optional[Union[np.ndarray, int]]:
    if isinstance(target_columns, int) or \
        (isinstance(target_columns, list) and isinstance(target_columns[0], int)):
        return target_columns
    elif isinstance(target_columns, list):
        mask = np.isin(features_names, target_columns)
        return np.where(mask)[0]
    else:
        return np.where(features_names == target_columns)[0]
