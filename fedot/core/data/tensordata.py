from dataclasses import dataclass, field
from typing import Optional, Dict, Any, ClassVar, List, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Callable, Type, TypeAlias

from fedot.core.data.array_utilities import atleast_2d
from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot.core.data.data_tools import (
    is_existed_csv_path, replace_missing_with_nan, encode_categorical_features, 
    _drop_rows_with_nan_in_target, encode_target, get_text_column_indices,
    encode_text_columns_np, get_target_and_features, get_target_idx)
from fedot.core.data.data import (get_df_from_csv,
                                  autodetect_data_type)



POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']
PathType = Union[os.PathLike, str]

class LazyTensor:
    def __init__(self, create_fn):
        self._create_fn = create_fn
        self._data = None

    def get(self) -> "TensorData":
        if self._data is None:
            self._data = self._create_fn()
        return self._data

    def to(self, device: str):
        data = self.get()
        return data.to(device)

    def __repr__(self):
        return f"LazyTensor(initialized={self._data is not None})"


TensorLike: TypeAlias = Optional[Union[torch.Tensor, np.ndarray, LazyTensor]]


@dataclass
class TensorData:
    task: Any
    data_type: Any

    idx: Optional[np.ndarray] = None

    features: TensorLike = None
    target: TensorLike = None
    predict: TensorLike = None

    target_idx: Optional[Union[int, np.ndarray]] = None
    categorical_features: TensorLike = None
    categorical_idx: Optional[np.ndarray] = None
    numerical_idx: Optional[np.ndarray] = None
    encoded_idx: Optional[np.ndarray] = None
    features_names: Optional[np.ndarray] = None

    embedder_name: Optional[str] = None
    embedder_batch_size: int = 16

    device: Optional[str] = None

    _creators: ClassVar[List[Tuple[Callable, Callable]]] = []

    # TODO: add ability to take kwargs from config 
    dataloader_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False
    })

    def __post_init__(self):
        """
        preprocessing for torch export"""
        if isinstance(self.task, str):
            self.task = Task(TaskTypesEnum(self.task))

        self.features = replace_missing_with_nan(self.features)

        # TODO: is it available to have target=None?
        if self.target is not None:
            self.target = atleast_2d(self.target)
            self.target = replace_missing_with_nan(self.target)
            self.target = encode_target(self.target)
        else:
            self.features, self.target = get_target_and_features(self.features,
                                                                 self.target_idx)

        if self.target is not None:
            self.features, self.target = _drop_rows_with_nan_in_target(self.features,
                                                      self.target)

        # check for text features
        text_idx = get_text_column_indices(self.features)

        # encode categorical features
        if self.categorical_idx is not None:
            if self.categorical_idx.size != 0 and isinstance(self.categorical_idx[0], str) and self.features_names is None:
                raise ValueError(
                    'Impossible to specify categorical features by name when the features_names are not specified'
                )

            if self.categorical_idx.size != 0 and isinstance(self.categorical_idx[0], str):
                self.categorical_idx = np.array(
                    [idx for idx, column in enumerate(self.features_names) if column in set(self.categorical_idx)]
                )

            # if categorical features are recognized like text
            if text_idx is not None and not set(self.categorical_idx).isdisjoint(set(text_idx)):
                text_idx = np.setdiff1d(text_idx, self.categorical_idx)

        self.features, self.categorical_idx = encode_categorical_features(
                    self.features, self.categorical_idx, text_idx
            )
        if self.categorical_idx:
            self.categorical_features = torch.tensor(
                self.features[:, self.categorical_idx]
            )
        else:
            self.categorical_features = None
        
        # get embeddings for text features
        if text_idx is not None:
            text_columns = self.features[:, text_idx]
            text_tensors = encode_text_columns_np(text_columns, self.embedder_name, self.embedder_batch_size)
            self.features = np.delete(self.features, text_idx, axis=1)

        # convert to tensor
        if self.features.shape[1] != 0:
            self.features = torch.tensor(self.features)
            if text_idx is not None:
                self.features = torch.cat((self.features, text_tensors), dim=1)
        else:
            self.features = text_tensors
            self.data_type = DataTypesEnum.text

        self.idx = np.arange(self.features.shape[0])

        if self.task.task_type is TaskTypesEnum.ts_forecasting:
            self.target = self.features.copy()
        elif self.target is not None:
            self.target = torch.tensor(self.target)


    @classmethod
    def _resolve_creator(cls, source_data):
        for predicate, creator in cls._creators:
            if predicate(source_data):
                return creator
        raise ValueError(f"No creator registered for input: {source_data}")
    
    @classmethod
    def register_creator(cls, predicate: Callable[[Any], bool]):
        def decorator(func):
            cls._creators.append((predicate, func))
            return func
        return decorator

    @classmethod
    def create(cls, source_data, **kwargs):
        try:
            creator = cls._resolve_creator(source_data)
            return creator(source_data, **kwargs)
        except Exception as e:
            raise ValueError(f"Unsupported data type: {source_data}")

    @classmethod
    def create_lazy(cls, source_data, **kwargs):
        creator = cls._resolve_creator(source_data)

        def _create():
            return creator(source_data, **kwargs)

        return LazyTensor(_create)

    def to(self, device: str):
        self.device = device

        if isinstance(self.features, LazyTensor):
            self.features = self.features.get().to(device)
        elif isinstance(self.features, torch.Tensor):
            self.features = self.features.to(device)

        if self.target is not None:
            self.target = self.target.to(device)

        return self
    
    def save_predict(self, path_to_save: PathType) -> PathType:
        prediction = self.predict.tolist()
        prediction_df = pd.DataFrame({'Index': self.idx, 'Prediction': prediction})
        try:
            prediction_df.to_csv(path_to_save, index=False)
        except (FileNotFoundError, PermissionError, OSError):
            path_to_save = './predictions.csv'
            prediction_df.to_csv(path_to_save, index=False)

        return Path(path_to_save).resolve()
    
    def to_csv(self, path_to_save: PathType) -> PathType:
        features = self.features.tolist()
        features_df = pd.DataFrame({'Index': self.idx, 'Features': features})
        if self.target is not None:
            features_df['target'] = self.target
        try:
            features_df.to_csv(path_to_save, index=False)
        except (FileNotFoundError, PermissionError, OSError):
            path_to_save = './features.csv'
            features_df.to_csv(path_to_save, index=False)
        return Path(path_to_save).resolve()
    
    @property
    def memory_usage(self):
        """
        Returns the memory usage of the features in bytes.

        For torch.Tensor, this is the element size multiplied by the number of elements.
        For numpy arrays, this is the sum of the number of bytes for each feature.

        Returns:
            int: The memory usage in bytes.
        """
        if isinstance(self.features, torch.Tensor):
            return self.features.element_size() * self.features.nelement()
        else:
            # for numpy
            return sum([feature.nbytes for feature in self.features.T])


@TensorData.register_creator(
    lambda x: isinstance(x, np.ndarray)
)
def from_numpy(features: np.ndarray,
               target: Optional[np.ndarray] = None,
               task: Task = Task(TaskTypesEnum.classification),
               data_type: Optional[DataTypesEnum] = None,
               features_names: np.ndarray[str] = None,
               target_idx: Optional[Union[int, np.ndarray]] = None,
               categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None,
               embedder_name: str = None,
               embedder_batch_size=16,
               device: str = 'cpu',
               dataloader_kwargs = None,) -> TensorData:

    if data_type is None:
        data_type = autodetect_data_type(task)

    if isinstance(target, int) and target < features.shape[1]:
        target = features[:, target]
        features = np.delete(features, target, axis=1)
    
    data = TensorData(
        features=features,
        target=target,
        features_names=features_names,
        target_idx=target_idx,
        categorical_idx=categorical_idx,
        task=task,
        data_type=data_type,
        embedder_name=embedder_name,
        embedder_batch_size=embedder_batch_size,
        device=device,
        dataloader_kwargs=dataloader_kwargs
    )

    return data


@TensorData.register_creator(
    lambda x: isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)
)
def from_pandas(
    features: Union[pd.DataFrame, pd.Series],
    target: Optional[Union[pd.DataFrame, pd.Series]] = None,
    task: Union[Task, str] = 'classification',
    target_columns: Optional[Union[List[str], List[int], str, int]] = None,
    categorical_idx=None,
    data_type: DataTypesEnum = DataTypesEnum.table,
    embedder_name: str = None,
    embedder_batch_size=16,
    device: str = 'cpu',
    dataloader_kwargs = None,
) -> TensorData:
    
    features_names = features.columns.to_numpy()
    
    features = features.values

    if target is not None:
        target = target.values
    
    if target_columns is not None:
        target_idx = get_target_idx(target_columns, features_names)
    else:
        target_idx = None

    return TensorData(
        features=features,
        target=target,
        task=task,
        target_idx=target_idx,
        categorical_idx=categorical_idx,
        data_type=data_type,
        features_names=features_names,
        embedder_name=embedder_name,
        embedder_batch_size=embedder_batch_size,
        device=device,
        dataloader_kwargs=dataloader_kwargs
    )


@TensorData.register_creator(
    lambda x: isinstance(x, str) and x.endswith(".csv")
)
def from_csv(
    file_path: str,
    delimiter: str = ',',
    max_rows: Optional[int] = None,
    task: Union[Task, str] = 'classification',
    data_type: DataTypesEnum = DataTypesEnum.table,
    columns_to_drop: list = None,
    target_columns: str = '',
    categorical_idx: Optional[Union[list[int, str], np.ndarray[int, str]]] = None,
    index_col = None,
    possible_idx_keywords = None,
    embedder_name = None,
    embedder_batch_size=16,
    device: str = 'cpu',
    dataloader_kwargs = None
) -> TensorData:

    if not is_existed_csv_path(file_path):
        raise ValueError(f'File {file_path} does not exist')

    possible_idx_keywords = (
        possible_idx_keywords or POSSIBLE_TABULAR_IDX_KEYWORDS
    )

    features = get_df_from_csv(
        file_path=file_path,
        delimiter=delimiter,
        index_col=index_col,
        possible_idx_keywords=possible_idx_keywords,
        columns_to_drop=columns_to_drop,
        nrows=max_rows
    )
    
    features_names = features.columns.to_numpy()
    features = features.values

    if target_columns is not None:
        target_idx = get_target_idx(target_columns, features_names)
    else:
        target_idx = None
    
    return TensorData(
        features=features,
        task=task,
        target_idx=target_idx,
        data_type=data_type,
        categorical_idx=categorical_idx,
        features_names=features_names,
        embedder_name=embedder_name,
        embedder_batch_size=embedder_batch_size,
        device=device,
        dataloader_kwargs=dataloader_kwargs
    )
