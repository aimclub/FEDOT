from dataclasses import dataclass, field
from typing import Optional, Dict, Any, ClassVar, List

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Callable, Type

from fedot.core.data.array_utilities import atleast_2d
from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot.core.data.data_tools import is_existed_csv_path, replace_missing_with_nan, encode_categorical_features, _drop_rows_with_nan_in_target, encode_target
from fedot.core.data.data import get_df_from_csv, process_target_and_features, autodetect_data_type


POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']
PathType = Union[os.PathLike, str]

class LazyTensor:
    def __init__(self, create_fn):
        self._create_fn = create_fn
        self._tensor = None

    def get(self):
        if self._tensor is None:
            self._tensor = self._create_fn()
        return self._tensor

    def to(self, device):
        return self.get().to(device)

    def __repr__(self):
        return f"LazyTensor(initialized={self._tensor is not None})"


@dataclass
class TensorData:
    idx: Union[torch.Tensor, np.ndarray]
    task: Any
    data_type: Any

    features: Union[torch.Tensor, np.ndarray, LazyTensor]
    target: Optional[torch.Tensor] = None

    categorical_features: Optional[torch.Tensor] = None
    categorical_idx: Optional[torch.Tensor] = None
    numerical_idx: Optional[torch.Tensor] = None
    encoded_idx: Optional[torch.Tensor] = None
    features_names: Optional[np.ndarray] = None

    device: Optional[str] = None
    # TODO: backend_name, I think it's not needed
    # backend_name: str = "torch"

    # TODO: add ability to take kwargs from config 
    dataloader_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False
    })

    # registry: type -> creator
    _creators: ClassVar[Dict[Type, Callable]] = {}

    @classmethod
    def register_creator(cls, data_type: Type):
        def decorator(func):
            cls._creators[data_type] = func
            return func
        return decorator

    @classmethod
    def _resolve_creator(cls, source_data):
        data_type = type(source_data)
        if data_type in cls._creators:
            return cls._creators[data_type]

        for registered_type, creator in cls._creators.items():
            if isinstance(source_data, registered_type):
                return creator

        raise ValueError(f"No creator registered for type {data_type}")

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
            return creator(source_data)

        lazy_features = LazyTensor(_create)
        return cls(features=lazy_features, **kwargs)

    def to(self, device: str):
        self.device = device

        if isinstance(self.features, LazyTensor):
            self.features = self.features.get().to(device)
        elif isinstance(self.features, torch.Tensor):
            self.features = self.features.to(device)

        if self.target is not None:
            self.target = self.target.to(device)

        return self


@TensorData.register_creator(np.ndarray)
def from_numpy(features: np.ndarray,
               target: Optional[np.ndarray] = None,
                idx: Optional[np.ndarray] = None,
                task: Task = Task(TaskTypesEnum.classification),
                data_type: Optional[DataTypesEnum] = None,
                features_names: np.ndarray[str] = None,
                categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None) -> TensorData:
    if idx is None:
        idx = np.arange(len(features))

    if data_type is None:
        data_type = autodetect_data_type(task)
    
    if categorical_idx is not None:
        if categorical_idx.size != 0 and isinstance(categorical_idx[0], str) and features_names is None:
            raise ValueError(
                'Impossible to specify categorical features by name when the features_names are not specified'
            )

        if categorical_idx.size != 0 and isinstance(categorical_idx[0], str):
            categorical_idx = np.array(
                [idx for idx, column in enumerate(features_names) if column in set(categorical_idx)]
            )
    else:
        features, categorical_idx = encode_categorical_features(features, categorical_idx)

    if categorical_idx is not None:
        categorical_features = features[:, categorical_idx]
    else:
        categorical_features = None
    
    features = torch.from_numpy(features).float()

    if task.task_type is TaskTypesEnum.ts_forecasting:
        target = features.copy()
    elif isinstance(target, int):
        if target < features.shape[1]:
            target = features[:, target]
            features = np.delete(features, target, axis=1)
        else:
            target = torch.tensor([])
    elif target is not None:
        target = encode_target(target)
        target = torch.from_numpy(target).float()
    else:
        target = torch.tensor([])
    
    data = TensorData(
        idx=idx,
        features=features,
        target=target,
        features_names=features_names,
        categorical_idx=categorical_idx,
        categorical_features=categorical_features,
        task=task,
        data_type=data_type
    )

    return data


#  TODO: add case with time series
@TensorData.register_creator(str)
def from_csv(file_path: str,
             delimiter: str = ',',
        task: Union[Task, str] = 'classification',
        data_type: DataTypesEnum = DataTypesEnum.table,
        columns_to_drop: Optional[List[Union[str, int]]] = None,
        target_columns: Union[str, List[Union[str, int]], None] = '',
        categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None,
        index_col: Optional[Union[str, int]] = None,
        possible_idx_keywords: Optional[List[str]] = None) -> TensorData:
    """
    Docstring for from_csv
    
    """
    if is_existed_csv_path(file_path):
        possible_idx_keywords = (possible_idx_keywords or 
                                 POSSIBLE_TABULAR_IDX_KEYWORDS)
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))

        # get as pandas DataFrame
        df = get_df_from_csv(file_path, 
                             delimiter, 
                             index_col, 
                             possible_idx_keywords, 
                             columns_to_drop=columns_to_drop)

        idx = df.index.to_numpy()
        if target_columns:
            features_names = df.drop(target_columns, axis=1).columns.to_numpy()

        else:
            features_names = df.columns.to_numpy()
        
        
        df = replace_missing_with_nan(df)
        
        features, target = process_target_and_features(df, target_columns)# split features and target
        features, target = _drop_rows_with_nan_in_target(features, target)
        features, categorical_idx = encode_categorical_features(features, categorical_idx)
        target = encode_target(target)
        if categorical_idx:
            categorical_features = torch.tensor(df[:, categorical_idx].values)
        else:
            categorical_features = None


        idx = np.arange(len(features))
        features = torch.tensor(features)
        target = torch.tensor(target)
 
        data = TensorData(
            idx=idx,
            features=features,
            target=target,
            task=task,
            data_type=data_type,
            features_names=features_names,
            categorical_idx=categorical_idx,
            categorical_features=categorical_features
        )
        return data

    else:
        raise ValueError(f'File {file_path} does not exist')


# TODO: add these methods
@TensorData.register_creator(torch.Tensor)
def from_torch(tensor: torch.Tensor):
    return tensor.clone().float()


@TensorData.register_creator(pd.DataFrame)
def from_pandas(df: pd.DataFrame):
    return torch.tensor(df.values, dtype=torch.float32)


@TensorData.register_creator(tuple)
def from_tuple(df: pd.DataFrame):
    return torch.tensor(df.values, dtype=torch.float32)
