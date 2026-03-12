from dataclasses import dataclass, field
from typing import Optional, Dict, Any, ClassVar, List, Tuple

import torch
import numpy as np
import cupy as cp
import os

from pathlib import Path
import pandas as pd
import cudf
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Callable, Type, TypeAlias

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot.core.data.data_tools import (
    get_device_from_str, is_existed_csv_path, encode_text_columns_np, BackendDefiner, TSDataConverter)

from fedot.core.data.data import (autodetect_data_type)

import logging


logger = logging.getLogger(__name__)


POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']
PathType = Union[os.PathLike, str]

IndexType: TypeAlias = Optional[Union[int, str, np.ndarray, List[int], List[str]]]
TaskType: TypeAlias = Optional[Union[Task, str]]
DataType: TypeAlias = Optional[Union[DataTypesEnum, str]]

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
    """
    state: str ['fit', 'predict']
    """
    task: Union[Task, str]
    data_type: Union[DataTypesEnum, str]

    state: str = 'fit'
    idx: IndexType = None
    features: TensorLike = None
    target: TensorLike = None
    predict: TensorLike = None
    target_idx: IndexType = None
    categorical_features: TensorLike = None
    categorical_idx: IndexType = None
    text_idx: IndexType = None
    numerical_idx: IndexType = None
    encoded_idx: IndexType = None
    features_names: IndexType = None
    embedder_name: Optional[str] = None
    embedder_batch_size: int = 16
    ts_orientation: Optional[str] = None
    ts_terms_idx: Optional[Union[int, str]] = None
    ts_forecast_horizon: Optional[int] = None
    device: Optional[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend: BackendDefiner = BackendDefiner(name="cpu")

    # TODO: test dataloader
    dataloader_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False
    })

    _creators: ClassVar[List[Tuple[Callable, Callable]]] = []

    def __post_init__(self):
        self.device = get_device_from_str(self.device)

        if isinstance(self.task, str):
            self.task = Task(TaskTypesEnum(self.task))

        if isinstance(self.data_type, str):
            self.data_type = DataTypesEnum(self.data_type)

        if not isinstance(self.features, torch.Tensor):
            self._post_init_raw()
        
        if self.device.type != self.backend.device.type:
            self.features = self.features.to(self.device)

            if self.target is not None:
                self.target = self.target.to(self.device)


    def _post_init_raw(self):
        """
        preprocessing for torch export"""

        self.target_idx = self.backend.convert_idx_to_array(self.target_idx)
        self.categorical_idx = self.backend.convert_idx_to_array(self.categorical_idx)
        self.text_idx = self.backend.convert_idx_to_array(self.text_idx)
        self.encoded_idx = self.backend.convert_idx_to_array(self.encoded_idx)
        self.features_names = self.backend.convert_idx_to_array(self.features_names)
        self.ts_terms_idx = self.backend.convert_idx_to_array(self.ts_terms_idx)

        try:
            self.features = self.backend.xp.array(self.features)
        except:
            raise ValueError(f"Fedot preprocessing doesn't support categorical data in gpu mode")

        self.features = self.backend.replace_missing_with_nan(self.features)

        if self.data_type == DataTypesEnum.ts:
            ts_preproccessor = TSDataConverter(self.backend.xp)
            self.features, self.target, self.ts_init_shape, self.ts_terms_idx = ts_preproccessor.process_ts_data(self.features,
                                                        self.target,
                                                        self.features_names,
                                                        self.state,
                                                        self.ts_orientation,
                                                        self.ts_terms_idx,
                                                        self.ts_forecast_horizon)
        if self.target is not None:
            self.target = self.backend.xp.array(self.target)
            self.target = self.backend.atleast_n_dimensions(self.target, 2)
            self.target = self.backend.encode_target(self.target)
            self.target = self.backend.replace_missing_with_nan(self.target)
        elif (self.data_type != DataTypesEnum.ts) and (self.state == 'fit'):
            self.features, self.target, self.target_idx = self.backend.get_target_and_features(self.features,
                                                                self.features_names,
                                                                self.target_idx)

        if self.target is not None:
            self.features, self.target = self.backend._drop_rows_with_nan_in_target(self.features,
                                                    self.target)

        # check for text features
        if self.text_idx is None and self.backend.name == 'cpu':
            self.text_idx = self.backend.get_text_column_indices(self.features,
                                                        self.text_idx)

        # encode categorical features
        if self.categorical_idx is not None:
            self.categorical_idx = self.backend.get_idx_from_features_names(self.categorical_idx, self.features_names)

            # if categorical features are recognized like text
            if self.text_idx is not None and not set(self.categorical_idx).isdisjoint(set(self.text_idx)):
                self.text_idx = self.backend.xp.setdiff1d(self.text_idx, self.categorical_idx)

        self.features, self.categorical_idx = self.backend.encode_categorical_features(
                    self.features, self.categorical_idx, self.text_idx
            )
        if self.categorical_idx is not None:
            self.categorical_features = self.backend.to_tensor(
                self.features[:, self.categorical_idx], dtype=torch.float32
            )
        else:
            self.categorical_features = None
        
        # get embeddings for text features
        if self.text_idx is not None:
            text_columns = self.features[:, self.text_idx]
            text_tensors = encode_text_columns_np(text_columns, 
                                                  self.embedder_name, 
                                                  self.embedder_batch_size, 
                                                  self.backend.device)
            self.features = self.backend.xp.delete(self.features, self.text_idx, axis=1)

        # convert to tensor
        if self.features.shape[1] != 0:
            self.features = self.backend.to_tensor(self.features, dtype=torch.float32)
            if self.text_idx is not None:
                self.features = torch.cat((self.features, text_tensors), dim=1)
        else:
            self.features = text_tensors
            self.data_type = DataTypesEnum.text

        if self.task.task_type is TaskTypesEnum.ts_forecasting:
            self.target = self.features.copy()
        elif self.target is not None:
            self.target = self.backend.to_tensor(self.target, dtype=torch.float32)
        
        self.idx = torch.arange(self.features.shape[1], device=self.backend.device)
        self.target_idx = self.backend.to_tensor(self.target_idx, dtype=torch.int32)
        self.categorical_idx = self.backend.to_tensor(self.categorical_idx, dtype=torch.int32)
        self.text_idx = self.backend.to_tensor(self.text_idx, dtype=torch.int32)
        self.numerical_idx = self.backend.to_tensor(self.numerical_idx, dtype=torch.int32)
        self.encoded_idx = self.backend.to_tensor(self.encoded_idx, dtype=torch.int32)
        self.ts_terms_idx = self.backend.to_tensor(self.ts_terms_idx, dtype=torch.int32)

    @classmethod
    def _resolve_creator(cls, source_data: Any) -> Callable:
        for predicate, creator in cls._creators:
            result = predicate(source_data)

            if not isinstance(result, bool):
                raise TypeError(
                    f"Predicate {predicate.__name__} must return bool, got {type(result)}"
                )

            if result:
                return creator

        raise ValueError(f"No creator registered for input: {type(source_data)}")
    
    @classmethod
    def register_creator(cls, predicate: Callable[[Any], bool]) -> Callable[[Callable], Callable]:
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
            raise ValueError(f"Error creating TensorData: {e}")

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


@TensorData.register_creator(lambda x: isinstance(x, torch.Tensor))
def from_torch(features: torch.Tensor,
               target: Optional[torch.Tensor] = None,
               task: TaskType = Task(TaskTypesEnum.classification),
               state: str = 'fit',
               data_type: DataType = None,
               device: Optional[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               dataloader_kwargs = None,) -> TensorData:
    return TensorData(features=features, target=target, task=task, state=state,
                      data_type=data_type, device=device,
                      dataloader_kwargs=dataloader_kwargs)


@TensorData.register_creator(
    lambda x: isinstance(x, np.ndarray) or isinstance(x, cp.ndarray)
)
def from_numpy(features: np.ndarray,
               target: Optional[np.ndarray] = None,
               task: TaskType = Task(TaskTypesEnum.classification),
               state: str = 'fit',
               data_type: DataType = None,
               features_names: IndexType = None,
               target_idx: IndexType = None,
               categorical_idx: IndexType = None,
               text_idx: IndexType = None,
               embedder_name: Optional[str] = None,
               embedder_batch_size: int = 16,
               device: Optional[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               dataloader_kwargs = None,
               backend_name: Optional[str] = "cpu") -> TensorData:

    backend = BackendDefiner(backend_name)

    if data_type is None:
        data_type = autodetect_data_type(task)

    if isinstance(target, int) and target < features.shape[1]:
        target_idx = target.copy()
        target = None
    
    data = TensorData(
        features=features,
        target=target,
        features_names=features_names,
        target_idx=target_idx,
        categorical_idx=categorical_idx,
        text_idx=text_idx,
        task=task,
        state=state,
        data_type=data_type,
        embedder_name=embedder_name,
        embedder_batch_size=embedder_batch_size,
        device=device,
        backend = backend,
        dataloader_kwargs=dataloader_kwargs,
    )

    return data


@TensorData.register_creator(
    lambda x: isinstance(x, pd.DataFrame) or isinstance(x, pd.Series) or isinstance(x, cudf.DataFrame)
)
def from_pandas(
    features: Union[pd.DataFrame, pd.Series],
    target: Optional[Union[pd.DataFrame, pd.Series]] = None,
    task: TaskType = Task(TaskTypesEnum.classification),
    state: str = 'fit',
    target_idx: IndexType = None,
    categorical_idx: IndexType = None,
    text_idx: IndexType = None,
    data_type: DataType = DataTypesEnum.table,
    embedder_name: Optional[str] = None,
    embedder_batch_size: int = 16,
    device: Optional[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    backend_name: Optional[str] = "cpu",
    dataloader_kwargs = None,
) -> TensorData:
    
    backend = BackendDefiner(backend_name)

    features_names = features.columns.to_numpy()
    
    features = backend.get_values_from_df(features)

    if target is not None:
        target = backend.get_values_from_df(target)

    return TensorData(
        features=features,
        target=target,
        task=task,
        state=state,
        target_idx=target_idx,
        categorical_idx=categorical_idx,
        text_idx=text_idx,
        data_type=data_type,
        features_names=features_names,
        embedder_name=embedder_name,
        embedder_batch_size=embedder_batch_size,
        device=device,
        backend = backend,
        dataloader_kwargs=dataloader_kwargs
    )


@TensorData.register_creator(
    lambda x: isinstance(x, str) and (x.endswith(".csv") or x.endswith(".tsv"))
)
def from_csv_tsv(
    file_path: str,
    delimiter: str = ',',
    max_rows: Optional[int] = None,
    task: TaskType = Task(TaskTypesEnum.classification),
    state: str = 'fit',
    data_type: DataType = DataTypesEnum.table,
    columns_to_drop: IndexType = None,
    target_idx: IndexType = None,
    categorical_idx: IndexType = None,
    text_idx: IndexType = None,
    index_col: IndexType = None,
    possible_idx_keywords: Optional[List[str]] = None,
    embedder_name: Optional[str] = None,
    embedder_batch_size: int = 16,
    device: Optional[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    backend_name: Optional[str] = "cpu",
    dataloader_kwargs = None
) -> TensorData:
    
    backend = BackendDefiner(backend_name)

    if not is_existed_csv_path(file_path):
        raise ValueError(f'File {file_path} does not exist')

    possible_idx_keywords = (
        possible_idx_keywords or POSSIBLE_TABULAR_IDX_KEYWORDS
    )

    features = backend.get_df_from_csv(
        file_path=file_path,
        delimiter=delimiter,
        index_col=index_col,
        possible_idx_keywords=possible_idx_keywords,
        columns_to_drop=columns_to_drop,
        nrows=max_rows
    )
    shape = features.shape
    features_names = features.columns.to_numpy()

    # TODO: use cudf for get embeddings/encoding in gpu to resolve this
    features = backend.get_values_from_df(features)
    
    return TensorData(
        features=features,
        task=task,
        state=state,
        target_idx=target_idx,
        data_type=data_type,
        categorical_idx=categorical_idx,
        text_idx=text_idx,
        features_names=features_names,
        embedder_name=embedder_name,
        embedder_batch_size=embedder_batch_size,
        device=device,
        backend = backend,
        dataloader_kwargs=dataloader_kwargs
    )


@TensorData.register_creator(
    lambda x: isinstance(x, str) and x.endswith(".arff")
)
def from_arff(
    file_path: str,
    task: TaskType = TaskTypesEnum.ts_forecasting,
    state: str = 'fit',
    data_type: DataType = DataTypesEnum.ts,
    target_idx: IndexType = None,
    device: Optional[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    backend_name: Optional[str] = "cpu",
    dataloader_kwargs: Optional[Dict[str, Any]] = None
) -> TensorData:
    
    backend = BackendDefiner(backend_name)

    features, target = backend.read_arff_file(file_path, target_idx=target_idx)

    return TensorData(
        features=features,
        target=target,
        task=task,
        state=state,
        data_type=data_type,
        device=device,
        dataloader_kwargs=dataloader_kwargs
    )
