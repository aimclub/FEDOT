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
from typing import Optional, Union, Dict, Any, Callable, TypeAlias

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot.core.data.tools import StateEnum, TSOrientationEnum

from fedot.core.data.data_tools import (
    get_device_from_str, is_existed_csv_path, get_values_from_df, 
    convert_idx_to_array, convert_to_list, replace_missing_with_nan, get_target_and_features,
    transform_to_tensor, choose_categorical_encoding, encode_torch_tensors,
    encode_categorical_features, get_text_embeddings)

from fedot.core.data.data_reader import get_df_from_csv, read_arff_file
from fedot.preprocessing.ts_preprocessing import process_ts_data
from fedot.core.backend.backend import backend
from fedot.core.data.data import (autodetect_data_type)
from fedot.core.data.complex_types import PathType, IndexType, PandasType, ArrayType

import logging


logger = logging.getLogger(__name__)


POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']


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


TensorLike: TypeAlias = Optional[Union[torch.Tensor, np.ndarray, LazyTensor, cp.ndarray]]


@dataclass
class LoadDataSpec:

    task: Optional[Union[Task, str]] = Task(TaskTypesEnum.classification)
    data_type: Optional[Union[DataTypesEnum, str]] = DataTypesEnum.tabular

    state: Union[str, StateEnum] = StateEnum.FIT

    idx: IndexType = None
    target: TensorLike = None
    predict: TensorLike = None
    target_idx: IndexType = None
    target_encoder: Any = None
    categorical_idx: IndexType = None
    encoding_strategy: Optional[Union[str, Dict]] = None
    text_idx: IndexType = None
    embedding_strategy: Optional[Union[Dict]] = field(default_factory=dict)
    features_names: IndexType = None

    ts_orientation: Union[TSOrientationEnum, str] = None
    ts_terms_idx: IndexType = None
    ts_forecast_horizon: Optional[int] = None
    ts_init_shape: Optional[int] = None

    dataloader_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False
    })

    delimiter: str = ','
    max_rows: Optional[int] = None
    columns_to_drop: IndexType = None
    index_col: IndexType = None
    possible_idx_keywords: Optional[List[str]] = None

    def to_tensor_data(self, features) -> "TensorData":
        return TensorData(
            features=features,
            target=self.target,
            task=self.task,
            data_type=self.data_type,
            state=self.state,
            idx=self.idx,
            predict=self.predict,
            features_names=self.features_names,
            target_idx=self.target_idx,
            target_encoder=self.target_encoder,
            categorical_idx=self.categorical_idx,
            encoding_strategy=self.encoding_strategy,
            text_idx=self.text_idx,
            embedding_strategy=self.embedding_strategy,
            ts_orientation=self.ts_orientation,
            ts_terms_idx=self.ts_terms_idx,
            ts_forecast_horizon=self.ts_forecast_horizon,
            ts_init_shape=self.ts_init_shape,
            dataloader_kwargs=self.dataloader_kwargs,
        )


@dataclass
class TensorData:
    """
    state: str ['fit', 'predict']
    """
    task: Union[Task, str]
    data_type: Union[DataTypesEnum, str]

    state: Union[str, StateEnum] = StateEnum.FIT
    idx: IndexType = None
    features: TensorLike = None
    target: TensorLike = None
    predict: TensorLike = None
    target_idx: IndexType = None
    target_encoder: Any = None
    categorical_idx: IndexType = None
    encoding_strategy: Optional[Union[str, Dict]] = None
    text_idx: IndexType = None
    embedding_strategy: Optional[Union[Dict]] = field(default_factory=dict)
    features_names: IndexType = None
    ts_orientation: Optional[str] = None
    ts_terms_idx: Optional[Union[int, str]] = None
    ts_forecast_horizon: Optional[int] = None
    ts_init_shape: Optional[int] = None

    dataloader_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False
    })

    _creators: ClassVar[List[Tuple[Callable, Callable]]] = []

    def __post_init__(self):

        if isinstance(self.state, str):
            self.state = StateEnum(self.state)

        if isinstance(self.task, str):
            self.task = Task(TaskTypesEnum(self.task))

        if isinstance(self.data_type, str):
            self.data_type = DataTypesEnum(self.data_type)

        # TODO: if TS and tensors
        if isinstance(self.features, torch.Tensor):
            self.encoding_strategy = encode_torch_tensors(
                self.features, 
                self.encoding_strategy,
                self.categorical_idx, 
                self.state, 
                self.features_names
            )
        else:
            try:
                self.features = backend.xp.array(self.features)
                self._post_init_raw()
            except:
                with backend.override("cpu"):
                    logger.info("Turning to cpu backend to get TensorData due to failed to convert features to cupy array")
                    self.features = backend.xp.array(self.features)
                    self._post_init_raw()
        
        if self.features.device.type != backend.device.type:
            self.to(backend.device)

    def _post_init_raw(self):
        """
        preprocessing for torch export
        """

        self.target_idx = convert_idx_to_array(self.target_idx)
        self.categorical_idx = convert_idx_to_array(self.categorical_idx)
        self.text_idx = convert_idx_to_array(self.text_idx)
        self.features_names = convert_to_list(self.features_names)
        self.ts_terms_idx = convert_idx_to_array(self.ts_terms_idx)

        self.features = replace_missing_with_nan(self.features)

        self.features, self.target, self.ts_init_shape, self.ts_terms_idx = process_ts_data(self.features,
                                                    self.target,
                                                    self.features_names,
                                                    self.state,
                                                    self.ts_orientation,
                                                    self.ts_terms_idx,
                                                    self.ts_forecast_horizon,
                                                    self.data_type)

        self.features, self.target, self.target_encoder = get_target_and_features(self.features,
                                                            self.target,
                                                            self.features_names,
                                                            self.target_idx,
                                                            self.state,
                                                            self.data_type)

        # get embeddings
        text_tensors, self.text_idx, self.features = get_text_embeddings(self.features, 
                                                        self.text_idx, 
                                                        self.embedding_strategy,
                                                        self.features_names)

        # encoding categorical features
        encoding_decisions, non_cat_features = choose_categorical_encoding(
            self.features, self.categorical_idx, self.encoding_strategy, 
            self.features_names, self.state
        )
        self.features, self.encoding_strategy = encode_categorical_features(
            self.features, encoding_decisions, non_cat_features
        )

        self.features, self.target = transform_to_tensor(self.features, 
                                                         self.target,
                                                         text_tensors,
                                                         self.text_idx,
                                                         self.ts_init_shape)
        
        self.idx = torch.arange(self.features.shape[1], dtype=torch.int32)


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
    def create(cls, source_data, backend_name, **kwargs):

        backend.set(backend_name)        

        spec = LoadDataSpec(**kwargs)

        try:
            creator = cls._resolve_creator(source_data)
            return creator(source_data, spec)
        except Exception as e:
            raise ValueError(f"Error creating TensorData") from e

    @classmethod
    def create_lazy(cls, source_data, backend_name, **kwargs):

        backend.set(backend_name)

        spec = LoadDataSpec(**kwargs)

        creator = cls._resolve_creator(source_data)

        def _create():
            return creator(source_data, spec)

        return LazyTensor(_create)

    def to(self, device: Union[str, torch.device]):
        device = get_device_from_str(device)

        if isinstance(self.features, LazyTensor):
            self.features = self.features.get().to(device)
        elif isinstance(self.features, torch.Tensor):
            self.features = self.features.to(device)

        if self.target is not None:
            self.target = self.target.to(device)

        return self

    def save_predict(self, path_to_save: PathType) -> PathType:
        prediction = self.predict.detach().cpu().tolist()
        prediction_df = pd.DataFrame({'Index': self.idx, 'Prediction': prediction})
        try:
            prediction_df.to_csv(path_to_save, index=False)
            logger.info("Predictions saved to %s", path_to_save.resolve())
            return path_to_save.resolve()

        except (FileNotFoundError, PermissionError, OSError) as exc:
            fallback_path = Path("./predictions.csv")

            logger.warning(
                "Failed to save predictions to %s: %s. "
                "Trying fallback path: %s",
                path_to_save,
                exc,
                fallback_path.resolve(),
            )

            try:
                prediction_df.to_csv(fallback_path, index=False)
                logger.info(
                    "Predictions saved to fallback path %s",
                    fallback_path.resolve(),
                )
                return fallback_path.resolve()

            except (FileNotFoundError, PermissionError, OSError) as fallback_exc:
                logger.exception(
                    "Failed to save predictions to both %s and fallback path %s",
                    path_to_save,
                    fallback_path,
                )
                raise RuntimeError(
                    f"Could not save predictions to '{path_to_save}' "
                    f"or fallback path '{fallback_path}'"
                ) from fallback_exc
    
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

        Returns:
            int: The memory usage in bytes.
        """
        if isinstance(self.features, torch.Tensor):
            return self.features.element_size() * self.features.nelement()
        else:
            logger.warning("Memory usage is not available for non-torch tensors.")
            return 0


@TensorData.register_creator(lambda x: isinstance(x, torch.Tensor))
def from_torch(features: torch.Tensor, spec: LoadDataSpec) -> TensorData:
    
    return spec.to_tensor_data(features)


@TensorData.register_creator(
    lambda x: isinstance(x, np.ndarray) or isinstance(x, cp.ndarray)
)
def from_numpy(features: ArrayType, spec: LoadDataSpec) -> TensorData:
    
    if spec.data_type is None:
        spec.data_type = autodetect_data_type(spec.task)

    if isinstance(spec.target, int) and spec.target < features.shape[1]:
        spec.target_idx = spec.target.copy()
        spec.target = None
    
    return spec.to_tensor_data(features)


@TensorData.register_creator(
    lambda x: isinstance(x, pd.DataFrame) or isinstance(x, pd.Series) or isinstance(x, cudf.DataFrame)
)
def from_pandas(
    features: PandasType, 
    spec: LoadDataSpec) -> TensorData:

    spec.features_names = features.columns.to_numpy()

    features = get_values_from_df(features)

    if spec.target is not None:
        spec.target = get_values_from_df(spec.target)

    return spec.to_tensor_data(features)


@TensorData.register_creator(
    lambda x: isinstance(x, str) and (x.endswith(".csv") or x.endswith(".tsv"))
)
def from_csv_tsv(
    file_path: str, spec: LoadDataSpec
) -> TensorData:

    if not is_existed_csv_path(file_path):
        raise ValueError(f'File {file_path} does not exist')

    spec.possible_idx_keywords = (
        spec.possible_idx_keywords or POSSIBLE_TABULAR_IDX_KEYWORDS
    )

    features = get_df_from_csv(
        file_path=file_path,
        delimiter=spec.delimiter,
        index_col=spec.index_col,
        possible_idx_keywords=spec.possible_idx_keywords,
        columns_to_drop=spec.columns_to_drop,
        nrows=spec.max_rows
    )

    spec.features_names = features.columns.to_numpy()

    features = get_values_from_df(features)
    
    return spec.to_tensor_data(features)


@TensorData.register_creator(
    lambda x: isinstance(x, str) and x.endswith(".arff")
)
def from_arff(
    source: str, spec: LoadDataSpec
) -> TensorData:
    
    features, spec.target = read_arff_file(source, target_idx=spec.target_idx)

    return spec.to_tensor_data(features)
