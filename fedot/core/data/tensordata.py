from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Callable, TypeAlias, ClassVar, List, Tuple
from contextlib import nullcontext
import logging

from pathlib import Path

import torch
import numpy as np
import cupy as cp
import pandas as pd
import cudf

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot.core.data.tools import StateEnum, TSOrientationEnum

from fedot.core.data.data_tools import (
    get_device_from_str, is_existed_csv_path, get_values_from_df, 
    convert_idx_to_list, replace_missing_with_nan, get_target_and_features,
    transform_to_tensor, _drop_rows_with_nan_in_target)

from fedot.core.data.data_reader import get_df_from_csv, read_arff_file
from fedot.preprocessing.ts_preprocessing import process_ts_data
from fedot.core.backend.backend import Backend, torch_to_xp
from fedot.core.data.data import autodetect_data_type
from fedot.core.data.complex_types import PathType, IndexType, PandasType, ArrayType
from fedot.preprocessing.planner_tools import (get_embedding_step, 
                                               get_encoding_steps, 
                                               get_target_encoding_step)

from fedot.preprocessing.obligatory_executor import apply_obligatory_steps
from fedot.preprocessing.preprocessing_tools import (create_index_mapping, 
                                                     update_indices,
                                                     agregate_idx_from_step)


logger = logging.getLogger(__name__)


POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']


class LazyTensor:
    """
    Lazy wrapper around tensor data creation.

    It stores a callable used to build :class:`TensorData` only when needed via :meth:`get`,
    which can delay expensive preprocessing until consumption.

    e.g. 
        lazy_td = TensorData.create_lazy(X, backend_name="cpu")
        ...
        # and then we can materialize the tensor data
        td = lazy_td.get()
    """
    def __init__(self, create_fn):
        """
        Args:
            create_fn (Callable[[], TensorData]): Factory function to create `TensorData`.
        """
        self._create_fn = create_fn
        self._data = None

    def get(self) -> "TensorData":
        """
        Materialize and return the underlying :class:`TensorData`.

        Returns:
            TensorData: Created tensor data.
        """
        if self._data is None:
            self._data = self._create_fn()
        return self._data

    def to(self, device: str):
        """
        Move the underlying :class:`TensorData` to the given device.

        Args:
            device (str): Target device (e.g. `"cpu"`, `"cuda"`).

        Returns:
            TensorData: TensorData moved to the requested device.
        """
        data = self.get()
        return data.to(device)

    def __repr__(self):
        """
        Returns:
            str: Debug representation including initialization state.
        """
        return f"LazyTensor(initialized={self._data is not None})"


TensorLike: TypeAlias = Optional[Union[torch.Tensor, np.ndarray, LazyTensor, cp.ndarray]]


@dataclass
class LoadDataSpec:
    """
    Specification used to construct :class:`TensorData` from various input sources.

    Attributes:
        task (Optional[Union[Task, str]]): Task descriptor (e.g. classification/regression)
            used by preprocessing to decide target handling and output expectations.
        data_type (Optional[Union[DataTypesEnum, str]]): Dataset type
            ("table" vs "time_series").
        state (Union[str, StateEnum]): Preprocessing state, typically `fit` or `predict`.

        target (TensorLike): Optional target data (already extracted by the creator or will be 
        extracted automatically).
        predict (TensorLike): Optional prediction tensor provided externally.

        target_idx (IndexType): Target column index or name (for FIT mode).

        categorical_idx (IndexType): Indices/names of categorical feature columns.
        encoding_strategy (Optional[Dict]): Categorical encoding strategy.
            A dict describing per-column strategies.
            e.g. {"label": ["column1", "column2"],
                  "ohe": ["column3"]}

        text_idx (IndexType): Indices/names of text feature columns to embed.
        embedding_strategy (Optional[Union[Dict]]): Configuration for the text embedding method.
            e.g.    text_idx = ["column1", "column2"]
                    embedding_strategy= {
                        "method": "sentence_transformer",
                        "model_name": "all-distilroberta-v1",
                        "batch_size": 3,
                        "device": "cpu",
                    }
        features_names (IndexType): Names of all feature columns (used to resolve indices from strings). 
            May be automatically extracted by the creator.

        ts_orientation (Union[TSOrientationEnum, str]): Time-series orientation. ("long" or "wide").
        ts_terms_idx (IndexType): Indice of the column with terms for "long" orientation.
        ts_forecast_horizon (Optional[int]): Forecast horizon for time-series tasks.

        dataloader_kwargs (Dict[str, Any]): Parameters passed to the future dataloader creation
            (batch size, shuffle, num_workers, drop_last).
        delimiter (str): CSV/TSV delimiter used by file-based creators.
        max_rows (Optional[int]): Optional limit on number of rows to read from files.
        columns_to_drop (IndexType): Columns to drop while loading tabular files.
        index_col (IndexType): Optional explicit column to use as index in tabular files.
        possible_idx_keywords (Optional[List[str]]): Keywords used to auto-detect an index column
            when `index_col` is not provided.
    """

    task: Optional[Union[Task, str]] = Task(TaskTypesEnum.classification)
    data_type: Optional[Union[DataTypesEnum, str]] = DataTypesEnum.tabular

    state: Union[str, StateEnum] = StateEnum.FIT

    target: TensorLike = None
    target_idx: IndexType = None
    categorical_idx: IndexType = field(default_factory=list)
    encoding_strategy: Optional[Dict] = None
    embedding_strategy: Optional[Union[Dict]] = None
    features_names: IndexType = None

    ts_orientation: Union[TSOrientationEnum, str] = None
    ts_terms_idx: IndexType = None
    ts_forecast_horizon: Optional[int] = None

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
        """
        Build a :class:`TensorData` instance from preprocessed `features`.

        Args:
            features: Preprocessed feature array/tensor created by a registered creator.

        Returns:
            TensorData: TensorData populated with fields from this spec.
        """
        return TensorData(
            features=features,
            target=self.target,
            task=self.task,
            data_type=self.data_type,
            state=self.state,
            features_names=self.features_names,
            target_idx=self.target_idx,
            categorical_idx=self.categorical_idx,
            encoding_strategy=self.encoding_strategy,
            embedding_strategy=self.embedding_strategy,
            ts_orientation=self.ts_orientation,
            ts_terms_idx=self.ts_terms_idx,
            ts_forecast_horizon=self.ts_forecast_horizon,
            dataloader_kwargs=self.dataloader_kwargs,
        )


@dataclass
class TensorData:
    """
    Unified tensor-based data container for node-to-node communication.

    TensorData normalizes different input formats (torch tensors, numpy/cupy arrays,
    pandas/cudf dataframes, and file paths such as CSV/TSV and ARFF) into a consistent
    representation backed by `torch.Tensor`.

    It also performs preprocessing steps needed for modeling (e.g. target extraction,
    missing value handling, categorical encoding, and optional text embeddings).

    e.g. create TensorData from a CSV file:
        csv_path = 'path/to/csv/file.csv'

        td = TensorData.create(
            csv_path,
            backend_name="cpu",
            target_idx = "target"
        )

    e.g. add new way of creating TensorData:
        @TensorData.register_creator("you predicate")
        def new_way(source_data, spec: LoadDataSpec) -> TensorData:
            # way of reading data from source_data
            return spec.to_tensor_data(features)
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
    categorical_idx: IndexType = field(default_factory=list)
    numerical_idx: IndexType = field(default_factory=list)
    encoding_strategy: Optional[Union[str, Dict]] = None
    text_idx: IndexType = field(default_factory=list)
    embedding_strategy: Optional[Union[Dict]] = None
    features_names: IndexType = None
    idx_mapping: dict[int, int] = field(default_factory=dict)
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
        """
        Post-initialization pipeline. 

        This method:
        - converts `state`, `task`, and `data_type` from strings to enums,
        - if features is already a torch.Tensor, it runs the encoding strategy on it, if necessary,
        - converts `features` to the active backend (CPU/GPU) when needed,
        - runs the raw preprocessing steps required for converting everything to torch tensors,
        - ensures the final tensors are on `backend.device`.
        
        Attention: if backend was chosen to be GPU, but features contain str/object values, 
            it will be preprocessed for tensors creation on CPU. But finally, all tensors
            will be moved to GPU.
        """

        if isinstance(self.state, str):
            self.state = StateEnum(self.state)

        if isinstance(self.task, str):
            self.task = Task(TaskTypesEnum(self.task))

        if isinstance(self.data_type, str):
            self.data_type = DataTypesEnum(self.data_type)

        if isinstance(self.features, torch.Tensor):
            self.features = torch_to_xp(self.features, Backend().xp)
            self.target = torch_to_xp(self.target, Backend().xp)
            self._post_init_raw()

        else:
            ctx = nullcontext()
            try:
                self.features = Backend().xp.array(self.features)
            except:
                logger.info("Turning to cpu backend to get TensorData due to failed to convert features to cupy array")
                ctx = Backend().override("cpu")
                with Backend().override("cpu"):
                    self.features = Backend().xp.array(self.features, dtype=object)
            
            with ctx:
                self._post_init_raw()

        if self.features.device.type != Backend().device.type:
            self.to(Backend().device)

    def _post_init_raw(self):
        """
        Preprocessing steps required before exporting to `torch.Tensor`.

        It normalizes indices and missing values, applies time-series preprocessing
        (`process_ts_data`), extracts `(features, target)` (`get_target_and_features`),
        computes optional text embeddings, performs categorical encoding, and finally
        converts arrays to torch tensors (`transform_to_tensor`).
        """

        self.target_idx = convert_idx_to_list(self.target_idx)
        self.categorical_idx = convert_idx_to_list(self.categorical_idx)
        self.text_idx = convert_idx_to_list(self.text_idx)
        self.features_names = convert_idx_to_list(self.features_names)
        self.ts_terms_idx = convert_idx_to_list(self.ts_terms_idx)

        self.features = replace_missing_with_nan(self.features)

        self.idx_mapping = create_index_mapping(self.features)

        self.features, self.target, self.ts_init_shape, self.ts_terms_idx = process_ts_data(self.features,
                                                    self.target,
                                                    self.features_names,
                                                    self.state,
                                                    self.ts_orientation,
                                                    self.ts_terms_idx,
                                                    self.ts_forecast_horizon,
                                                    self.data_type)

        self.features, self.target, self.idx_mapping = get_target_and_features(self.features,
                                                            self.target,
                                                            self.features_names,
                                                            self.target_idx,
                                                            self.state,
                                                            self.data_type,
                                                            self.idx_mapping)

        # target encoding
        target_encoding_step = get_target_encoding_step(self.target)
        if target_encoding_step is not None:
            self.target, target_encoding_step, _ = apply_obligatory_steps(self.target, target_encoding_step)
            self.target = Backend().xp.asarray(self.target, dtype=Backend().xp.float32)
        
        self.features, self.target = _drop_rows_with_nan_in_target(self.features, self.target)

        # get embeddings
        embedding_step = get_embedding_step(self.embedding_strategy, self.features_names, self.idx_mapping)
        if embedding_step is not None:
            self.text_idx = agregate_idx_from_step(embedding_step)
            self.features, embedding_step, self.idx_mapping = apply_obligatory_steps(self.features, embedding_step, self.idx_mapping)
        else:
            embedding_step = None

        # encoding categorical features
        encoding_steps = get_encoding_steps(self.encoding_strategy, self.features, self.features_names, self.idx_mapping)
        if encoding_steps is not None:
            encoding_idx = agregate_idx_from_step(encoding_steps)
            self.features, encoding_steps, self.idx_mapping = apply_obligatory_steps(self.features, encoding_steps, self.idx_mapping)
        else:
            encoding_steps = None
            encoding_idx = []
            # TODO: how to save steps?      

        self.features, self.target = transform_to_tensor(self.features, 
                                                         self.target,
                                                         self.ts_init_shape)
        
        self.idx = torch.arange(self.features.shape[1], dtype=torch.int32)

        if embedding_step is not None:
            self.text_idx = update_indices(self.idx_mapping, self.text_idx)

        if encoding_steps is not None:
            encoding_idx = update_indices(self.idx_mapping, encoding_idx)
            if len(self.categorical_idx) == 0:
                self.categorical_idx = encoding_idx
            else:
                self.categorical_idx = update_indices(self.idx_mapping, self.categorical_idx)
                self.categorical_idx = list(set(self.categorical_idx.extend(encoding_idx)))
        else:
            if len(self.categorical_idx) != 0:
                self.categorical_idx = update_indices(self.idx_mapping, self.categorical_idx) 

        self.numerical_idx = list(set(range(self.features.shape[1])) - set(self.categorical_idx) - set(self.text_idx))


    @classmethod
    def _resolve_creator(cls, source_data: Any) -> Callable:
        """
        Resolve the appropriate creator function for a given `source_data`.

        Registered creators are checked in the order they were added.

        Args:
            source_data (Any): Input data to be handled.

        Returns:
            Callable: Creator function that accepts `(source_data, spec)`.

        Raises:
            ValueError: If no creator matches the input.
            TypeError: If a predicate returns a non-boolean value.
        """
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
        """
        Register a new creator for :class:`TensorData`.

        Args:
            predicate (Callable[[Any], bool]): Function that returns True if the creator can
                handle the given input.

        Returns:
            Callable[[Callable], Callable]: Decorator that registers the creator function.
        """
        def decorator(func):
            cls._creators.append((predicate, func))
            return func
        return decorator

    @classmethod
    def create(cls, source_data, backend_name, **kwargs):
        """
        Eagerly create a :class:`TensorData` instance.

        Args:
            source_data (Any): Raw input (tensor, array, dataframe, or a file path).
            backend_name (str): Backend name passed to `backend.set` (e.g. `"cpu"`, `"gpu"`).
            **kwargs: Arguments forwarded to :class:`LoadDataSpec`.

        Returns:
            TensorData: Materialized tensor data object.
        """

        Backend().set(backend_name)        

        spec = LoadDataSpec(**kwargs)

        try:
            creator = cls._resolve_creator(source_data)
            return creator(source_data, spec)
        except Exception as e:
            raise ValueError(f"Error creating TensorData") from e

    @classmethod
    def create_lazy(cls, source_data, backend_name, **kwargs):
        """
        Lazily create a :class:`TensorData` instance.

        Args:
            source_data (Any): Raw input to be handled by a registered creator.
            backend_name (str): Backend name passed to `backend.set`.
            **kwargs: Arguments forwarded to :class:`LoadDataSpec`.

        Returns:
            LazyTensor: Lazy wrapper that builds `TensorData` on demand.
        """

        Backend().set(backend_name)

        spec = LoadDataSpec(**kwargs)

        creator = cls._resolve_creator(source_data)

        def _create():
            return creator(source_data, spec)

        return LazyTensor(_create)

    def to(self, device: Union[str, torch.device]):
        """
        Move internal tensors to the given device.

        Args:
            device (Union[str, torch.device]): Target device.

        Returns:
            TensorData: `self` moved to the requested device.
        """
        device = get_device_from_str(device)

        if isinstance(self.features, LazyTensor):
            self.features = self.features.get().to(device)
        elif isinstance(self.features, torch.Tensor):
            self.features = self.features.to(device)

        if self.target is not None:
            self.target = self.target.to(device)

        return self

    def save_predict(self, path_to_save: PathType) -> PathType:
        """
        Save `self.predict` to a CSV file.

        Args:
            path_to_save (PathType): Destination path.

        Returns:
            PathType: Resolved path to the written file (fallback may be used).
        """
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
        """
        Save features (and optionally target) to a CSV file.

        Args:
            path_to_save (PathType): Destination path.

        Returns:
            PathType: Resolved path to the written file.
        """
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
        Estimate memory usage of `features` in bytes.

        For `torch.Tensor` this is `element_size * number_of_elements`.

        Returns:
            int: Memory usage in bytes (or `0` when unavailable).
        """
        if isinstance(self.features, torch.Tensor):
            return self.features.element_size() * self.features.nelement()
        else:
            logger.warning("Memory usage is not available for non-torch tensors.")
            return 0


@TensorData.register_creator(lambda x: isinstance(x, torch.Tensor))
def from_torch(features: torch.Tensor, spec: LoadDataSpec) -> TensorData:
    """
    Creator for :class:`TensorData` when the input is already a `torch.Tensor`.

    Args:
        features (torch.Tensor): Input features.
        spec (LoadDataSpec): Creation specification.

    Returns:
        TensorData: TensorData created from the provided tensor.
    """
    return spec.to_tensor_data(features)


@TensorData.register_creator(
    lambda x: isinstance(x, np.ndarray) or isinstance(x, cp.ndarray)
)
def from_numpy(features: ArrayType, spec: LoadDataSpec) -> TensorData:
    """
    Creator for :class:`TensorData` when the input is a numpy/cupy array.

    Args:
        features (ArrayType): Input features array.
        spec (LoadDataSpec): Creation specification.

    Returns:
        TensorData: TensorData created from the provided array.
    """
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
    """
    Creator for :class:`TensorData` when the input is a pandas/cudf dataframe or series.

    Args:
        features (PandasType): Input dataframe/series.
        spec (LoadDataSpec): Creation specification.

    Returns:
        TensorData: TensorData created from extracted values.
    """

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
    """
    Creator for :class:`TensorData` when the input is a `.csv` or `.tsv` file path.

    Args:
        file_path (str): Path to the CSV/TSV file.
        spec (LoadDataSpec): Creation specification (delimiter, index/target settings, etc.).

    Returns:
        TensorData: TensorData created from values loaded from the file.
    """

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
    """
    Creator for :class:`TensorData` when the input is an `.arff` file path.

    Args:
        source (str): Path to the ARFF file.
        spec (LoadDataSpec): Creation specification (target index/name, etc.).

    Returns:
        TensorData: TensorData created from extracted ARFF features and target.
    """
    
    features, spec.target = read_arff_file(source, target_idx=spec.target_idx)

    return spec.to_tensor_data(features)
