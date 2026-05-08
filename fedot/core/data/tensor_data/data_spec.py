from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.common.enums import StateEnum, TSOrientationEnum
from fedot.core.data.tensor_data.rules import DEFAULT_DATALOADER_KWARGS, build_load_data_spec_normalization
from fedot.core.data.common.types import IndexType, TensorLike


@dataclass
class DataSpec:
    """
    Mutable specification used while creating `TensorData`.

    `DataSpec` carries user options and intermediate values through the
    `DataReader` and `TensorDataCreator` pipeline. Reader functions fill `features`,
    `target`, and `features_names`; preprocessing then normalizes task metadata,
    extracts targets, applies obligatory tabular/time-series preprocessing, and
    converts arrays to torch tensors.

    Attributes:
        task: FEDOT task or task name. Defaults to classification.
        data_type: FEDOT data type or alias. Common aliases include `"tabular"`
            and `"time_series"`.
        features: Raw or already-read feature data.
        state: Processing state, usually `StateEnum.FIT`.
        target: Optional target data. If omitted in fit mode, it can be extracted
            from `features` using `target_idx` or the default target convention.
        target_idx: Target column index/name or a list of indices/names.
        without_target: If True, target is not extracted from features.
        categorical_idx: Known categorical feature indices/names.
        numerical_idx: Known numerical feature indices. Filled during preprocessing.
        encoding_strategy: Optional categorical encoding strategy.
        embedding_strategy: Optional text embedding strategy or list of strategies.
        custom_strategy: Optional custom preprocessing strategy.
        features_names: Source feature names used to resolve string indices.
        ts_orientation: Time-series orientation, for example `"long"` or `"wide"`.
        ts_terms_idx: Column index/name with time-series terms for long format.
        ts_forecast_horizon: Optional forecast horizon for time-series tasks.
        ts_init_shape: Original time-series shape captured during preprocessing.
        predict: Optional externally provided predictions.
        idx: Optional sample index.
        idx_mapping: Mapping between original and preprocessed row indices.
        dataloader_kwargs: Dataloader options merged with defaults.
        delimiter: CSV/TSV delimiter for file inputs.
        max_rows: Optional row limit for file inputs.
        columns_to_drop: Columns to drop while loading tabular files.
        index_col: Explicit index column for tabular file inputs.
        possible_idx_keywords: Keywords used to auto-detect an index column when
            `index_col` is not provided.
    """

    task: Optional[Union[Task, str]] = Task(TaskTypesEnum.classification)
    data_type: Optional[Union[DataTypesEnum, str]] = DataTypesEnum.tabular

    features: TensorLike = None
    state: Union[str, StateEnum] = StateEnum.FIT
    target: TensorLike = None
    target_idx: Optional[IndexType] = None
    without_target: bool = False
    categorical_idx: IndexType = field(default_factory=list)
    numerical_idx: IndexType = field(default_factory=list)
    encoding_strategy: Optional[Dict] = None
    embedding_strategy: Optional[Union[Dict]] = None
    custom_strategy: Optional[Dict] = None
    features_names: IndexType = None

    ts_orientation: Union[TSOrientationEnum, str] = None
    ts_terms_idx: IndexType = None
    ts_forecast_horizon: Optional[int] = None
    ts_init_shape: Optional[Tuple[int]] = None

    predict: TensorLike = None
    idx: IndexType = None
    idx_mapping: dict[int, int] = field(default_factory=dict)

    dataloader_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_DATALOADER_KWARGS))

    delimiter: str = ','
    max_rows: Optional[int] = None
    columns_to_drop: IndexType = None
    index_col: IndexType = None
    possible_idx_keywords: Optional[List[str]] = None

    def __post_init__(self):
        """
        Normalize user-provided options after dataclass initialization.

        String aliases for task, data type, state, and time-series orientation are
        converted to FEDOT enum objects. Dataloader options are merged with default
        values and embedding strategy format is normalized. Index fields are also
        aligned with their default semantics: optional references use `None`, while
        index collections use empty lists.
        """
        normalization = build_load_data_spec_normalization(
            task=self.task,
            data_type=self.data_type,
            state=self.state,
            ts_orientation=self.ts_orientation,
            embedding_strategy=self.embedding_strategy,
            dataloader_kwargs=self.dataloader_kwargs,
            target_idx=self.target_idx,
            categorical_idx=self.categorical_idx,
            numerical_idx=self.numerical_idx,
            ts_terms_idx=self.ts_terms_idx,
            features_names=self.features_names,
        )
        self.task = normalization.task
        self.state = normalization.state
        self.data_type = normalization.data_type
        self.ts_orientation = normalization.ts_orientation
        self.embedding_strategy = normalization.embedding_strategy
        self.dataloader_kwargs = normalization.dataloader_kwargs
        self.target_idx = normalization.target_idx
        self.categorical_idx = normalization.categorical_idx
        self.numerical_idx = normalization.numerical_idx
        self.ts_terms_idx = normalization.ts_terms_idx
        self.features_names = normalization.features_names
